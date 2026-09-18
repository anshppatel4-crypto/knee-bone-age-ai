"""Ridge regression on raw voxels: the floor any trained model must clear.

The mean-age baseline says whether a model learned anything at all. This says
something sharper -- whether the network is worth its weights. It fits a linear
model to the raw preprocessed voxels, with no convolutions, no augmentation and
no pretraining, so whatever it scores is obtainable from the pixels by the
simplest thing that could work.

A 3D ResNet that cannot beat this is not limited by data or by resolution. It is
being let down by its training setup.

Because the design matrix is ~10^6 features wide and only a few hundred scans
tall, every fit is solved in the dual: the Gram matrix is computed once and each
fold is then algebra on a small square array.
"""
import argparse
import json
import os
import sys
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.dataset import KneeVolumeDataset
from src.preprocess import DEFAULT_INPUT_SHAPE
from src.train import DEFAULT_DATA_PATTERNS, build_catalog, regression_metrics

ALPHAS = np.logspace(0, 6, 13)


def _centred_gram(gram, train_idx, rows):
    """<x_i - mu_train, x_j - mu_train> for i in rows, j in train, from raw inner products."""
    block = gram[np.ix_(train_idx, train_idx)]
    per_column = block.mean(axis=0)
    per_row = gram[np.ix_(rows, train_idx)].mean(axis=1)
    return gram[np.ix_(rows, train_idx)] - per_row[:, None] - per_column[None, :] + block.mean()


def _fit_predict(gram, targets, train_idx, test_idx, alpha):
    centre = targets[train_idx].mean()
    kernel = _centred_gram(gram, train_idx, train_idx)
    weights = np.linalg.solve(kernel + alpha * np.eye(len(train_idx)), targets[train_idx] - centre)
    return _centred_gram(gram, train_idx, test_idx) @ weights + centre


def probe(gram, targets, folds=5, seed=0):
    """Cross-validated predictions; alpha is chosen inside the training part of each fold."""
    order = np.random.default_rng(seed).permutation(len(targets))
    predictions = np.zeros(len(targets))

    for fold in range(folds):
        test_idx = order[fold::folds]
        train_idx = np.setdiff1d(order, test_idx)

        best_alpha, best_error = ALPHAS[0], np.inf
        for alpha in ALPHAS:
            errors = []
            for inner in range(4):
                inner_test = train_idx[inner::4]
                inner_train = np.setdiff1d(train_idx, inner_test)
                predicted = _fit_predict(gram, targets, inner_train, inner_test, alpha)
                errors.append(np.mean(np.abs(predicted - targets[inner_test])))
            if np.mean(errors) < best_error:
                best_error, best_alpha = np.mean(errors), alpha

        predictions[test_idx] = _fit_predict(gram, targets, train_idx, test_idx, best_alpha)

    return predictions


def run(data_patterns=DEFAULT_DATA_PATTERNS, input_shape=DEFAULT_INPUT_SHAPE, limit=None, seed=0,
        out=None):
    catalog = build_catalog(data_patterns)
    if limit:
        catalog = catalog.sample(n=min(limit, len(catalog)), random_state=seed).reset_index(drop=True)

    dataset = KneeVolumeDataset(catalog, input_shape=tuple(input_shape), augment=False)
    loader = DataLoader(dataset, batch_size=8)
    volumes, ages, sexes = [], [], []
    for batch in loader:
        volumes.append(batch["image"].numpy().reshape(len(batch["bone_age"]), -1))
        ages.append(batch["bone_age"].numpy())
        sexes.append(batch["sex"].numpy())

    features = np.concatenate(volumes).astype(np.float64)
    ages = np.concatenate(ages).astype(np.float64)
    features = np.column_stack([features, np.concatenate(sexes)])

    print(f"{len(ages)} scans at {tuple(input_shape)} -> {features.shape[1]} features")
    predictions = probe(features @ features.T, ages, seed=seed)

    metrics = regression_metrics(predictions, ages)
    mean_age_mae = float(np.mean(np.abs(ages - ages.mean())))
    print(f"\n  mean-age baseline : MAE {mean_age_mae:.3f} y")
    print(f"  ridge on voxels   : MAE {metrics['mae']:.3f} y | slope {metrics['slope']:.3f} | "
          f"corr {metrics['corr']:.3f} | within 1y {metrics['within_1y']:.0%}")
    print(f"\nA trained model scoring worse than {metrics['mae']:.2f} y, or flatter than "
          f"slope {metrics['slope']:.2f},\nis losing to linear regression on raw pixels.")

    # Which fold a scan lands in moves MAE by a few hundredths of a year: two
    # implementations of this probe scored 1.453 and 1.517 on the same 60 phantoms.
    # It is a bar with a tolerance, not an exact threshold.
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump({"mae": metrics["mae"], "slope": metrics["slope"],
                       "mean_age_mae": mean_age_mae, "n": metrics["n"],
                       "input_shape": list(input_shape)}, handle, indent=2)
        print(f"floor written to {out}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear-probe floor for the knee bone age data")
    parser.add_argument("--data", nargs="+", default=DEFAULT_DATA_PATTERNS)
    parser.add_argument("--input-shape", type=int, nargs=3, default=list(DEFAULT_INPUT_SHAPE))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="Write the floor to this JSON file")
    args = parser.parse_args()
    run(args.data, tuple(args.input_shape), args.limit, args.seed, args.out)
