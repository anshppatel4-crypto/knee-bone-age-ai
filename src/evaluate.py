"""Score a trained checkpoint on a catalog of scans, or on one scan folder."""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.dataset import KneeVolumeDataset
from src.model import load_checkpoint
from src.preprocess import DEFAULT_INPUT_SHAPE, load_series
from src.predict import predict_scan
from src.train import build_catalog, predict_loader, regression_metrics, split_catalog


def evaluate_catalog(weights, catalog, batch_size=4, tta=True):
    """Report accuracy on every scan in a catalog, overall and per age band."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_checkpoint(weights, device)
    input_shape = tuple(metadata.get("input_shape", DEFAULT_INPUT_SHAPE))

    dataset = KneeVolumeDataset(catalog, input_shape=input_shape, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size)
    predictions, targets = predict_loader(model, loader, device, tta=tta)

    overall = regression_metrics(predictions, targets)
    print(f"\nScans: {overall['n']} | MAE {overall['mae']:.3f}y | RMSE {overall['rmse']:.3f}y | "
          f"within 1y {overall['within_1y']:.0%} | bias {overall['bias']:+.3f}y | R² {overall['r2']:.3f}")

    print("\nPer age band:")
    bands = np.digitize(targets, [6, 9, 12, 15, 17])
    labels = ["<6", "6-9", "9-12", "12-15", "15-17", "17+"]
    for band in np.unique(bands):
        mask = bands == band
        print(f"  {labels[band]:>6}: n={mask.sum():3d}  MAE {np.mean(np.abs(predictions[mask] - targets[mask])):.3f}y")

    worst = np.argsort(-np.abs(predictions - targets))[:5]
    print("\nWorst predictions:")
    for index in worst:
        print(f"  true {targets[index]:5.2f}y → predicted {predictions[index]:5.2f}y "
              f"({predictions[index] - targets[index]:+.2f})")
    return overall


def evaluate_folder(weights, folder, sex):
    """Predict a single scan folder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_checkpoint(weights, device)
    volume = load_series(folder, tuple(metadata.get("input_shape", DEFAULT_INPUT_SHAPE)))
    result = predict_scan(model, volume, sex, device)
    print(f"{folder}: {result['bone_age']:.2f} ± {result['uncertainty']:.2f} years "
          f"(stage {result['stage']})")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a knee bone age checkpoint")
    parser.add_argument("--weights", default="final_knee_model_resnet34.pth")
    parser.add_argument("--catalog", help="CSV catalog; defaults to the held-out test split")
    parser.add_argument("--folder", help="Score a single DICOM folder instead")
    parser.add_argument("--sex", default="m", help="Sex for --folder mode")
    parser.add_argument("--seed", type=int, default=0, help="Split seed; must match training")
    args = parser.parse_args()

    if args.folder:
        evaluate_folder(args.weights, args.folder, args.sex)
    else:
        catalog = pd.read_csv(args.catalog) if args.catalog else split_catalog(build_catalog(), seed=args.seed)[2]
        evaluate_catalog(args.weights, catalog)
