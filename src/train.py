"""Train the 3D ResNet bone age model with honest, patient-level evaluation.

Every reported number comes from a held-out split that was never trained on, and
is compared against the trivial "always predict the mean age" baseline.
"""
import argparse
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.dataset import KneeVolumeDataset
from src.knee_phantom import growth_stage, skeletal_maturity
from src.model import KneeBoneAgeMultiTaskNet, save_checkpoint
from src.preprocess import DEFAULT_INPUT_SHAPE

DICOM_AGE_UNITS_PER_YEAR = {"D": 365.25, "W": 52.18, "M": 12.0, "Y": 1.0}
DEFAULT_DATA_PATTERNS = ["data/phantom_knees*", "data/synthetic_knee_dense", "data/cohort_pt_*"]


# ---------------------------------------------------------------------------
# Catalog assembly and splitting
# ---------------------------------------------------------------------------
def parse_dicom_age(age_string):
    """DICOM ages look like '150M' or '014Y'."""
    text = str(age_string).strip()
    return int(text[:-1]) / DICOM_AGE_UNITS_PER_YEAR[text[-1].upper()]


def build_catalog(patterns=DEFAULT_DATA_PATTERNS):
    """Collect labelled scan folders: phantom label files, or DICOM headers elsewhere."""
    frames, folders = [], []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            labels = os.path.join(path, "labels.csv")
            if os.path.exists(labels):
                frames.append(pd.read_csv(labels))
            elif glob.glob(os.path.join(path, "*.dcm")):
                folders.append(path)
            else:
                folders.extend(sorted(d for d in glob.glob(os.path.join(path, "*"))
                                      if glob.glob(os.path.join(d, "*.dcm"))))

    for folder in folders:
        files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
        header = pydicom.dcmread(files[0], force=True, stop_before_pixels=True)
        age = parse_dicom_age(header.PatientAge)
        sex = str(header.get("PatientSex", "M")).upper()
        frames.append(pd.DataFrame([{
            "folder_path": folder.replace("\\", "/"),
            "sex": 1.0 if sex.startswith("M") else 0.0,
            "bone_age": age,
            "growth_stage": growth_stage(skeletal_maturity(age, sex)),
        }]))

    if not frames:
        raise FileNotFoundError(f"No labelled scan folders found under: {patterns}")

    catalog = pd.concat(frames, ignore_index=True)
    return catalog.drop_duplicates(subset="folder_path").reset_index(drop=True)


def split_catalog(catalog, seed=0, val_fraction=0.15, test_fraction=0.15):
    """Split by scan, stratified across age bins so every split spans the age range."""
    rng = np.random.default_rng(seed)
    bins = np.digitize(catalog["bone_age"], np.arange(4, 19, 2))
    train_idx, val_idx, test_idx = [], [], []

    for value in np.unique(bins):
        members = np.flatnonzero(bins == value)
        rng.shuffle(members)
        n_test = max(1, int(round(len(members) * test_fraction))) if len(members) > 3 else 0
        n_val = max(1, int(round(len(members) * val_fraction))) if len(members) > 3 else 0
        test_idx += list(members[:n_test])
        val_idx += list(members[n_test:n_test + n_val])
        train_idx += list(members[n_test + n_val:])

    # Tiny cohorts (smoke runs) can leave a split empty; borrow from train so metrics still work
    for holdout in (val_idx, test_idx):
        while not holdout and len(train_idx) > 1:
            holdout.append(train_idx.pop())

    take = lambda idx: catalog.iloc[sorted(idx)].reset_index(drop=True)
    return take(train_idx), take(val_idx), take(test_idx)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(predictions, targets):
    """Bone age accuracy summary, in years."""
    predictions, targets = np.asarray(predictions, float), np.asarray(targets, float)
    errors = predictions - targets
    variance = float(np.var(targets))

    # Slope of predicted-on-true. A model that hedges toward the mean scores a
    # respectable MAE while landing well under 1.0 here, so it is reported
    # alongside MAE rather than left to be inferred from it.
    # The mean-age baseline predicts a constant, so guard both variances: slope is 0
    # there and correlation is genuinely undefined rather than merely awkward.
    if variance > 1e-8 and len(targets) > 2:
        slope = float(np.polyfit(targets, predictions, 1)[0])
        correlation = (float(np.corrcoef(targets, predictions)[0, 1])
                      if np.var(predictions) > 1e-8 else float("nan"))
    else:
        slope = correlation = float("nan")

    return {
        "n": int(len(targets)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "bias": float(np.mean(errors)),
        "within_1y": float(np.mean(np.abs(errors) <= 1.0)),
        "within_2y": float(np.mean(np.abs(errors) <= 2.0)),
        "r2": float(1.0 - np.mean(errors ** 2) / variance) if variance > 1e-8 else float("nan"),
        "slope": slope,
        "corr": correlation,
        "pred_range": [float(predictions.min()), float(predictions.max())],
        "true_range": [float(targets.min()), float(targets.max())],
    }


def seed_worker(worker_id):
    """Fresh augmentation stream per worker, per epoch.

    Workers are forked holding an identical copy of the dataset's RNG. PyTorch
    draws a new base seed each epoch, so deriving from it keeps the workers
    independent of each other and of the previous epoch.
    """
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "reseed"):
        info.dataset.reseed(torch.initial_seed() % (2 ** 32))


@torch.no_grad()
def predict_loader(model, loader, device, tta=True):
    """Predictions for a loader, optionally averaged with the left/right mirrored volume."""
    model.eval()
    predictions, targets = [], []

    for batch in loader:
        image, sex = batch["image"].to(device), batch["sex"].to(device)
        age, _ = model(image, sex)
        if tta:
            flipped, _ = model(torch.flip(image, dims=[2]), sex)
            age = (age + flipped) / 2
        predictions.append(age.cpu().numpy())
        targets.append(batch["bone_age"].numpy())

    return np.concatenate(predictions), np.concatenate(targets)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(data_patterns=DEFAULT_DATA_PATTERNS, output="final_knee_model_resnet34.pth", arch="resnet34",
          epochs=40, batch_size=4, lr=3e-4, input_shape=DEFAULT_INPUT_SHAPE, seed=0, limit=None,
          patience=8, stage_loss_weight=0.3, num_workers=0, enhance_synthetic=False):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Training on {device}")

    catalog = build_catalog(data_patterns)
    if limit:
        catalog = catalog.sample(n=min(limit, len(catalog)), random_state=seed).reset_index(drop=True)
    train_df, val_df, test_df = split_catalog(catalog, seed=seed)
    print(f"📚 {len(catalog)} scans → train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")

    common = dict(input_shape=input_shape, enhance_synthetic=enhance_synthetic)
    train_set = KneeVolumeDataset(train_df, augment=True, seed=seed, **common)
    val_set = KneeVolumeDataset(val_df, augment=False, **common)
    test_set = KneeVolumeDataset(test_df, augment=False, **common)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=False, worker_init_fn=seed_worker,
                              persistent_workers=num_workers > 0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    # The number every model must beat: always predict the training mean
    mean_age = float(train_df["bone_age"].mean())
    baseline = {
        "val": regression_metrics(np.full(len(val_df), mean_age), val_df["bone_age"].to_numpy()),
        "test": regression_metrics(np.full(len(test_df), mean_age), test_df["bone_age"].to_numpy()),
    }
    print(f"📏 Baseline (predict {mean_age:.1f}y): val MAE {baseline['val']['mae']:.2f} | "
          f"test MAE {baseline['test']['mae']:.2f}")

    model = KneeBoneAgeMultiTaskNet(arch=arch, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs,
                                                    steps_per_epoch=max(1, len(train_loader)), pct_start=0.25)
    age_loss = nn.L1Loss()
    stage_loss = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_mae, best_epoch, history = float("inf"), 0, []
    print("\n🚀 Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            image, sex = batch["image"].to(device), batch["sex"].to(device)
            target_age, target_stage = batch["bone_age"].to(device), batch["growth_stage"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                age, stage = model(image, sex)
                loss = age_loss(age, target_age) + stage_loss_weight * stage_loss(stage, target_stage)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item() * image.size(0)

        predictions, targets = predict_loader(model, val_loader, device, tta=False)
        val = regression_metrics(predictions, targets)
        history.append({"epoch": epoch, "train_loss": epoch_loss / max(1, len(train_set)), **val})
        print(f"📈 Epoch {epoch:02d}/{epochs} | loss {history[-1]['train_loss']:.3f} | "
              f"val MAE {val['mae']:.3f}y | within 1y {val['within_1y']:.0%}")

        if val["mae"] < best_mae - 1e-4:
            best_mae, best_epoch = val["mae"], epoch
            save_checkpoint(model, output, input_shape=list(input_shape), val_mae=best_mae,
                            epoch=epoch, mean_age=mean_age)
        elif epoch - best_epoch >= patience:
            print(f"⏹️ Early stop: validation MAE has not improved since epoch {best_epoch}")
            break

    # Final scoring uses the best checkpoint, never the last epoch
    model.load_state_dict(torch.load(output, map_location=device)["state_dict"])
    test_predictions, test_targets = predict_loader(model, test_loader, device, tta=True)
    test = regression_metrics(test_predictions, test_targets)

    print("\n" + "=" * 58)
    print(f" TEST (held out, n={test['n']}, best epoch {best_epoch})")
    print("=" * 58)
    print(f" MAE          : {test['mae']:.3f} years   (baseline {baseline['test']['mae']:.3f})")
    print(f" RMSE         : {test['rmse']:.3f} years")
    print(f" Within 1 year: {test['within_1y']:.0%}     Within 2 years: {test['within_2y']:.0%}")
    print(f" Bias         : {test['bias']:+.3f} years   R²: {test['r2']:.3f}")
    print(f" Slope        : {test['slope']:.3f}           corr: {test['corr']:.3f}")
    print(f" Pred range   : {test['pred_range'][0]:.1f}-{test['pred_range'][1]:.1f}y "
          f"(true {test['true_range'][0]:.1f}-{test['true_range'][1]:.1f}y)")
    print("=" * 58)

    metrics_path = os.path.splitext(output)[0] + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump({"test": test, "val_best_mae": best_mae, "baseline": baseline,
                   "history": history, "arch": arch, "input_shape": list(input_shape),
                   "n_scans": len(catalog)}, handle, indent=2)
    print(f"💾 Weights: {output}\n📊 Metrics: {metrics_path}")
    return test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the 3D ResNet knee bone age model")
    parser.add_argument("--data", nargs="+", default=DEFAULT_DATA_PATTERNS)
    parser.add_argument("--output", default="final_knee_model_resnet34.pth")
    parser.add_argument("--arch", default="resnet34", choices=["resnet18", "resnet34"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--input-shape", type=int, nargs=3, default=list(DEFAULT_INPUT_SHAPE))
    parser.add_argument("--limit", type=int, default=None, help="Use only N scans (quick smoke runs)")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(data_patterns=args.data, output=args.output, arch=args.arch, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, input_shape=tuple(args.input_shape),
          limit=args.limit, num_workers=args.workers, seed=args.seed)
