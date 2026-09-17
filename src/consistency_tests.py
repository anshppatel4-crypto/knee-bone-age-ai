"""Sanity checks that a checkpoint behaves sensibly: repeatable, noise-tolerant, order-aware."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from src.model import load_checkpoint
from src.predict import predict_scan
from src.preprocess import DEFAULT_INPUT_SHAPE, load_series


def _setup(weights, scan_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_checkpoint(weights, device)
    volume = load_series(scan_dir, tuple(metadata.get("input_shape", DEFAULT_INPUT_SHAPE)))
    return model, volume, device


def repeat_test(scan_dir, sex, weights):
    """Inference must be deterministic in eval mode."""
    print("\n=== REPEAT TEST ===")
    model, volume, device = _setup(weights, scan_dir)
    outputs = [predict_scan(model, volume, sex, device)["bone_age"] for _ in range(10)]
    print("Outputs:", [round(o, 4) for o in outputs])
    print("Range:", round(max(outputs) - min(outputs), 6))


def noise_test(scan_dir, sex, weights):
    """Small amounts of noise should barely move the prediction."""
    print("\n=== NOISE TEST ===")
    model, volume, device = _setup(weights, scan_dir)
    rng = np.random.default_rng(0)
    baseline = predict_scan(model, volume, sex, device)["bone_age"]
    print(f"Clean: {baseline:.4f}")

    for noise_level in [0.01, 0.05, 0.10]:
        noisy = volume + rng.normal(0, noise_level, volume.shape)
        age = predict_scan(model, noisy, sex, device)["bone_age"]
        print(f"Noise {noise_level}: {age:.4f}  (delta {age - baseline:+.4f})")


def shuffle_test(scan_dir, sex, weights):
    """Shuffling slices destroys 3D structure, so the prediction should change."""
    print("\n=== SLICE SHUFFLE TEST ===")
    model, volume, device = _setup(weights, scan_dir)
    rng = np.random.default_rng(0)

    shuffled = volume.copy()
    rng.shuffle(shuffled)
    print("Normal:  ", round(predict_scan(model, volume, sex, device)["bone_age"], 4))
    print("Shuffled:", round(predict_scan(model, shuffled, sex, device)["bone_age"], 4))


def sex_sensitivity_test(scan_dir, sex, weights):
    """Sex must matter: girls mature ~1.8 years ahead, so the same image means a
    younger child if female. A model that ignores the sex input has found a shortcut."""
    print()
    print("=== SEX SENSITIVITY TEST ===")
    model, volume, device = _setup(weights, scan_dir)

    male = predict_scan(model, volume, "m", device)["bone_age"]
    female = predict_scan(model, volume, "f", device)["bone_age"]
    print(f"As male:   {male:.4f}")
    print(f"As female: {female:.4f}")
    print(f"Delta (M-F): {male - female:+.4f} years  (expected roughly +1.5 to +2.0)")
    if abs(male - female) < 0.5:
        print("⚠️  Model is ignoring sex - it is likely reading age off some other cue.")


if __name__ == "__main__":
    scan = sys.argv[1] if len(sys.argv) > 1 else "data/imported_patient_scan"
    weights = sys.argv[2] if len(sys.argv) > 2 else "final_knee_model_resnet34.pth"
    sex = sys.argv[3] if len(sys.argv) > 3 else "F"

    repeat_test(scan, sex, weights)
    noise_test(scan, sex, weights)
    shuffle_test(scan, sex, weights)
    sex_sensitivity_test(scan, sex, weights)
