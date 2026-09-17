"""Inference for the 3D ResNet knee bone age model."""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.model import load_checkpoint
from src.preprocess import DEFAULT_INPUT_SHAPE, load_series

STAGE_NAMES = {
    0: "Stage 0 (Completely Open Epiphyseal Plates)",
    1: "Stage 1 (Initial Sclerosis Zone)",
    2: "Stage 2 (Partial Structural Fusion)",
    3: "Stage 3 (Complete Terminal Fusion)",
}


def load_and_sort_dicom_volume(dicom_dir, target_shape=DEFAULT_INPUT_SHAPE):
    """Read and preprocess a DICOM folder into the model's input grid."""
    return load_series(dicom_dir, target_shape)


@torch.no_grad()
def predict_scan(model, volume, biological_sex, device="cpu", tta=True):
    """Predict bone age and physeal stage for one preprocessed volume.

    Test-time augmentation over the left/right mirror both improves accuracy and
    gives a spread that is reported as uncertainty.
    """
    sex_value = 1.0 if str(biological_sex).lower() in ("m", "male", "1", "1.0") else 0.0
    image = torch.as_tensor(volume, dtype=torch.float32)[None, None].to(device)
    sex = torch.tensor([sex_value], dtype=torch.float32, device=device)

    views = [image, torch.flip(image, dims=[2])] if tta else [image]
    ages, stage_logits = [], []
    for view in views:
        age, stage = model(view, sex)
        ages.append(age.item())
        stage_logits.append(stage)

    probabilities = torch.softmax(torch.stack(stage_logits).mean(0), dim=1).squeeze(0).cpu().numpy()
    return {
        "bone_age": float(np.mean(ages)),
        "uncertainty": float(np.std(ages)),
        "stage": int(np.argmax(probabilities)),
        "stage_probabilities": probabilities,
    }


def generate_radiographic_description(bone_age, stage, probs):
    """Structured report text, with closure percentage derived from stage probabilities."""
    closure_pct = float(np.clip(probs[1] * 25.0 + probs[2] * 65.0 + probs[3] * 100.0, 0.0, 100.0))

    description = (
        f"**EXAMINATION:** Volumetric Pediatric 3D Knee MRI Assessment Pipeline.\n\n"
        f"**FINDINGS:** Volumetric evaluation of the knee demonstrates structural features "
        f"corresponding to an estimated skeletal maturity of **{bone_age:.2f} years**.\n\n"
    )

    if stage == 0:
        description += (
            f"The epiphyseal plates are wide open with an estimated **{closure_pct:.1f}% closure**. "
            f"Hyperintense cartilaginous physeal zones remain visible across the distal femur and "
            f"proximal tibia, without bone bridging.\n\n"
        )
    elif stage == 1:
        description += (
            f"Early maturation with an estimated **{closure_pct:.1f}% closure**. Sclerosis is "
            f"thickening along the central physis while the plate remains largely open.\n\n"
        )
    elif stage == 2:
        description += (
            f"Advanced maturation with an estimated **{closure_pct:.1f}% closure**. Bone bridging "
            f"crosses central and peripheral physeal segments, indicating ongoing fusion.\n\n"
        )
    else:
        description += (
            f"The physes are closed, with **{closure_pct:.1f}% closure** and a consolidated bone "
            f"matrix replacing the growth plate.\n\n"
        )

    description += f"**IMPRESSION:** Skeletal development matches **{STAGE_NAMES[stage]}**."
    return closure_pct, description


def run_production_inference(dicom_dir, biological_sex, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_checkpoint(weights_path, device)

    input_shape = tuple(metadata.get("input_shape", DEFAULT_INPUT_SHAPE))
    volume = load_series(dicom_dir, input_shape)
    result = predict_scan(model, volume, biological_sex, device)
    closure_pct, clinical_text = generate_radiographic_description(
        result["bone_age"], result["stage"], result["stage_probabilities"])

    print("\n" + "=" * 58)
    print(" 🏥 PEDIATRIC KNEE MRI BONE AGE INFERENCE REPORT ")
    print("=" * 58)
    print(f"📂 Scan          : {dicom_dir}")
    print(f"🧬 Sex           : {str(biological_sex).upper()}")
    print(f"📐 Input grid    : {volume.shape}")
    print(f"🎯 Bone age      : {result['bone_age']:.2f} ± {result['uncertainty']:.2f} years")
    if "val_mae" in metadata:
        print(f"📊 Model val MAE : {metadata['val_mae']:.2f} years")
    print("-" * 58)
    print(clinical_text)
    print("=" * 58 + "\n")
    return result["bone_age"], result["stage"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knee MRI bone age inference")
    parser.add_argument("--dir", required=True, help="Path to the patient DICOM folder")
    parser.add_argument("--sex", required=True, choices=["m", "f", "male", "female"])
    parser.add_argument("--weights", default="final_knee_model_resnet34.pth")
    args = parser.parse_args()
    run_production_inference(args.dir, args.sex, args.weights)
