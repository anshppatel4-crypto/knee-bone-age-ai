import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
from src.predict import load_and_sort_dicom_volume
from src.model import KneeBoneAgeMultiTaskNet
def load_model(weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KneeBoneAgeMultiTaskNet()
    model.load_state_dict(torch.load(weights, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model, device

def repeat_test(scan_dir, sex, weights):
    print("\n=== REPEAT TEST ===")
    model, device = load_model(weights)

    vol = load_and_sort_dicom_volume(scan_dir)
    vol_tensor = torch.tensor(vol).unsqueeze(0).unsqueeze(0).float().to(device)
    sex_tensor = torch.tensor([[1.0 if sex=="M" else 0.0]]).float().to(device)

    outputs = []
    for _ in range(10):
        with torch.no_grad():
            age, _ = model(vol_tensor, sex_tensor)
            outputs.append(age.item())

    print("Outputs:", outputs)
    print("Range:", max(outputs) - min(outputs))

def noise_test(scan_dir, sex, weights):
    print("\n=== NOISE TEST ===")
    model, device = load_model(weights)

    vol = load_and_sort_dicom_volume(scan_dir)
    sex_tensor = torch.tensor([[1.0 if sex=="M" else 0.0]]).float().to(device)

    for noise_level in [0.001, 0.005, 0.01]:
        noisy = vol + np.random.normal(0, noise_level, vol.shape)
        vol_tensor = torch.tensor(noisy).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            age, _ = model(vol_tensor, sex_tensor)

        print(f"Noise {noise_level}: {age.item():.4f}")

def shuffle_test(scan_dir, sex, weights):
    print("\n=== SLICE SHUFFLE TEST ===")
    model, device = load_model(weights)

    vol = load_and_sort_dicom_volume(scan_dir)
    sex_tensor = torch.tensor([[1.0 if sex=="M" else 0.0]]).float().to(device)

    # normal
    vol_tensor = torch.tensor(vol).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        base_age, _ = model(vol_tensor, sex_tensor)

    # shuffled
    shuffled = vol.copy()
    np.random.shuffle(shuffled)
    vol_tensor_shuf = torch.tensor(shuffled).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        shuf_age, _ = model(vol_tensor_shuf, sex_tensor)

    print("Normal:", base_age.item())
    print("Shuffled:", shuf_age.item())

if __name__ == "__main__":
    scan = "data/imported_patient_scan"
    weights = "final_knee_model_fine_tuned.pth"
    sex = "M"

    repeat_test(scan, sex, weights)
    noise_test(scan, sex, weights)
    shuffle_test(scan, sex, weights)
