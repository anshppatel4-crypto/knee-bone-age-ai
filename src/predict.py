import os
import argparse
import pydicom
import numpy as np
import torch
from scipy.ndimage import zoom
from src.model import KneeBoneAgeMultiTaskNet

def load_and_sort_dicom_volume(dicom_dir, target_depth=64, target_resolution=256):
    """Parses a directory containing raw DICOM slices, sorts them chronologically

    by physical SliceLocation, and resizes the volume to a standardized 3D matrix.
    """
    # 1. Read all files inside the directory, using force=True to handle custom generated data
    dicom_files = [
        pydicom.dcmread(os.path.join(dicom_dir, f), force=True) 
        for f in os.listdir(dicom_dir) 
        if f.lower().endswith('.dcm') or f.lower().endswith('.dicom') or f.lower().startswith('slice_')
    ]
    
    if not dicom_files:
        raise FileNotFoundError(f"❌ Error: No readable DICOM slices found in target directory: {dicom_dir}")
        
    # 2. Sort the slices strictly along the physical Z-axis using the scanner metadata tags
    dicom_files.sort(key=lambda x: float(x.SliceLocation) if 'SliceLocation' in x else 0.0)
    
    # 3. Extract the raw pixel grids and stack them: Shape (Slices, Height, Width)
    volume = np.stack([dcm.pixel_array for dcm in dicom_files], axis=0)
    
    # 4. Standardize and normalize pixel intensity values to the [0.0, 1.0] window
    volume = volume.astype(np.float32)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)
    
    # 5. FIXED: 3D Volumetric Standardizer
    # Resizes the input volume to exactly (64, 256, 256) so MONAI's 3D layers don't collapse
    current_d, current_h, current_w = volume.shape
    print(f"📐 Standardizing volume: Current shape ({current_d}, {current_h}, {current_w})")
    
    zoom_d = target_depth / current_d
    zoom_h = target_resolution / current_h
    zoom_w = target_resolution / current_w
    
    # Apply bilinear/trilinear spline interpolation scaling
    print(f"🔄 Interpolating 3D matrices to clinical target grid ({target_depth}, {target_resolution}, {target_resolution})...")
    volume = zoom(volume, (zoom_d, zoom_h, zoom_w), order=1)
    
    return volume

def run_production_inference(dicom_dir, biological_sex, weights_path):
    """Loads the model architecture, injects the user variables, and executes

    a forward pass to calculate continuous pediatric bone age.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize our custom multi-task network structure
    model = KneeBoneAgeMultiTaskNet()
    
    # 2. Safely map and bind your saved weights file to your current hardware device
    if os.path.exists(weights_path):
        print(f"🔄 Loading trained neural network weights from: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    else:
        print(f"⚠️ Warning: Checkpoint '{weights_path}' not found. Running inference with initialization baselines.")
        
    model.to(device)
    model.eval()  # Put layers into locked evaluation mode
    
    # 3. Read and process the target 3D image stack volume
    volume = load_and_sort_dicom_volume(dicom_dir)
    
    # Expand dimensions to fit PyTorch batch constraints: Shape (Batch=1, Channels=1, Depth, Height, Width)
    volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # 4. Process sex metadata: 1.0 for Male, 0.0 for Female to match your network's embedded layers
    sex_value = 1.0 if biological_sex.lower() in ['m', 'male'] else 0.0
    sex_tensor = torch.tensor([sex_value], dtype=torch.float32).unsqueeze(0).to(device)
    
    # 5. Perform the forward pass without tracking gradients
    with torch.no_grad():
        age_prediction, stage_prediction = model(volume_tensor, sex_tensor)
        
        # Pull the absolute scalar float value out of the regression head
        calculated_bone_age = age_prediction.item()
        
        # Pull the highest-probability category index out of the growth plate tier head
        predicted_stage_tier = torch.argmax(stage_prediction, dim=1).item()
        
    # 6. Output professional, clean interface metrics to the terminal
    print("\n" + "="*55)
    print("   🏥 PEDIATRIC KNEE MRI BONE AGE INFERENCE REPORT   ")
    print("="*55)
    print(f"📂 Target Scan Path    : {dicom_dir}")
    print(f"🧬 Patient Biological Sex: {biological_sex.upper()}")
    print(f"📊 3D Volume Dimensions : {volume.shape[0]} Slices | {volume.shape[1]}x{volume.shape[2]} Matrix")
    print("-"*55)
    print(f"🎯 Calculated Bone Age   : {calculated_bone_age:.2f} Years")
    print(f"📈 Growth Plate Closure : Stage {predicted_stage_tier}")
    print("="*55 + "\n")
    
    return calculated_bone_age, predicted_stage_tier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production CLI Inference Engine for Pediatric Knee Bone Age Estimation")
    parser.add_argument("--dir", type=str, required=True, help="Path to the directory containing a patient's DICOM slice set")
    parser.add_argument("--sex", type=str, required=True, choices=['m', 'f', 'male', 'female'], help="Patient biological sex")
    parser.add_argument("--weights", type=str, default="skeletal_maturity_backbone.pth", help="Path to trained weights file")
    
    args = parser.parse_args()
    run_production_inference(args.dir, args.sex, args.weights)
