import os
import torch
import torch.nn as nn
from src.model import KneeBoneAgeMultiTaskNet
from src.predict import load_and_sort_dicom_volume

def execute_production_tuning(weights_input="inflated_knee_backbone3d.pth", weights_output="final_knee_model_fine_tuned.pth"):
    # 1. Fall back dynamically to CPU if local GPU isn't available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Initializing fine-tuning loop on device target: {device}")
    
    model = KneeBoneAgeMultiTaskNet()
    
    # 2. Safely load your pre-trained weights file if it exists locally
    if os.path.exists(weights_input):
        print(f"🔄 Loading structural base weights from: {weights_input}")
        model.load_state_dict(torch.load(weights_input, map_location=device), strict=False)
    else:
        print(f"⚠️ Warning: Base weights '{weights_input}' not found. Starting with initial layers.")
        
    model.to(device)
    
    # 3. Layer Freezing Strategy: Lock down early features, optimize deep layers
    print("🔒 Locking down early feature layers for fine-tuning configuration...")
    for name, param in model.backbone.features.named_parameters():
        if "denseblock1" in name or "denseblock2" in name or "conv0" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            
    # 4. Optimization configurations
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion_age = nn.MSELoss()
    criterion_stage = nn.CrossEntropyLoss()
    
    # 5. Define your multi-patient training cohort configuration maps
    training_data = [
        {"dir": "data/cohort_pt_1", "age": 11.5, "sex": 0.0, "stage": 0},  # Early Maturation
        {"dir": "data/cohort_pt_2", "age": 13.5, "sex": 1.0, "stage": 1},  # Mid Maturation
        {"dir": "data/cohort_pt_3", "age": 15.5, "sex": 1.0, "stage": 2}   # Terminal Closure
    ]
    
    # 6. Gradient Descent Optimization Loop
    model.train()
    print("\n🚀 Commencing multi-patient clinical cohort training sweeps...")
    for epoch in range(1, 41):  # 40 full epochs for convergence
        epoch_mae = 0.0
        active_cases = 0
        
        for pt in training_data:
            # Skip profile dynamically if the user hasn't generated the data folder yet
            if not os.path.exists(pt["dir"]):
                continue
                
            optimizer.zero_grad()
            
            # Read, normalize, and trilinear-resize volumetric blocks
            vol = load_and_sort_dicom_volume(pt["dir"])
            vol_tensor = torch.tensor(vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            sex_tensor = torch.tensor([[pt["sex"]]], dtype=torch.float32).to(device)
            
            t_age = torch.tensor([[pt["age"]]], dtype=torch.float32).to(device)
            t_stage = torch.tensor([pt["stage"]], dtype=torch.long).to(device)
            
            # Forward execution pass
            age_pred, stage_pred = model(vol_tensor, sex_tensor)
            
            # Calculate multi-task composite loss values
            loss = criterion_age(age_pred, t_age) + (criterion_stage(stage_pred, t_stage) * 10.0)
            
            # Backward execution and parameter updates
            loss.backward()
            optimizer.step()
            
            epoch_mae += abs(age_pred.item() - pt["age"])
            active_cases += 1
            
        # Display progress status reports periodically
        if active_cases > 0 and (epoch % 5 == 0 or epoch == 1):
            avg_mae = epoch_mae / active_cases
            print(f"📈 Epoch {epoch:02d}/40 | Current Active Cohort MAE: {avg_mae:.4f} years")
            
    # Save optimized parameter matrices out to a clean binary file
    torch.save(model.state_dict(), weights_output)
    print(f"✅ Production weights successfully saved to disk at: {weights_output}")

if __name__ == "__main__":
    execute_production_tuning()
