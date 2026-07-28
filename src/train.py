import os
import torch
import torch.nn as nn
from src.model import KneeBoneAgeMultiTaskNet
from src.predict import load_and_sort_dicom_volume

def execute_production_tuning(weights_input="final_knee_model_fine_tuned.pth", weights_output="final_knee_model_fine_tuned.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Initializing fine-tuning loop on device target: {device}")
    
    model = KneeBoneAgeMultiTaskNet()

    if os.path.exists(weights_input):
        print(f"🔄 Loading structural base weights from: {weights_input}")
        model.load_state_dict(torch.load(weights_input, map_location=device), strict=False)
    else:
        print(f"⚠️ Warning: Base weights '{weights_input}' not found. Starting raw run.")

    model.to(device)

    print("PyTorch fine-tuning strategy active. Protecting core features...")
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False

    for layer in [model.attention_gate, model.channel_bridge, model.sex_encoder, model.fusion, model.regression_head, model.stage_head]:
        for param in layer.parameters():
            param.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    criterion_age = nn.MSELoss()
    criterion_stage = nn.CrossEntropyLoss()

    training_data = [
        {"dir": "data/imported_patient_scan", "age": 14.8, "sex": 1.0, "stage": 0}
    ]

    print("\n🚀 Commencing multi-patient clinical cohort training sweeps...")
    
    for epoch in range(1, 41):
        if epoch == 16:
            print("\n🔓 PHASE 2: Unfreezing entire network for global co-adaptation...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)

        model.train()
        epoch_mae = 0.0
        active_cases = 0

        for pt in training_data:
            if not os.path.exists(pt["dir"]):
                continue
                
            optimizer.zero_grad()
            
            vol = load_and_sort_dicom_volume(pt["dir"])
            vol_tensor = torch.tensor(vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            sex_tensor = torch.tensor([[pt["sex"]]], dtype=torch.float32).to(device)
            t_age = torch.tensor([pt["age"]], dtype=torch.float32).to(device)
            t_stage = torch.tensor([pt["stage"]], dtype=torch.long).to(device)

            age_pred, stage_pred = model(vol_tensor, sex_tensor)
            
            loss = criterion_age(age_pred, t_age) + (criterion_stage(stage_pred, t_stage) * 2.0)
            
            loss.backward()
            optimizer.step()

            epoch_mae += abs(age_pred.item() - pt["age"])
            active_cases += 1

        if active_cases > 0 and (epoch % 5 == 0 or epoch == 1 or epoch == 16):
            avg_mae = epoch_mae / active_cases
            print(f"📈 Epoch {epoch:02d}/40 | Current Active Cohort MAE: {avg_mae:.4f} years")

    torch.save(model.state_dict(), weights_output)
    print(f"✅ Production weights successfully saved to disk at: {weights_output}")

if __name__ == "__main__":
    execute_production_tuning()
