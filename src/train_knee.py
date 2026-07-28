from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.dataset import KneeDicomDataset
from src.model import KneeBoneAgeMultiTaskNet

def train_pipeline(data_catalog_csv: Optional[str] = None, epochs: int = 5, batch_size: int = 2) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training pipeline initialized. Active compute target: {device}")
    repo_root = Path(__file__).resolve().parent.parent

    rsna_path = repo_root / "boneage-training-dataset.csv"
    if rsna_path.exists():
        print(f"--> Found real pediatric dataset: {rsna_path.name}. Entering Pre-Training Mode.")
        catalog_path = rsna_path
        is_knee_3d = False 
    else:
        print("--> Real dataset sheet missing from root. Falling back to local data index mapping.")
        catalog_path = Path(data_catalog_csv) if data_catalog_csv else repo_root / "data" / "data_catalog.csv"
        is_knee_3d = True

    if not catalog_path.exists():
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"folder_path": "data/imported_patient_scan", "sex": 1.0, "bone_age": 14.8, "growth_stage": 0}]).to_csv(catalog_path, index=False)

    dataset = KneeDicomDataset(catalog_path, is_knee_3d=is_knee_3d)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = KneeBoneAgeMultiTaskNet(num_growth_stages=4).to(device)
    
    # Freeze backbone initially for custom attention compatibility
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
        
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-5)
    criterion_age = nn.MSELoss()
    criterion_stage = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_age_loss = 0.0
        running_stage_loss = 0.0
        running_mae = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            sex = batch["sex"].to(device).float().view(-1, 1)
            age_targets = batch["bone_age"].to(device).float().view(-1)
            stage_targets = batch["growth_stage"].to(device).long().view(-1)

            optimizer.zero_grad(set_to_none=True)
            predicted_age, predicted_stage = model(images, sex)

            loss_age = criterion_age(predicted_age, age_targets)
            loss_stage = criterion_stage(predicted_stage, stage_targets)
            total_loss = loss_age + (2.0 * loss_stage)

            total_loss.backward()
            optimizer.step()

            running_age_loss += loss_age.item()
            running_stage_loss += loss_stage.item()
            current_mae = torch.mean(torch.abs(predicted_age - age_targets)).item()
            running_mae += current_mae

            print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(train_loader)} | MAE={current_mae:.4f} years")

    # Save out updated state representations cleanly
    torch.save(model.state_dict(), repo_root / "final_knee_model_fine_tuned.pth")
    print("✅ Model tuning step complete. Checkpoint exported.")

if __name__ == "__main__":
    train_pipeline()
