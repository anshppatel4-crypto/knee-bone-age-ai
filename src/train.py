from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.dataset import KneeDicomDataset
from src.model import KneeBoneAgeMultiTaskNet

def train_pipeline(
    data_catalog_csv: Optional[str] = None,
    epochs: int = 5,
    batch_size: int = 2,
) -> None:
    # 1. Determine local hardware compute acceleration profiles
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training pipeline initialized. Active compute target: {device}")
    
    repo_root = Path(__file__).resolve().parent.parent
    
    # 2. Automatically detect if we are running Pre-training or Local Testing
    # If the real RSNA child spreadsheet is in our folder, prioritize it!
    rsna_path = repo_root / "boneage-training-dataset.csv"
    
    if rsna_path.exists():
        print(f"--> Found real pediatric dataset: {rsna_path.name}. Entering Pre-Training Mode.")
        catalog_path = rsna_path
        is_knee_3d = False  # Tells our data loader to parse the real 2D childhood rows
    else:
        # Fallback to local catalog if the large dataset isn't loaded yet
        print("--> Real dataset sheet missing from root. Falling back to local data index mapping.")
        catalog_path = Path(data_catalog_csv) if data_catalog_csv else repo_root / "data" / "data_catalog.csv"
        is_knee_3d = True
        
        if not catalog_path.exists():
            catalog_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [{"folder_path": "data/patient_001", "sex": 0.0, "bone_age": 14.5, "growth_stage": 2}]
            ).to_csv(catalog_path, index=False)

    # 3. Instantiate our upgraded dataset with explicit dimensionality routing
    dataset = KneeDicomDataset(catalog_path, is_knee_3d=is_knee_3d)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 4. Instantiate model and route onto target hardware matrix blocks
    model = KneeBoneAgeMultiTaskNet(num_growth_stages=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    criterion_age = nn.MSELoss()
    criterion_stage = nn.CrossEntropyLoss()

    # 5. Core Multi-Task Optimization Execution Training Loop
    for epoch in range(epochs):
        model.train()
        running_age_loss = 0.0
        running_stage_loss = 0.0
        running_mae = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            # Reshape sex matrix to fit the expected multi-modal network concatenation layers [Batch, 1]
            sex = batch["sex"].to(device).float().view(-1, 1)
            age_targets = batch["bone_age"].to(device).float().view(-1, 1)
            stage_targets = batch["growth_stage"].to(device).long().view(-1)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass calculations
            predicted_age, predicted_stage = model(images, sex)
            
            # Compute compound clinical validation metrics
            loss_age = criterion_age(predicted_age, age_targets)
            loss_stage = criterion_stage(predicted_stage, stage_targets)
            
            # Combined Loss Calculation weighted for architectural stability
            total_loss = loss_age + (2.0 * loss_stage)
            
            # Backpropagation updates
            total_loss.backward()
            optimizer.step()

            # Track processing error updates
            running_age_loss += loss_age.item()
            running_stage_loss += loss_stage.item()
            
            # Real-time Absolute Month/Year Calculation Matrix tracking
            current_mae = torch.mean(torch.abs(predicted_age - age_targets)).item()
            running_mae += current_mae

            print(
                f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Age Loss={loss_age.item():.4f} | Stage Loss={loss_stage.item():.4f} | "
                f"MAE={current_mae:.4f} years"
            )

        # Print consolidated validation summary logs
        num_batches = max(1, len(train_loader))
        print(
            f"=== Epoch {epoch + 1}/{epochs} complete ===\n"
            f"-> Avg Age Loss: {running_age_loss / num_batches:.4f}\n"
            f"-> Avg Stage Loss: {running_stage_loss / num_batches:.4f}\n"
            f"-> Avg MAE Accuracy: {running_mae / num_batches:.4f} years\n"
            f"================================="
        )

if __name__ == "__main__":
    train_pipeline()
