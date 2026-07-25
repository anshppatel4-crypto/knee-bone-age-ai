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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_root = Path(__file__).resolve().parent.parent
    catalog_path = Path(data_catalog_csv) if data_catalog_csv else repo_root / "data" / "data_catalog.csv"

    if not catalog_path.exists():
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"patient_dir": str(repo_root / "data"), "sex": 0.0, "bone_age": 8.0, "growth_stage": 0}]
        ).to_csv(catalog_path, index=False)

    dataset = KneeDicomDataset(catalog_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = KneeBoneAgeMultiTaskNet(num_growth_stages=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion_age = nn.MSELoss()
    criterion_stage = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_age_loss = 0.0
        running_stage_loss = 0.0
        running_mae = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            sex = batch["sex"].to(device)
            age_targets = batch["bone_age"].to(device).float()
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
            running_mae += torch.mean(torch.abs(predicted_age - age_targets)).item()

            print(
                f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Age Loss={loss_age.item():.4f} | Stage Loss={loss_stage.item():.4f} | "
                f"MAE={torch.mean(torch.abs(predicted_age - age_targets)).item():.4f}"
            )

        print(
            f"Epoch {epoch + 1}/{epochs} complete | "
            f"Avg Age Loss={running_age_loss / max(1, len(train_loader)):.4f} | "
            f"Avg Stage Loss={running_stage_loss / max(1, len(train_loader)):.4f} | "
            f"Avg MAE={running_mae / max(1, len(train_loader)):.4f}"
        )


if __name__ == "__main__":
    train_pipeline()
