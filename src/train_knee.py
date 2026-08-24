
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pydicom

# -----------------------------
# Dataset Loader
# -----------------------------
class KneeAgeDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        for folder in os.listdir(root):
            path = os.path.join(root, folder)
            if os.path.isdir(path):
                # Extract age from folder name
                age = float(folder.split("_")[3])
                slices = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".dcm")])
                self.samples.append((slices, age))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        slice_paths, age = self.samples[idx]

        # Load all slices and stack into [D, H, W]
        volume = []
        for sp in slice_paths:
            ds = pydicom.dcmread(sp)
            img = ds.pixel_array.astype(np.float32)
            img = img / 65535.0  # normalize
            volume.append(img)

        volume = np.stack(volume, axis=0)  # [D, H, W]
        volume = torch.tensor(volume).unsqueeze(0)  # [1, D, H, W]

        return volume, torch.tensor(age, dtype=torch.float32)

# -----------------------------
# Simple 3D CNN Model
# -----------------------------
class KneeAgeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -----------------------------
# Training Loop
# -----------------------------
def train():
    dataset = KneeAgeDataset("data/synthetic_knees_batch")
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = KneeAgeNet().cuda()
    opt = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.L1Loss()  # MAE

    for epoch in range(20):
        total_loss = 0
        for volume, age in loader:
            volume = volume.cuda()
            age = age.cuda()

            pred = model(volume)
            loss = loss_fn(pred.squeeze(), age)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/20 — MAE: {total_loss/len(loader):.3f}")

    torch.save(model.state_dict(), "knee_age_model.pth")
    print("Model saved as knee_age_model.pth")

if __name__ == "__main__":
    train()
