from __future__ import annotations

import torch
from monai.networks.nets import DenseNet121
from torch import nn


class KneeBoneAgeMultiTaskNet(nn.Module):
    def __init__(self, num_growth_stages: int = 4) -> None:
        super().__init__()
        self.backbone = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=512,
        )
        self.sex_encoder = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Dropout(0.2))
        self.fusion = nn.Sequential(nn.Linear(528, 256), nn.ReLU(), nn.Dropout(0.3))
        self.regression_head = nn.Linear(256, 1)
        self.stage_head = nn.Linear(256, num_growth_stages)

    def _collapse_spatial_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() > 2:
            features = features.mean(dim=tuple(range(2, features.dim())))
        return features.flatten(1)

    def forward(self, image: torch.Tensor, sex: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.float()
        sex = sex.float().view(-1, 1)
        spatial_features = self.backbone(image)
        spatial_features = self._collapse_spatial_features(spatial_features)
        sex_features = self.sex_encoder(sex)
        fused_features = torch.cat([spatial_features, sex_features], dim=1)
        shared_representation = self.fusion(fused_features)
        predicted_age = self.regression_head(shared_representation).squeeze(-1)
        predicted_stage = self.stage_head(shared_representation)
        return predicted_age, predicted_stage
