from __future__ import annotations
import torch
from monai.networks.nets import DenseNet121
from torch import nn

class AnisotropicAttention3D(nn.Module):
    """Novelty Component: Evaluates cross-slice spatial dynamics to protect 
    micro-gap growth plate boundaries against thick-slice blurring.
    """
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = x.size()
        
        proj_query = self.query_conv(x).view(batch, -1, depth * height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch, -1, depth * height * width)
        
        energy = torch.bmm(proj_query, proj_key)
        attention_map = self.softmax(energy)
        
        proj_value = self.value_conv(x).view(batch, -1, depth * height * width)
        out = torch.bmm(proj_value, attention_map.permute(0, 2, 1))
        out = out.view(batch, channels, depth, height, width)
        
        return self.gamma * out + x

class KneeBoneAgeMultiTaskNet(nn.Module):
    """Multi-Task Deep Learning Network with Anisotropic Spatial Attention
    and fused biological metadata embeddings.
    """
    def __init__(self, num_growth_stages: int = 4) -> None:
        super().__init__()
        self.backbone = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=512,
        )
        self.attention_gate = AnisotropicAttention3D(in_channels=1024)
        self.channel_bridge = nn.Conv3d(1024, 512, kernel_size=1)
        
        # 👑 FIX: Replaced BatchNorm1d with LayerNorm to handle single-patient scans safely
        self.sex_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 👑 FIX: Replaced BatchNorm1d with LayerNorm to guarantee stability
        self.fusion = nn.Sequential(
            nn.Linear(512 + 16, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.regression_head = nn.Linear(256, 1)
        self.stage_head = nn.Linear(256, num_growth_stages)

    def _collapse_spatial_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() > 2:
            features = features.mean(dim=tuple(range(2, features.dim())))
        return features.flatten(1)

    def forward(self, image: torch.Tensor, sex: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.float()
        sex = sex.float().view(-1, 1)
        
        spatial_features = self.backbone.features(image)
        spatial_features = self.attention_gate(spatial_features)
        spatial_features = self.channel_bridge(spatial_features)
        spatial_features = self._collapse_spatial_features(spatial_features)
        
        sex_features = self.sex_encoder(sex)
        fused_features = torch.cat([spatial_features, sex_features], dim=1)
        shared_representation = self.fusion(fused_features)
        
        raw_age = self.regression_head(shared_representation).squeeze(-1)
        predicted_stage = self.stage_head(shared_representation)
        
        predicted_age = torch.sigmoid(raw_age) * 20.0
        return predicted_age, predicted_stage
