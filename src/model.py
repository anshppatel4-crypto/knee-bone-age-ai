from __future__ import annotations
import torch
from monai.networks.nets.resnet import get_medicalnet_pretrained_resnet_args, resnet18, resnet34
from torch import nn

# Bone age is squashed into a plausible paediatric range instead of being unbounded
MAX_AGE_YEARS = 20.0

# Basic-block ResNets expose a 512-d pooled feature vector
BACKBONES = {"resnet18": (resnet18, 18, 512), "resnet34": (resnet34, 34, 512)}


class KneeBoneAgeMultiTaskNet(nn.Module):
    """3D ResNet trunk with fused sex metadata, predicting bone age and physeal stage.

    The trunk is initialised from MedicalNet (ResNet pre-trained on 23 medical
    imaging datasets), which matters far more than architecture depth when the
    training set is small.
    """

    def __init__(self, arch: str = "resnet34", num_growth_stages: int = 4,
                 pretrained: bool = True, dropout: float = 0.3,
                 conditioning: str = "film") -> None:
        super().__init__()
        if arch not in BACKBONES:
            raise ValueError(f"Unsupported arch '{arch}'. Choose from {sorted(BACKBONES)}.")

        builder, depth, feature_dim = BACKBONES[arch]
        self.arch = arch
        bias_downsample, shortcut_type = get_medicalnet_pretrained_resnet_args(depth)
        backbone_kwargs = dict(spatial_dims=3, n_input_channels=1, feed_forward=False,
                               shortcut_type=shortcut_type, bias_downsample=bias_downsample)

        if pretrained:
            try:
                self.backbone = builder(pretrained=True, **backbone_kwargs)
            except Exception as exc:  # offline, or cache unavailable
                print(f"⚠️ MedicalNet weights unavailable ({exc}); using random initialisation.")
                self.backbone = builder(pretrained=False, **backbone_kwargs)
        else:
            self.backbone = builder(pretrained=False, **backbone_kwargs)

        # LayerNorm rather than BatchNorm so single-scan inference stays stable
        self.conditioning = conditioning
        sex_dim = 64 if conditioning == "film" else 16
        self.sex_encoder = nn.Sequential(
            nn.Linear(1, sex_dim),
            nn.LayerNorm(sex_dim),
            nn.ReLU(),
        )

        # Concatenating 16 sex dims onto 512 image dims lets the network ignore sex
        # entirely. FiLM makes sex scale and shift every image feature, so it cannot.
        if conditioning == "film":
            self.feature_norm = nn.LayerNorm(feature_dim)
            self.film = nn.Linear(sex_dim, 2 * feature_dim)

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + sex_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regression_head = nn.Linear(256, 1)
        self.stage_head = nn.Linear(256, num_growth_stages)

    def forward(self, image: torch.Tensor, sex: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(image.float())
        sex_features = self.sex_encoder(sex.float().view(-1, 1))

        if self.conditioning == "film":
            scale, offset = self.film(sex_features).chunk(2, dim=-1)
            features = self.feature_norm(features) * (1.0 + scale) + offset

        shared = self.fusion(torch.cat([features, sex_features], dim=1))
        predicted_age = torch.sigmoid(self.regression_head(shared).squeeze(-1)) * MAX_AGE_YEARS
        return predicted_age, self.stage_head(shared)


def save_checkpoint(model: KneeBoneAgeMultiTaskNet, path: str, **metadata) -> None:
    """Store weights together with the architecture and metrics needed to rebuild and judge them."""
    torch.save({"arch": model.arch, "conditioning": model.conditioning,
                "state_dict": model.state_dict(), **metadata}, path)


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> tuple[KneeBoneAgeMultiTaskNet, dict]:
    """Rebuild a model from a checkpoint written by `save_checkpoint`."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "state_dict" not in checkpoint:
        raise ValueError(
            f"'{path}' predates the 3D ResNet pipeline (it holds a bare DenseNet state dict). "
            "Retrain with src/train.py to produce a compatible checkpoint."
        )

    # Checkpoints written before FiLM conditioning existed used plain concatenation
    model = KneeBoneAgeMultiTaskNet(arch=checkpoint.get("arch", "resnet34"), pretrained=False,
                                    conditioning=checkpoint.get("conditioning", "concat"))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    return model, metadata
