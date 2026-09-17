"""Knee MRI volume dataset with a preprocessing cache and MRI-shaped augmentation."""
import os
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import rotate, shift, zoom
from torch.utils.data import Dataset

from src.preprocess import DEFAULT_INPUT_SHAPE, load_series
# Biological enhancer for Stable Diffusion volumes (phantoms already model this physically)
from src.synthetic_biology import enhance_synthetic_knee

REQUIRED_COLUMNS = ("folder_path", "sex", "bone_age", "growth_stage")


class KneeVolumeDataset(Dataset):
    """Serves preprocessed 3D knee volumes with age, sex and physeal stage labels.

    Preprocessing (DICOM read, windowing, resampling) is cached to disk on first
    use, because repeating it every epoch dominates training time.
    """

    def __init__(self, catalog, input_shape=DEFAULT_INPUT_SHAPE, cache_dir="data/cache",
                 augment=False, enhance_synthetic=False, seed=0):
        self.catalog = catalog if isinstance(catalog, pd.DataFrame) else pd.read_csv(catalog)
        missing = [c for c in REQUIRED_COLUMNS if c not in self.catalog.columns]
        if missing:
            raise ValueError(f"Catalog is missing required column(s): {missing}")

        self.input_shape = tuple(input_shape)
        self.cache_dir = cache_dir
        self.augment = augment
        self.enhance_synthetic = enhance_synthetic
        self.rng = np.random.default_rng(seed)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.catalog)

    def _cache_path(self, folder_path):
        key = f"{os.path.normpath(folder_path).replace(os.sep, '_')}_{'x'.join(map(str, self.input_shape))}"
        return os.path.join(self.cache_dir, f"{key}.npy")

    def _volume(self, folder_path):
        if not self.cache_dir:
            return load_series(folder_path, self.input_shape)

        path = self._cache_path(folder_path)
        if os.path.exists(path):
            return np.load(path).astype(np.float32)

        volume = load_series(folder_path, self.input_shape)
        np.save(path, volume.astype(np.float16))  # half precision halves cache size
        return volume

    def _augment(self, volume):
        """Geometric and intensity variation that mimics real acquisition differences."""
        rng = self.rng

        if rng.random() < 0.5:  # left vs right knee
            volume = volume[::-1].copy()

        angle = rng.uniform(-8.0, 8.0)
        if abs(angle) > 0.5:
            volume = rotate(volume, angle, axes=(1, 2), reshape=False, order=1, mode="nearest")

        offsets = rng.uniform(-5, 5, size=3) * [0.4, 1.0, 1.0]
        volume = shift(volume, offsets, order=1, mode="nearest")

        volume = volume * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
        if rng.random() < 0.5:  # gamma on the positive part mimics contrast differences
            span = volume.max() - volume.min() + 1e-8
            normalised = (volume - volume.min()) / span
            volume = normalised ** rng.uniform(0.75, 1.3) * span + volume.min()

        # Smooth multiplicative bias field: upsampling a tiny random grid is much
        # cheaper than evaluating cosine terms over every voxel
        coarse = rng.normal(size=(3, 4, 4)).astype(np.float32)
        bias = zoom(coarse, [s / c for s, c in zip(volume.shape, coarse.shape)], order=3)
        bias = bias[:volume.shape[0], :volume.shape[1], :volume.shape[2]]
        volume = volume * (1.0 + 0.08 * bias / (np.abs(bias).max() + 1e-8))

        volume = volume + rng.normal(0, rng.uniform(0.0, 0.05), volume.shape)
        return ((volume - volume.mean()) / (volume.std() + 1e-8)).astype(np.float32)

    def __getitem__(self, index):
        row = self.catalog.iloc[index]
        age = float(row["bone_age"])
        volume = self._volume(row["folder_path"])

        if self.enhance_synthetic:
            volume = enhance_synthetic_knee(volume, age)
        if self.augment:
            volume = self._augment(volume)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(volume[np.newaxis])).float(),
            "sex": torch.tensor(float(row["sex"]), dtype=torch.float32),
            "bone_age": torch.tensor(age, dtype=torch.float32),
            "growth_stage": torch.tensor(int(row["growth_stage"]), dtype=torch.long),
        }


if __name__ == "__main__":
    print("Knee volume dataset ready.")
