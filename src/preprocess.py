"""Volume preprocessing shared by training and inference.

Training and inference must see identically prepared inputs; keeping this in one
module is what guarantees it.
"""
import numpy as np
from scipy.ndimage import zoom

from src.dicom_io import read_mr_series

# (depth, height, width) fed to the network
DEFAULT_INPUT_SHAPE = (32, 192, 192)


def preprocess_volume(volume, target_shape=DEFAULT_INPUT_SHAPE):
    """Normalise intensities and resample a [D, H, W] volume onto the model grid."""
    volume = np.asarray(volume, dtype=np.float32)

    # MRI background is air: if the corner is brighter than average the series is inverted
    if volume[:, :8, :8].mean() > volume.mean():
        volume = volume.max() - volume

    # Robust windowing, so a few bright voxels cannot set the scale
    lo, hi = np.percentile(volume, (1.0, 99.0))
    volume = (np.clip(volume, lo, hi) - lo) / (hi - lo + 1e-8)

    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    if factors != [1.0, 1.0, 1.0]:
        volume = zoom(volume, factors, order=1)

    # Zero mean, unit variance: what the MedicalNet-pretrained trunk expects
    volume = (volume - volume.mean()) / (volume.std() + 1e-8)
    return volume.astype(np.float32)


def load_series(folder_path, target_shape=DEFAULT_INPUT_SHAPE):
    """Read a DICOM folder and return a preprocessed [D, H, W] volume."""
    volume, _ = read_mr_series(folder_path)
    return preprocess_volume(volume, target_shape)
