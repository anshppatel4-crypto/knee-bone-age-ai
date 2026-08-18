import numpy as np
from scipy.ndimage import gaussian_filter

# Biological + MRI-physics inspired synthetic MRI enhancer
# Functions are intentionally simple and fast, using only numpy operations so they
# can be integrated into both generation and training data pipelines.


def apply_biological_constraints(vol: np.ndarray, age_years: float) -> np.ndarray:
    """
    Apply age-dependent anatomical modifications:
    - Simulate growth plate (epiphyseal) thickness and its gradual fusion with age
    - Brighten cartilage for younger ages and dim for older ages
    - Adjust metaphyseal intensity to roughly approximate water->fat transition

    vol: [D, H, W] float image
    returns enhanced volume (same shape)
    """
    vol = vol.astype(np.float32)
    D, H, W = vol.shape

    # Normalize local copy to 0..1 to make relative adjustments predictable
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin > 1e-8:
        norm = (vol - vmin) / (vmax - vmin)
    else:
        norm = vol - vmin

    # Determine growth plate thickness (in slices) as a decreasing function of age
    # Younger: thicker growth plate (e.g., up to 8 slices). Older: thin (1-2 slices)
    gp_thickness = int(np.clip(np.round(np.interp(age_years, [0, 20], [8, 1]))), 1, max(1, D // 4))

    # Position growth plate near the mid-epiphyseal region (approx center third)
    center = D // 2
    gp_start = max(0, center - gp_thickness // 2)
    gp_end = min(D, gp_start + gp_thickness)

    enhanced = norm.copy()

    # Brighten cartilage-like surface regions for younger ages; dim for older
    # Use a depth-based weighting that favors slices near the growth plate
    age_factor = float(np.clip(1.25 - (age_years - 10.0) * 0.02, 0.8, 1.4))

    # Simple cartilage surrogate: thin band just adjacent to growth plate on both sides
    cartilage_band = max(1, gp_thickness // 2)
    cartilage_slices = list(range(max(0, gp_start - cartilage_band), gp_start)) + list(range(gp_end, min(D, gp_end + cartilage_band)))
    for s in cartilage_slices:
        # Slightly boost intensity in these slices proportional to age_factor
        enhanced[s, :, :] = enhanced[s, :, :] * (1.0 * age_factor)

    # Epiphyseal fusion blending: gradually blend epiphysis into metaphysis as age increases
    # fusion_weight ranges 0 (young, separate) -> 1 (fused) with age
    fusion_weight = float(np.clip((age_years - 12.0) / 8.0, 0.0, 1.0))

    # Create blending ramps around the growth plate
    ramp_len = max(1, gp_thickness)
    for i in range(ramp_len):
        # top ramp (epiphysis side)
        idx_top = gp_start - 1 - i
        idx_bottom = gp_end + i
        if 0 <= idx_top < D:
            w = fusion_weight * (1.0 - (i / max(1, ramp_len)))
            enhanced[idx_top] = enhanced[idx_top] * (1.0 - 0.15 * w) + enhanced[gp_start] * (0.15 * w)
        if 0 <= idx_bottom < D:
            w = fusion_weight * (1.0 - (i / max(1, ramp_len)))
            enhanced[idx_bottom] = enhanced[idx_bottom] * (1.0 - 0.15 * w) + enhanced[gp_end - 1] * (0.15 * w)

    # Metaphyseal water->fat signal changes: with age, metaphysis gets more fat-like (lower T2)
    # Approximate metaphysis as slices away from the growth plate
    metaphysis_mask = np.ones((D,), dtype=np.float32)
    metaphysis_mask[gp_start:gp_end] = 0.0
    # Older -> reduce intensity slightly in metaphysis to simulate lower T2 signal
    metaphyseal_scale = 1.0 - (np.clip((age_years - 10.0) / 30.0, 0.0, 0.35))
    for s in range(D):
        if metaphysis_mask[s] > 0:
            enhanced[s] = enhanced[s] * metaphyseal_scale

    # Rescale back to original range
    enhanced = enhanced * (vmax - vmin + 1e-8) + vmin
    return enhanced


def apply_mri_physics(vol: np.ndarray, age_years: float) -> np.ndarray:
    """
    Apply simple MRI physics-like adjustments:
    - T2-like intensity modulation: cartilage should appear bright, bone darker
    - Apply small spatial smoothing to mimic point-spread function
    - Apply mild slice-wise T2 decay curve to deeper slices
    """
    vol = vol.astype(np.float32)
    D, H, W = vol.shape

    # Create a cartilage surrogate mask by selecting brighter voxels in each slice's top percentile
    phys = vol.copy()

    # Smooth a bit to avoid noisy thresholds
    phys = gaussian_filter(phys, sigma=0.5)

    # Slice-wise T2-like decay: make central slices (around growth plate) relatively brighter
    center = D // 2
    distances = np.abs(np.arange(D) - center).astype(np.float32)
    decay_curve = np.exp(-distances / (max(1.0, D / (6.0 + age_years * 0.05))))
    # Normalize curve to mean 1
    decay_curve = decay_curve / (np.mean(decay_curve) + 1e-8)

    for s in range(D):
        phys[s] = phys[s] * (0.9 + 0.2 * decay_curve[s])

    # Cartilage vs bone separation by local contrast: enhance higher-percentile voxels more
    for s in range(D):
        slice_ = phys[s]
        p90 = np.percentile(slice_, 90)
        # cartilage-like voxels (above p90) get a boost, scaled by age (younger => stronger T2)
        cartilage_boost = np.clip(1.15 - (age_years - 10.0) * 0.01, 1.02, 1.25)
        mask = slice_ >= p90
        slice_[mask] = slice_[mask] * cartilage_boost
        phys[s] = slice_

    # Small Gaussian smoothing to mimic scanner blur
    phys = gaussian_filter(phys, sigma=0.6)

    return phys


def enhance_synthetic_knee(vol: np.ndarray, age_years: float) -> np.ndarray:
    """
    High-level API: take a raw synthetic volume and make it biologically realistic and
    MRI-physics plausible.

    Steps:
    1. Apply biological constraints (cartilage, growth plate, metaphysis)
    2. Apply MRI physics (T2-like intensity shaping)
    3. Add subtle Gaussian noise to mimic texture
    """
    if not isinstance(vol, np.ndarray):
        vol = np.array(vol, dtype=np.float32)

    # Ensure shape is [D, H, W]
    if vol.ndim == 3:
        working = vol.astype(np.float32)
    elif vol.ndim == 4 and vol.shape[0] == 1:
        # [1, D, H, W]
        working = vol[0].astype(np.float32)
    else:
        # As a fallback, try to squeeze
        working = np.squeeze(vol).astype(np.float32)

    # Stage 1: structural, age-based changes
    bio = apply_biological_constraints(working, age_years)

    # Stage 2: MRI physics shaping
    phys = apply_mri_physics(bio, age_years)

    # Stage 3: subtle noise texture
    # Relative noise amplitude tuned to input dynamic range
    dyn = phys.max() - phys.min() if phys.max() - phys.min() > 1e-8 else 1.0
    noise_std = dyn * 0.005  # ~0.5% of dynamic range
    noise = np.random.normal(scale=noise_std, size=phys.shape).astype(np.float32)
    out = phys + noise

    # Clip back to original numeric range
    # Try to preserve original dtype/scaling
    return out.astype(np.float32)
