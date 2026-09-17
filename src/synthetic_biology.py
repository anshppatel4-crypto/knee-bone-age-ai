"""Age-dependent skeletal biology and MRI physics for synthetic knee MRI volumes.

Volumes are sagittal stacks shaped [D, H, W]:
    D = mediolateral slice index
    H = superior -> inferior rows (distal femur on top, proximal tibia below)
    W = anterior -> posterior columns
The distal femoral and proximal tibial growth plates (physes) are therefore
horizontal bands across H, sitting above and below the tibiofemoral joint line.
"""
import numpy as np

# T2-like contrast model: S = PD * exp(-TE / T2)
TE_MS = 30.0
T2_MARROW_MS = 35.0
T2_BONE_MS = 5.0          # cortical / subchondral bone: very short T2, dark
T2_WEIGHTING = 0.35       # how strongly T2 decay modulates the diffusion output
NOISE_FRACTION = 0.01     # Gaussian noise sigma as a fraction of dynamic range


# -----------------------------------------------------------------------------
# Age curves
# -----------------------------------------------------------------------------
def growth_plate_thickness(age_years, height):
    """Physeal thickness in rows: wide in childhood, thinning through puberty."""
    fraction = np.interp(age_years, [0, 10, 13, 15, 17, 19], [0.035, 0.030, 0.022, 0.012, 0.004, 0.0])
    return max(float(fraction * height), 1.0)


def fusion_fraction(age_years):
    """0 = open physis, 1 = fully fused. Knee physes close over roughly 14-18 years."""
    return float(1.0 / (1.0 + np.exp(-(age_years - 15.5) / 0.9)))


def marrow_water_fraction(age_years):
    """Metaphyseal red (water-rich) to yellow (fatty) marrow conversion."""
    return float(np.interp(age_years, [0, 5, 10, 15, 20], [0.70, 0.60, 0.45, 0.30, 0.20]))


def cartilage_t2_ms(age_years):
    """Younger, more hydrated cartilage has a longer T2 and appears brighter."""
    return float(np.interp(age_years, [0, 10, 15, 20], [55.0, 48.0, 40.0, 35.0]))


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def _normalize(vol):
    vmin, vmax = float(vol.min()), float(vol.max())
    return (vol - vmin) / (vmax - vmin + 1e-8), vmin, vmax


def _row_band(height, center, half_width):
    """Soft Gaussian weight per row, peaking at `center`."""
    rows = np.arange(height, dtype=np.float32)
    return np.exp(-0.5 * ((rows - center) / max(half_width, 0.5)) ** 2).astype(np.float32)


def _smooth_rows(vol, width):
    """Box filter along H using cumulative sums."""
    width = int(width) | 1
    pad = width // 2
    padded = np.pad(vol, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    csum = np.cumsum(padded, axis=1, dtype=np.float64)
    csum = np.concatenate([np.zeros_like(csum[:, :1]), csum], axis=1)
    return ((csum[:, width:] - csum[:, :-width]) / width).astype(np.float32)


def _knee_anatomy(norm, age_years):
    """Locate joint line, physes, epiphyses and metaphyses as per-row weights [H]."""
    height = norm.shape[1]

    # Joint line: darkest row (joint space between cortices) in the central band
    profile = _smooth_rows(norm.mean(axis=(0, 2))[None, :, None], 0.03 * height)[0, :, 0]
    lo, hi = int(0.35 * height), int(0.65 * height)
    joint = lo + int(np.argmin(profile[lo:hi]))

    thickness = growth_plate_thickness(age_years, height)
    femoral_physis = joint - 0.20 * height
    tibial_physis = joint + 0.17 * height
    articular_half = float(np.interp(age_years, [0, 10, 15, 20], [0.020, 0.015, 0.010, 0.008])) * height

    def pair(offset_femur, offset_tibia, half_width):
        return np.maximum(_row_band(height, femoral_physis + offset_femur, half_width),
                          _row_band(height, tibial_physis + offset_tibia, half_width))

    return {
        "thickness": thickness,
        "physis": pair(0.0, 0.0, thickness / 2),
        "articular": _row_band(height, joint, articular_half),
        # Epiphyses lie between each physis and the joint; metaphyses lie on the far side
        "epiphysis": pair(0.10 * height, -0.085 * height, 0.04 * height),
        "metaphysis": pair(-0.08 * height, 0.08 * height, 0.05 * height),
    }


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def apply_biological_constraints(vol, age_years, anatomy=None):
    """Age-dependent cartilage signal, physeal fusion and metaphyseal marrow conversion."""
    vol = np.asarray(vol, dtype=np.float32)
    norm, vmin, vmax = _normalize(vol)
    anatomy = anatomy or _knee_anatomy(norm, age_years)
    fusion = fusion_fraction(age_years)
    row = lambda weights: weights[None, :, None]

    # Cartilage: bright in hydrated young physes/articular surfaces, dimmer with maturity
    cartilage_gain = float(np.interp(age_years, [0, 10, 15, 20], [0.35, 0.25, 0.05, -0.10]))
    cartilage = np.maximum(anatomy["physis"] * (1.0 - fusion), anatomy["articular"])
    out = norm * (1.0 + cartilage_gain * row(cartilage))

    # Fusion: bridge the physis with its neighbouring epiphyseal + metaphyseal bone,
    # leaving a faint low-signal physeal scar once closed
    bridge = _smooth_rows(out, 3 * anatomy["thickness"] + 0.02 * norm.shape[1])
    blend = fusion * row(anatomy["physis"])
    out = out * (1.0 - blend) + bridge * blend
    out *= 1.0 - 0.08 * fusion * row(anatomy["physis"])

    # Metaphyseal marrow: water-rich red marrow brighter, fatty yellow marrow dimmer
    water = marrow_water_fraction(age_years)
    out *= 1.0 + 0.30 * (water - 0.45) * row(anatomy["metaphysis"])

    return out * (vmax - vmin) + vmin


def apply_mri_physics(vol, age_years, anatomy=None):
    """T2-like contrast: long-T2 cartilage brightens, short-T2 bone darkens."""
    vol = np.asarray(vol, dtype=np.float32)
    norm, vmin, vmax = _normalize(vol)
    anatomy = anatomy or _knee_anatomy(norm, age_years)

    foreground = norm > 0.05
    if not foreground.any():
        return vol

    # Tissue masks
    cartilage_rows = np.maximum(anatomy["physis"] * (1.0 - fusion_fraction(age_years)), anatomy["articular"])
    cartilage = cartilage_rows[None, :, None] * (norm >= np.median(norm[foreground]))
    bone = foreground & (norm < np.percentile(norm[foreground], 20))

    t2 = T2_MARROW_MS + (cartilage_t2_ms(age_years) - T2_MARROW_MS) * cartilage
    t2 = np.where(bone, T2_BONE_MS, t2)

    decay = np.exp(-TE_MS / t2) / np.exp(-TE_MS / T2_MARROW_MS)
    out = norm * (1.0 - T2_WEIGHTING + T2_WEIGHTING * decay)
    return (out * (vmax - vmin) + vmin).astype(np.float32)


def enhance_synthetic_knee(vol, age_years, rng=None):
    """Make a raw synthetic [D, H, W] knee volume biologically and physically plausible."""
    vol = np.asarray(vol, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected a [D, H, W] volume, got shape {vol.shape}")

    age_years = float(age_years)
    lo, hi = float(vol.min()), float(vol.max())
    anatomy = _knee_anatomy(_normalize(vol)[0], age_years)

    out = apply_biological_constraints(vol, age_years, anatomy)
    out = apply_mri_physics(out, age_years, anatomy)

    rng = rng or np.random.default_rng()
    out += rng.normal(0.0, NOISE_FRACTION * (hi - lo), out.shape).astype(np.float32)
    return np.clip(out, lo, hi).astype(np.float32)
