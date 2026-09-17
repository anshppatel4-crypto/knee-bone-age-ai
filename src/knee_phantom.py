"""Procedural 3D paediatric knee MRI phantom.

Stable Diffusion renders knee-like texture, not a knee: its slices are mutually
inconsistent and its growth plates are decorative. This module instead builds an
explicit 3D anatomical model in millimetres (femur, tibia, patella, fibula,
epiphyses, physes, cartilage, menisci, marrow, muscle, fat) whose geometry is
driven by skeletal maturity, then images it with a spin-echo signal equation.

Because the geometry generates the label, age supervision is exact.

Axes: [D, H, W] = medial->lateral, superior->inferior, anterior->posterior.
The joint line sits at y = 0.
"""
from __future__ import annotations
import argparse
import functools
import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from src.dicom_io import write_mr_series

# Spin-echo tissue properties at 3T: proton density, T1 (ms), T2 (ms), fat fraction
TISSUES = {
    "cortical_bone":  dict(pd=0.05, t1=250.0,  t2=0.5,   fat=0.0),
    "red_marrow":     dict(pd=0.85, t1=600.0,  t2=45.0,  fat=0.40),
    "yellow_marrow":  dict(pd=1.00, t1=370.0,  t2=130.0, fat=0.92),
    "cartilage":      dict(pd=0.90, t1=1200.0, t2=32.0,  fat=0.0),
    "physis":         dict(pd=0.95, t1=1400.0, t2=45.0,  fat=0.0),
    "fluid":          dict(pd=1.00, t1=3000.0, t2=200.0, fat=0.0),
    "muscle":         dict(pd=0.80, t1=1400.0, t2=30.0,  fat=0.05),
    "fat":            dict(pd=1.00, t1=370.0,  t2=130.0, fat=1.00),
    "fibrous":        dict(pd=0.35, t1=800.0,  t2=6.0,   fat=0.0),  # menisci, ligaments, tendon
    "skin":           dict(pd=0.70, t1=900.0,  t2=40.0,  fat=0.20),
}

# Clinical knee protocols; varying these stops the model keying on one contrast
SEQUENCES = {
    "PD FS SAG": dict(tr=2600.0, te=39.0, fat_sat=True),
    "T1 SAG":    dict(tr=600.0,  te=12.0, fat_sat=False),
    "T2 FS SAG": dict(tr=4000.0, te=80.0, fat_sat=True),
}

FAT_SAT_EFFICIENCY = 0.85

# Real knee studies frame the joint tightly; this fills a comparable share of the field of view
ANATOMY_FILL = 1.15


# ---------------------------------------------------------------------------
# Maturity curves
# ---------------------------------------------------------------------------
def skeletal_maturity(age_years, sex):
    """Girls mature roughly 1.8 years ahead of boys, which is why sex is a model input."""
    return age_years + (1.8 if str(sex).upper().startswith("F") else 0.0)


def physis_thickness_mm(maturity):
    return float(np.interp(maturity, [2, 8, 12, 14, 16, 17.5, 19], [3.2, 2.4, 1.8, 1.2, 0.6, 0.2, 0.0]))


def fusion_fraction(maturity):
    """Fraction of the physis already bridged by bone."""
    return float(1.0 / (1.0 + np.exp(-(maturity - 16.0) / 0.7)))


def ossified_fraction(maturity):
    """Share of the cartilaginous epiphysis replaced by an ossific nucleus."""
    return float(np.interp(maturity, [0, 2, 5, 8, 12, 15, 18], [0.20, 0.42, 0.62, 0.74, 0.86, 0.94, 1.0]))


def yellow_marrow_fraction(maturity):
    return float(np.interp(maturity, [2, 8, 12, 16, 20], [0.10, 0.28, 0.45, 0.65, 0.80]))


def bone_scale(age_years):
    """Linear size growth, normalised to adult dimensions at 18 years."""
    return float(np.interp(age_years, [2, 6, 10, 14, 18], [0.60, 0.72, 0.83, 0.94, 1.0]))


def growth_stage(maturity):
    """4-tier physeal closure stage matching the report text in predict.py."""
    fused = fusion_fraction(maturity)
    return int(np.digitize(fused, [0.05, 0.35, 0.90]))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _soft(distance_mm, width=0.45):
    """Occupancy in [0, 1] from a signed distance, giving partial-volume edges."""
    return 1.0 / (1.0 + np.exp(np.clip(distance_mm / width, -60.0, 60.0)))


def _ellipsoid(x, y, z, center, radii):
    """Approximate signed distance (mm) to an ellipsoid surface."""
    normalised = (((x - center[0]) / radii[0]) ** 2 +
                  ((y - center[1]) / radii[1]) ** 2 +
                  ((z - center[2]) / radii[2]) ** 2)
    return (np.sqrt(normalised) - 1.0) * float(min(radii))


def _smooth_field(rng, shape, scale=6.0):
    """Low-frequency random field for marrow patchiness and bias fields."""
    coarse = rng.normal(size=(max(2, shape[0] // 8), 5, 5))
    field = np.zeros(shape, dtype=np.float32)
    grids = [np.linspace(0, 1, s, dtype=np.float32) for s in shape]
    for k in range(3):
        for m in range(3):
            for n in range(3):
                amplitude = coarse[k % coarse.shape[0], m, n] / (1 + k + m + n)
                field += amplitude * (np.cos(np.pi * (k + 1) * grids[0])[:, None, None] *
                                      np.cos(np.pi * (m + 1) * grids[1])[None, :, None] *
                                      np.cos(np.pi * (n + 1) * grids[2])[None, None, :])
    return field / (np.abs(field).max() + 1e-8) * scale


# ---------------------------------------------------------------------------
# Phantom construction
# ---------------------------------------------------------------------------
def build_knee_phantom(age_years, sex="M", rng=None, shape=(32, 256, 256),
                       pixel_spacing_mm=0.5, slice_thickness_mm=3.0, sequence="PD FS SAG",
                       slice_supersample=3, return_components=False):
    """Render one synthetic knee. Returns (volume [D, H, W], metadata dict).

    With return_components=True the metadata carries the individual tissue masks,
    slice-averaged onto the output grid, which is what the geometry debug view uses.
    """
    rng = rng or np.random.default_rng()
    depth, height, width = shape
    maturity = skeletal_maturity(age_years, sex) + rng.normal(0, 0.35)  # biological spread
    # Children of the same age differ widely in size, and a +-5% spread made size a
    # cleaner age cue than the growth plate, so the model learned to ignore sex.
    # Wide jitter (~6 years' worth of size) forces it onto maturity features instead.
    scale = bone_scale(age_years) * rng.uniform(0.86, 1.14) * ANATOMY_FILL

    # Fine medial-lateral sampling, averaged down later to model thick slices
    fine_depth = depth * slice_supersample
    fine_dx = slice_thickness_mm / slice_supersample
    x = ((np.arange(fine_depth) - (fine_depth - 1) / 2) * fine_dx)[:, None, None].astype(np.float32)
    y = ((np.arange(height) - (height - 1) / 2) * pixel_spacing_mm)[None, :, None].astype(np.float32)
    z = ((np.arange(width) - (width - 1) / 2) * pixel_spacing_mm)[None, None, :].astype(np.float32)

    # Random patient pose
    y = y - rng.uniform(-8, 8)
    z = z - rng.uniform(-6, 6)
    fine_shape = (fine_depth, height, width)

    physis_t = physis_thickness_mm(maturity) * rng.uniform(0.85, 1.15)
    fused = fusion_fraction(maturity)
    ossified = ossified_fraction(maturity)

    # --- Femur: condyles (epiphysis) below the physis, metaphysis and shaft above
    condyle_offset = 13.0 * scale
    condyle_radii = (14.0 * scale, 20.0 * scale, 26.0 * scale)
    condyle_y, condyle_z = -20.0 * scale, 4.0 * scale
    femoral_physis_y = -36.0 * scale
    undulation = (1.2 * scale * np.sin(z / (9.0 * scale) + rng.uniform(0, 6.3)) *
                  np.cos(x / (11.0 * scale) + rng.uniform(0, 6.3)))
    physis_surface_femur = femoral_physis_y + undulation

    condyles = np.minimum(
        _ellipsoid(x, y, z, (-condyle_offset, condyle_y, condyle_z), condyle_radii),
        _ellipsoid(x, y, z, (condyle_offset, condyle_y, condyle_z), condyle_radii))
    femoral_epiphysis = _soft(condyles) * _soft(physis_surface_femur - y)

    # Shaft narrows proximally and flares into the metaphysis just above the physis
    shaft_radius = (12.0 + 15.0 * np.clip((y + 62.0 * scale) / (26.0 * scale), 0, 1)) * scale
    femoral_shaft_d = np.sqrt(x ** 2 + (z - condyle_z) ** 2) - shaft_radius
    femoral_meta = _soft(femoral_shaft_d) * _soft(y - (physis_surface_femur - physis_t))

    # --- Tibia: plateau epiphysis above its physis, metaphysis and shaft below
    tibial_physis_y = 30.0 * scale
    plateau_center = (0.0, 22.0 * scale, 3.0 * scale)
    plateau_radii = (28.0 * scale, 15.0 * scale, 26.0 * scale)
    physis_surface_tibia = tibial_physis_y + 0.8 * scale * np.sin(z / (10.0 * scale) + rng.uniform(0, 6.3))
    plateau = _ellipsoid(x, y, z, plateau_center, plateau_radii)
    tibial_epiphysis = _soft(plateau) * _soft(y - physis_surface_tibia)

    tibial_radius = (13.0 + 14.0 * np.clip((50.0 * scale - y) / (22.0 * scale), 0, 1)) * scale
    tibial_shaft_d = np.sqrt(x ** 2 + (z - 2.0 * scale) ** 2) - tibial_radius
    tibial_meta = _soft(tibial_shaft_d) * _soft((physis_surface_tibia + physis_t) - y)

    # Tibial tuberosity (anterior apophysis), ossifying through adolescence
    tuberosity = _soft(_ellipsoid(x, y, z, (0.0, 42.0 * scale, -24.0 * scale),
                                  (8.0 * scale, 12.0 * scale, 7.0 * scale)))
    tibial_meta = np.maximum(tibial_meta, tuberosity * min(1.0, 0.3 + 0.7 * ossified))

    # --- Patella and fibular head
    patella_center = (0.0, -18.0 * scale, -34.0 * scale)
    patella_radii = (16.0 * scale, 16.0 * scale, 9.0 * scale)
    patella_d = _ellipsoid(x, y, z, patella_center, patella_radii)
    patella = _soft(patella_d)
    fibula_d = np.sqrt((x - 26.0 * scale) ** 2 + (z - 20.0 * scale) ** 2) - 7.0 * scale
    fibula = _soft(fibula_d) * _soft(34.0 * scale - y)

    epiphysis = np.clip(femoral_epiphysis + tibial_epiphysis + patella, 0, 1)
    metaphysis = np.clip(femoral_meta + tibial_meta + fibula, 0, 1)

    # --- Ossification: an ossific nucleus inside a cartilage epiphysis, growing with age
    nucleus_femur = np.minimum(
        _ellipsoid(x, y, z, (-condyle_offset, condyle_y, condyle_z), tuple(r * ossified for r in condyle_radii)),
        _ellipsoid(x, y, z, (condyle_offset, condyle_y, condyle_z), tuple(r * ossified for r in condyle_radii)))
    nucleus_tibia = _ellipsoid(x, y, z, plateau_center, tuple(r * ossified for r in plateau_radii))
    nucleus_patella = _ellipsoid(x, y, z, patella_center, tuple(r * ossified for r in patella_radii))
    irregular = _smooth_field(rng, fine_shape, scale=1.2 * (1.0 - ossified) * 8.0)
    ossific_nucleus = _soft(np.minimum(np.minimum(nucleus_femur, nucleus_tibia), nucleus_patella) + irregular)

    ossified_bone = np.clip(metaphysis + epiphysis * ossific_nucleus, 0, 1)
    epiphyseal_cartilage = np.clip(epiphysis - epiphysis * ossific_nucleus, 0, 1)

    # --- Physes: cartilage plates, progressively bridged then replaced by a scar
    def plate(surface, half_thickness):
        return _soft(np.abs(y - surface) - half_thickness, width=0.35)

    # Each plate spans its own bone's cross-section
    physis_band = np.clip(plate(physis_surface_femur, physis_t / 2) * _soft(femoral_shaft_d) +
                          plate(physis_surface_tibia, physis_t / 2) * _soft(tibial_shaft_d), 0, 1)
    bridging = _soft(_smooth_field(rng, fine_shape, scale=3.0) + (0.5 - fused) * 6.0)
    open_physis = physis_band * (1.0 - bridging)
    physeal_scar = physis_band * bridging

    # --- Articular cartilage and joint fluid
    cartilage_t = float(np.interp(maturity, [2, 8, 14, 18], [3.6, 2.8, 2.2, 1.8])) * rng.uniform(0.9, 1.1)
    femoral_surface = _soft(condyles - cartilage_t) - _soft(condyles)
    tibial_surface = _soft(plateau - cartilage_t) - _soft(plateau)
    # Cartilage coats only the joint-facing side of each bone
    articular = np.clip(femoral_surface * _soft(-(y + 10.0 * scale)) +
                        tibial_surface * _soft(y - 16.0 * scale), 0, 1)

    # Fluid only exists inside the joint: within the tibial plateau silhouette, between the
    # articular surfaces, and never in bone or soft tissue
    joint_silhouette = _soft((np.sqrt((x / (28.0 * scale)) ** 2 +
                                      ((z - 3.0 * scale) / (25.0 * scale)) ** 2) - 1.0) * 25.0 * scale)
    joint_gap = (_soft(np.abs(y - 3.5 * scale) - 2.0 * scale) * joint_silhouette *
                 (1.0 - np.clip(ossified_bone + articular + epiphyseal_cartilage, 0, 1)))
    effusion = rng.uniform(0.0, 1.0) < 0.35
    # Fluid hugs the curved condylar surface rather than forming a straight band
    fluid = np.clip(joint_gap * _soft(condyles - 7.0 * scale) * (1.4 if effusion else 1.0), 0, 1)

    # --- Menisci, patellar tendon, cruciates
    meniscus_ring = _soft(np.abs(np.sqrt(x ** 2 + (z - 3.0 * scale) ** 2) - 22.0 * scale) - 7.0 * scale)
    menisci = meniscus_ring * _soft(np.abs(y - 4.0 * scale) - 3.0 * scale)
    # Patellar tendon runs obliquely from the patella down to the tuberosity
    tendon_axis = z + (34.0 * scale) - 0.12 * (y + 18.0 * scale)
    tendon = (_soft(np.abs(tendon_axis) - 2.6 * scale, width=0.5) * _soft(np.abs(x) - 9.0 * scale) *
              _soft(np.abs(y - 14.0 * scale) - 28.0 * scale))
    cruciate = _soft(_ellipsoid(x, y, z, (0.0, -6.0 * scale, 14.0 * scale),
                                (5.0 * scale, 18.0 * scale, 5.0 * scale)))
    # Ligaments and tendons run outside bone, never through it
    fibrous = np.clip(menisci + tendon + cruciate, 0, 1) * (1.0 - ossified_bone)

    # --- Soft-tissue envelope: muscle core, subcutaneous fat rim, skin
    # The limb tapers at the knee, widening into thigh above and calf below
    thigh = np.clip((-y - 24.0 * scale) / (44.0 * scale), 0, 1)
    calf = np.clip((y - 26.0 * scale) / (40.0 * scale), 0, 1)
    ml_radius = (42.0 + 9.0 * thigh + 7.0 * calf) * scale
    ap_radius = (44.0 + 12.0 * thigh + 14.0 * calf) * scale
    ap_center = (2.0 + 9.0 * calf) * scale  # the calf bulges posteriorly
    limb_d = (np.sqrt((x / ml_radius) ** 2 + ((z - ap_center) / ap_radius) ** 2) - 1.0) * ml_radius
    limb = _soft(limb_d, width=0.8)
    fat_thickness = rng.uniform(4.0, 9.0) * scale
    subcutaneous = limb - _soft(limb_d + fat_thickness, width=0.8)
    skin = limb - _soft(limb_d + 1.2, width=0.5)

    # --- Cortex and marrow
    # Cortex is the shell just inside each bone surface: thick along the shafts,
    # thin (subchondral) around the epiphyses and the ossific nucleus
    # functools.reduce broadcasts pairwise; np.minimum.reduce would need identical shapes
    bone_distance = functools.reduce(np.minimum, [condyles, femoral_shaft_d, plateau,
                                                  tibial_shaft_d, patella_d, fibula_d])
    cortex_t = np.where(np.abs(y) > 45.0 * scale, 3.0, 1.6) * scale
    outer_shell = (_soft(bone_distance) - _soft(bone_distance + cortex_t)) * ossified_bone
    nucleus_distance = functools.reduce(np.minimum, [nucleus_femur, nucleus_tibia, nucleus_patella])
    nucleus_shell = (_soft(nucleus_distance) - _soft(nucleus_distance + 0.9 * scale)) * epiphysis

    cortex = np.clip(outer_shell + nucleus_shell, 0, 1)
    marrow = np.clip(ossified_bone - cortex, 0, 1)

    yellow = np.clip(yellow_marrow_fraction(maturity) +
                     0.18 * _smooth_field(rng, fine_shape, scale=1.0) +
                     0.15 * epiphysis - 0.10 * metaphysis, 0.0, 1.0)

    # --- Composite tissue weights
    occupied = np.clip(ossified_bone + epiphyseal_cartilage + open_physis + physeal_scar +
                       articular + fluid + fibrous, 0, 1)
    soft_tissue = np.clip(limb - occupied, 0, 1)
    muscle = np.clip(soft_tissue - subcutaneous - skin, 0, 1)

    layers = [
        (cortex, "cortical_bone"),
        (marrow * (1.0 - yellow), "red_marrow"),
        (marrow * yellow, "yellow_marrow"),
        (epiphyseal_cartilage, "cartilage"),
        (articular, "cartilage"),
        (open_physis, "physis"),
        (physeal_scar * 0.6, "cortical_bone"),
        (physeal_scar * 0.4, "red_marrow"),
        (fluid, "fluid"),
        (fibrous, "fibrous"),
        (muscle, "muscle"),
        (np.clip(subcutaneous - skin, 0, 1), "fat"),
        (skin, "skin"),
    ]

    protocol = SEQUENCES[sequence]
    volume = np.zeros(fine_shape, dtype=np.float32)
    total_weight = np.zeros(fine_shape, dtype=np.float32)
    for weight, tissue in layers:
        weight = weight.astype(np.float32)
        volume += weight * _tissue_signal(TISSUES[tissue], protocol, rng)
        total_weight += weight
    volume /= np.maximum(total_weight, 1.0)  # keep overlapping structures from over-brightening

    # Trabecular marrow and muscle striation, so tissues are not flat grey
    volume *= 1.0 + 0.11 * _texture(rng, fine_shape, sigma=1.2) * np.broadcast_to(marrow, fine_shape)
    volume *= 1.0 + 0.07 * _texture(rng, fine_shape, sigma=2.5) * np.broadcast_to(muscle, fine_shape)

    # --- Acquisition physics: thick slices, PSF blur, bias field, Rician noise
    volume = volume.reshape(depth, slice_supersample, height, width).mean(axis=1)
    volume = _blur(volume, sigma_voxels=0.9)
    volume *= 1.0 + 0.12 * _smooth_field(rng, volume.shape, scale=1.0)

    sigma = rng.uniform(0.010, 0.030) * float(volume.max())
    volume = np.sqrt((volume + rng.normal(0, sigma, volume.shape)) ** 2 +
                     rng.normal(0, sigma, volume.shape) ** 2).astype(np.float32)

    if rng.random() < 0.5:  # left vs right knee
        volume = volume[::-1].copy()

    metadata = {
        "bone_age": round(float(age_years), 3),
        "sex": 1.0 if str(sex).upper().startswith("M") else 0.0,
        "growth_stage": growth_stage(maturity),
        "maturity": round(float(maturity), 3),
        "fusion_fraction": round(fused, 3),
        "sequence": sequence,
        "pixel_spacing_mm": pixel_spacing_mm,
        "slice_thickness_mm": slice_thickness_mm,
    }
    if return_components:
        metadata["components"] = {
            # Masks built from 1-D coordinate arrays are only broadcast-shaped, so expand first
            name: np.broadcast_to(weight, fine_shape).reshape(depth, slice_supersample, height, width).mean(axis=1)
            for weight, name in [(ossified_bone, "ossified_bone"), (cortex, "cortex"), (marrow, "marrow"),
                                 (epiphyseal_cartilage, "epiphyseal_cartilage"), (articular, "articular"),
                                 (open_physis, "open_physis"), (physeal_scar, "physeal_scar"),
                                 (fluid, "fluid"), (fibrous, "fibrous"), (muscle, "muscle"),
                                 (subcutaneous, "subcutaneous_fat"), (limb, "limb")]
        }
    return volume, metadata


def _tissue_signal(tissue, protocol, rng):
    """Spin-echo signal with optional fat saturation, plus small per-scan variation."""
    pd = tissue["pd"] * rng.uniform(0.95, 1.05)
    signal = pd * (1.0 - np.exp(-protocol["tr"] / tissue["t1"])) * np.exp(-protocol["te"] / tissue["t2"])
    if protocol["fat_sat"]:
        signal *= 1.0 - FAT_SAT_EFFICIENCY * tissue["fat"]
    return float(signal)


def _texture(rng, shape, sigma=1.2):
    """Zero-mean smoothed noise for tissue micro-structure."""
    noise = rng.normal(size=shape).astype(np.float32)
    noise = gaussian_filter1d(noise, sigma, axis=1, mode="nearest")
    noise = gaussian_filter1d(noise, sigma, axis=2, mode="nearest")
    return np.clip(noise / (noise.std() + 1e-8), -3.0, 3.0)


def _blur(volume, sigma_voxels=0.9):
    """Separable Gaussian blur in-plane to emulate the point spread function."""
    blurred = gaussian_filter1d(volume, sigma_voxels, axis=1, mode="nearest")
    blurred = gaussian_filter1d(blurred, sigma_voxels, axis=2, mode="nearest")
    return blurred.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def generate_dataset(count=200, output_root="data/phantom_knees", seed=0,
                     age_range=(3.0, 18.0), shape=(32, 256, 256), preview=True):
    """Generate a labelled cohort of phantom knees as valid DICOM series."""
    rng = np.random.default_rng(seed)
    os.makedirs(output_root, exist_ok=True)
    sequences = list(SEQUENCES)
    records = []

    for index in range(count):
        age = float(rng.uniform(*age_range))
        sex = "M" if rng.random() < 0.5 else "F"
        sequence = sequences[int(rng.integers(len(sequences)))]

        volume, metadata = build_knee_phantom(age, sex, rng=rng, shape=shape, sequence=sequence)
        folder = os.path.join(output_root, f"knee_{index:04d}_age_{age:04.1f}_{sex}")
        write_mr_series(volume, folder, age_years=age, sex=sex,
                        pixel_spacing_mm=metadata["pixel_spacing_mm"],
                        slice_thickness_mm=metadata["slice_thickness_mm"],
                        series_description=sequence,
                        repetition_time=SEQUENCES[sequence]["tr"],
                        echo_time=SEQUENCES[sequence]["te"])

        records.append({"folder_path": folder.replace("\\", "/"), **metadata})
        if (index + 1) % 25 == 0 or index == count - 1:
            print(f"   ↳ {index + 1}/{count} phantoms written")

        if preview and index < 3:
            _save_preview(volume, os.path.join(output_root, f"preview_{index:02d}_age_{age:04.1f}_{sex}.png"))

    catalog = pd.DataFrame(records)
    catalog_path = os.path.join(output_root, "labels.csv")
    catalog.to_csv(catalog_path, index=False)
    print(f"✅ {len(catalog)} phantom knees written to {output_root} (catalog: {catalog_path})")
    return catalog


def _save_preview(volume, path, columns=4):
    """Write a PNG montage so the anatomy can be eyeballed without a DICOM viewer."""
    try:
        from PIL import Image
    except ImportError:
        return

    indices = np.linspace(0, volume.shape[0] - 1, columns).astype(int)
    tiles = []
    for index in indices:
        slice_ = volume[index]
        slice_ = np.clip(slice_ / (np.percentile(slice_, 99.5) + 1e-8), 0, 1) * 255
        tiles.append(slice_.astype(np.uint8))

    montage = np.concatenate(tiles, axis=1)
    Image.fromarray(montage).save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate anatomical knee MRI phantoms")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--out", type=str, default="data/phantom_knees")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slices", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()

    generate_dataset(count=args.count, output_root=args.out, seed=args.seed,
                     shape=(args.slices, args.resolution, args.resolution))
