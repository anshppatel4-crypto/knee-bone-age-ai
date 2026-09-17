"""Reading and writing standards-compliant MR DICOM series.

The previous writer produced files without the 128-byte preamble or the 'DICM'
marker, so pydicom refused to open them without force=True, and it rescaled every
slice independently, which destroyed intensity relationships between slices.
"""
import os
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

# Sagittal acquisition: rows run superior->inferior, columns anterior->posterior
SAGITTAL_ORIENTATION = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
UINT16_MAX = 65535


def _save(dataset, path):
    try:  # pydicom >= 3
        dataset.save_as(path, enforce_file_format=True)
    except TypeError:  # pydicom 2.x
        dataset.save_as(path, write_like_original=False)


def write_mr_series(volume, output_dir, age_years, sex="M", pixel_spacing_mm=0.5,
                    slice_thickness_mm=3.0, series_description="PD FS SAG",
                    repetition_time=2600.0, echo_time=39.0, patient_id=None):
    """Write a [D, H, W] volume as a valid sagittal MR DICOM series.

    The volume is scaled to 16-bit once, globally, so relative slice brightness survives.
    """
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a [D, H, W] volume, got shape {volume.shape}")

    os.makedirs(output_dir, exist_ok=True)
    lo, hi = float(volume.min()), float(volume.max())
    scaled = ((volume - lo) / (hi - lo + 1e-8) * UINT16_MAX).astype(np.uint16)

    study_uid, series_uid = generate_uid(), generate_uid()
    patient_id = patient_id or os.path.basename(os.path.normpath(output_dir))
    depth = scaled.shape[0]

    for index in range(depth):
        file_meta = FileMetaDataset()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()

        path = os.path.join(output_dir, f"slice_{index:03d}.dcm")
        ds = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)

        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "MR"
        ds.SeriesDescription = series_description
        ds.BodyPartExamined = "KNEE"
        ds.PatientID = ds.PatientName = patient_id
        ds.PatientSex = sex
        ds.PatientAge = f"{int(round(age_years * 12)):03d}M"
        ds.RepetitionTime = repetition_time
        ds.EchoTime = echo_time
        ds.MagneticFieldStrength = 3.0
        ds.ScanningSequence = "SE"

        # Geometry: slices step along x (medial->lateral) for a sagittal series
        ds.ImageOrientationPatient = SAGITTAL_ORIENTATION
        offset = (index - (depth - 1) / 2) * slice_thickness_mm
        ds.ImagePositionPatient = [offset, -0.5 * pixel_spacing_mm * scaled.shape[1],
                                   0.5 * pixel_spacing_mm * scaled.shape[2]]
        ds.SliceLocation = offset
        ds.InstanceNumber = index + 1
        ds.PixelSpacing = [pixel_spacing_mm, pixel_spacing_mm]
        ds.SliceThickness = slice_thickness_mm
        ds.SpacingBetweenSlices = slice_thickness_mm

        ds.Rows, ds.Columns = scaled.shape[1], scaled.shape[2]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = scaled[index].tobytes()

        _save(ds, path)

    return output_dir


def read_mr_series(folder_path):
    """Load a DICOM folder as a [D, H, W] float array ordered along the true slice normal."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Not a DICOM folder: {folder_path}")

    slices = []
    for name in sorted(os.listdir(folder_path)):
        try:
            ds = pydicom.dcmread(os.path.join(folder_path, name), force=True)
            if "PixelData" in ds:
                slices.append(ds)
        except Exception:
            continue

    if not slices:
        raise FileNotFoundError(f"No readable DICOM images in: {folder_path}")

    # Keep the dominant image size so localisers and stray series drop out
    shapes = [tuple(s.pixel_array.shape) for s in slices]
    primary_shape = max(set(shapes), key=shapes.count)
    slices = [s for s, shape in zip(slices, shapes) if shape == primary_shape]

    slices.sort(key=_slice_position)
    volume = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
    return volume, slices[0]


def _slice_position(ds):
    """Position along the slice normal, so slices sort correctly in any orientation."""
    orientation = ds.get("ImageOrientationPatient")
    position = ds.get("ImagePositionPatient")
    if orientation is not None and position is not None:
        try:
            normal = np.cross(np.array(orientation[:3], dtype=float),
                              np.array(orientation[3:], dtype=float))
            return float(np.dot(np.array(position, dtype=float), normal))
        except (TypeError, ValueError):
            pass

    for attribute in ("SliceLocation", "InstanceNumber"):
        value = ds.get(attribute)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0
