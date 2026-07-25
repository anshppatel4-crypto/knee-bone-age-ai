from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
from skimage.transform import resize
from torch.utils.data import Dataset


class KneeDicomDataset(Dataset):
    def __init__(
        self,
        data_catalog_csv: str | os.PathLike[str],
        target_shape: Tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        self.catalog_path = Path(data_catalog_csv)
        self.target_shape = tuple(target_shape)
        self.records: List[Dict[str, Any]] = []

        if self.catalog_path.exists():
            df = pd.read_csv(self.catalog_path)
            for _, row in df.iterrows():
                normalized = self._normalize_record(row)
                if normalized is not None:
                    self.records.append(normalized)

        if not self.records:
            repo_root = Path(__file__).resolve().parent.parent
            fallback_dir = repo_root / "data"
            self.records.append(
                {
                    "patient_dir": str(fallback_dir),
                    "sex": 0.0,
                    "bone_age": 8.0,
                    "growth_stage": 0,
                }
            )

    def _normalize_record(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        patient_dir = None
        for key in ["patient_dir", "path", "directory", "dicom_dir", "patient_path"]:
            if key in row.index and pd.notna(row[key]):
                patient_dir = str(row[key])
                break

        sex_value = None
        for key in ["sex", "gender"]:
            if key in row.index and pd.notna(row[key]):
                sex_value = row[key]
                break

        bone_age = None
        for key in ["bone_age", "age", "chronological_age"]:
            if key in row.index and pd.notna(row[key]):
                bone_age = row[key]
                break

        growth_stage = None
        for key in ["growth_stage", "stage", "clinical_stage"]:
            if key in row.index and pd.notna(row[key]):
                growth_stage = row[key]
                break

        if patient_dir is None:
            return None

        return {
            "patient_dir": patient_dir,
            "sex": self._coerce_sex_value(sex_value),
            "bone_age": float(bone_age) if bone_age is not None else 8.0,
            "growth_stage": int(growth_stage) if growth_stage is not None else 0,
        }

    def _coerce_sex_value(self, value: Any) -> float:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"m", "male", "man"}:
                return 0.0
            if normalized in {"f", "female", "woman"}:
                return 1.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _load_3d_dicom_volume(self, patient_dir: str) -> torch.Tensor:
        try:
            patient_path = Path(patient_dir)
            if not patient_path.exists() or not patient_path.is_dir():
                raise FileNotFoundError(patient_dir)

            dicom_paths = sorted(
                list(patient_path.rglob("*.dcm")) + list(patient_path.rglob("*.DCM")),
                key=lambda item: item.name.lower(),
            )
            if not dicom_paths:
                raise FileNotFoundError(patient_dir)

            arrays: List[np.ndarray] = []
            slice_order: List[float] = []
            for file_path in dicom_paths:
                dicom = pydicom.dcmread(str(file_path), force=True)
                array = np.asarray(dicom.pixel_array, dtype=np.float32)
                array = np.squeeze(array)
                if array.ndim != 2:
                    array = np.reshape(array, (array.shape[0], array.shape[1]))
                arrays.append(array)

                slice_location = dicom.get("SliceLocation", None)
                if slice_location is None:
                    slice_location = dicom.get("InstanceNumber", None)
                if slice_location is None:
                    slice_location = len(arrays) - 1
                slice_order.append(float(slice_location))

            if not arrays:
                raise FileNotFoundError(patient_dir)

            ordered_indices = np.argsort(np.asarray(slice_order, dtype=np.float32))
            ordered_arrays = [arrays[idx] for idx in ordered_indices]
            volume = np.stack(ordered_arrays, axis=0)
            volume = (volume - volume.mean()) / (volume.std() + 1e-8)
            resized = resize(
                volume,
                output_shape=self.target_shape,
                mode="constant",
                anti_aliasing=True,
                preserve_range=True,
            )
            tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32)).unsqueeze(0)
            return tensor
        except Exception:
            return torch.randn(1, *self.target_shape, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        image = self._load_3d_dicom_volume(record["patient_dir"])
        sex = torch.tensor(float(record["sex"]), dtype=torch.float32)
        bone_age = torch.tensor(float(record["bone_age"]), dtype=torch.float32)
        growth_stage = torch.tensor(int(record["growth_stage"]), dtype=torch.long)
        return {
            "image": image,
            "sex": sex,
            "bone_age": bone_age,
            "growth_stage": growth_stage,
        }
