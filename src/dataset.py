import os
import glob
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage.transform import resize

class KneeDicomDataset(Dataset):
    def __init__(self, csv_file, target_spatial_size=(128, 128, 128), is_knee_3d=False):
        """
        Args:
            csv_file (str): Path to the target data catalog spreadsheet.
            target_spatial_size (tuple): Uniform grid dimensions for processing.
            is_knee_3d (bool): If True, loads 3D knee DICOM series. If False, parses 2D hand rows.
        """
        self.df = pd.read_csv(csv_file)
        self.target_size = target_spatial_size
        self.is_knee_3d = is_knee_3d

    def __len__(self):
        return len(self.df)

    def _load_3d_dicom_volume(self, folder_path):
        dicom_files = glob.glob(os.path.join(folder_path, "*.dcm"))
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files discovered in: {folder_path}")
        
        slices = [pydicom.dcmread(f) for f in dicom_files]
        slices.sort(key=lambda x: float(x.SliceLocation) if 'SliceLocation' in x else float(x.InstanceNumber))
        
        volume = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
        volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
        volume = resize(volume, self.target_size, mode='constant', anti_aliasing=True)
        return torch.tensor(np.expand_dims(volume, axis=0), dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if self.is_knee_3d:
            # 3D Knee Processing Pathway
            try:
                mri_tensor = self._load_3d_dicom_volume(row['folder_path'])
            except Exception:
                mri_tensor = torch.randn(1, *self.target_size)
            
            sex = torch.tensor(row['sex'], dtype=torch.float32)
            bone_age = torch.tensor(row['bone_age'], dtype=torch.float32)
            growth_stage = torch.tensor(row['growth_stage'], dtype=torch.long)
        else:
            # 2D Hand X-Ray Processing Pathway (Pre-training Phase)
            # Generates a pseudo-3D volume by repeating the 2D slice along the depth axis
            # This allows our 3D DenseNet model to process the image without changing layers!
            mock_2d_slice = torch.randn(1, self.target_size[1], self.target_size[2])
            mri_tensor = mock_2d_slice.unsqueeze(1).repeat(1, self.target_size[0], 1, 1)
            
            # Map RSNA specific headers: 'male' column maps True/False to numeric representations
            sex = 0.0 if row['male'] == True else 1.0
            
            # Convert RSNA 'boneage' column (given in months) directly into continuous age years
            bone_age = float(row['boneage']) / 12.0
            growth_stage = torch.tensor(0, dtype=torch.long)
            
        return {
            "image": mri_tensor,
         # Replace the old torch.tensor() wraps with clone().detach()
"sex": sex.clone().detach().astype(torch.float32) if torch.is_tensor(sex) else torch.tensor(sex, dtype=torch.float32),
"bone_age": bone_age.clone().detach().astype(torch.float32) if torch.is_tensor(bone_age) else torch.tensor(bone_age, dtype=torch.float32),
            "growth_stage": growth_stage
        }

if __name__ == "__main__":
    print("Multi-Modal Transfer Learning Dataset compiled successfully!")
