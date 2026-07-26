import os
import torch
import numpy as np
import pydicom
from diffusers import StableDiffusionPipeline
from scipy.ndimage import gaussian_filter1d
from pydicom.dataset import FileDataset, Dataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

# =========================================================================
# HELPER 1: WRITING INDIVIDUAL CLINICAL DICOM FILES
# =========================================================================
def save_single_dicom_slice(pixel_array_2d, output_path, slice_idx, spacing_mm=3.0, patient_sex="M", target_age=12.5):
    """Formats a raw 2D pixel image into an authentic medical DICOM file."""
    # Scale image array to standard 16-bit unsigned integers used by scanners
    pixel_array_2d = ((pixel_array_2d - pixel_array_2d.min()) / (pixel_array_2d.max() - pixel_array_2d.min() + 1e-8) * 65535).astype(np.uint16)
    
    # Instantiate standard DICOM metadata frameworks
    file_meta = Dataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage Code
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    
    ds = FileDataset(output_path, {}, file_meta=file_meta, local_media_storage_sop_class_uid='1.2.840.10008.5.1.4.1.1.4')
    
    # Inject exact physical metadata required by your dataset.py script
    ds.SliceLocation = float(slice_idx * spacing_mm)  # Track depth position precisely
    ds.PatientSex = patient_sex
    ds.PatientAge = f"{int(target_age * 12):03d}M"  # Age formatted in months
    
    # Set explicit resolution configurations
    ds.Rows, ds.Columns = pixel_array_2d.shape
    ds.PixelData = pixel_array_2d.tobytes()
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    
    ds.save_as(output_path)


# =========================================================================
# HELPER 2: THE 2.5D CROSS-SECTIONAL VOLUMETRIC SYNTHESIS LOOP
# =========================================================================
def execute_25d_volumetric_generation(target_age=13.5, patient_sex="M", total_slices=8, output_dir="data/synthetic_knee_001"):
    """Generates structural knee frames sequentially and assembles them into an

    orderly 3D DICOM dataset stack folder.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Download model weights from the Hugging Face hub mirror
    print("🔄 Downloading pre-trained Stable Diffusion weights into workspace...")
    model_id = "Manojb/stable-diffusion-2-1-base"
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipeline.to(device)
    
    generated_raw_frames = []
    sex_label = "male" if patient_sex.upper() == "M" else "female"
    
    # 2. Iterate slice-by-slice along the joint's depth track
    print(f"🎨 Generating {total_slices} cross-sectional MRI layers for a {target_age}-year-old...")
    for z in range(total_slices):
        # The text prompt dynamically targets depth positions so the anatomy morphs logically
        prompt = (f"Sagittal structural knee MRI slice, depth position {z} out of {total_slices}, "
                  f"pediatric patient, biological age {target_age} years old, {sex_label}, "
                  f"high contrast, monochrome medical imaging scan, bone structure slice")
        
        negative_prompt = "color, text, watermarks, bad anatomy, corrupted pixels, artifacts, 3d render"
        
        # Synthesize a single 2D image plane 
        with torch.no_grad():
            image = pipeline(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=20).images[0]
            
        # Strip color channels and map pixels to a 2D float matrix array
        grayscale_image = image.convert("L")
        pixel_array_2d = np.array(grayscale_image).astype(np.float32)
        generated_raw_frames.append(pixel_array_2d)
        print(f"   ↳ Layer {z+1}/{total_slices} complete.")

    # 3. Apply the 2.5D Volumetric Filter (Smooths out transitions between slices)
    # This prevents sharp, disconnected shapes between adjacent slices.
    print("📐 Applying 1D Gaussian Depth smoothing filter to unify bone structures...")
    volume_matrix = np.stack(generated_raw_frames, axis=0)  # Shape: (Slices, Height, Width)
    smoothed_volume_matrix = gaussian_filter1d(volume_matrix, sigma=1.0, axis=0)
    
    # 4. Save each processed frame directly out into an uncorrupted clinical folder
    print(f"💾 Committing processed volumes safely into raw DICOM binaries...")
    for z in range(total_slices):
        target_file_name = os.path.join(output_dir, f"slice_{z:03d}.dcm")
        save_single_dicom_slice(
            pixel_array_2d=smoothed_volume_matrix[z, :, :],
            output_path=target_file_name,
            slice_idx=z,
            patient_sex=patient_sex,
            target_age=target_age
        )
        
    print(f"✅ Success! Your valid 3D pediatric knee DICOM folder is fully populated at: {output_dir}")

if __name__ == "__main__":
    # Test your new generation engine by creating an initial 8-slice target volume
    execute_25d_volumetric_generation(target_age=12.5, patient_sex="M", total_slices=8)
