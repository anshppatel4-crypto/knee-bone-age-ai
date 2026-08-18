import sys, os
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from scipy.ndimage import gaussian_filter1d
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.generate_25d import save_single_dicom_slice

def build_local_cohort():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using local hardware device for generation: {device}")
    
    # Load the cached model weights
    model_id = "Manojb/stable-diffusion-2-1-base"
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipeline.to(device)
    
    # 3 distinct patient profiles to give the AI an age progression timeline to learn
    cohort_config = [
        {"age": 11.5, "sex": "F", "dir": "data/cohort_pt_1"},
        {"age": 13.5, "sex": "M", "dir": "data/cohort_pt_2"},
        {"age": 15.5, "sex": "M", "dir": "data/cohort_pt_3"}
    ]
    
    total_slices = 16
    
    for pt in cohort_config:
        print(f"\n🧬 Synthesizing Patient Scan: Age {pt['age']} | Sex {pt['sex']}...")
        os.makedirs(pt["dir"], exist_ok=True)
        generated_frames = []
        
        sex_label = "male" if pt["sex"] == "M" else "female"
        
        for z in range(total_slices):
            prompt = (f"Sagittal structural knee MRI slice, depth position {z} out of {total_slices}, "
                      f"pediatric patient, biological age {pt['age']} years old, {sex_label}, "
                      f"high contrast, monochrome medical imaging scan, bone structure slice")
            negative_prompt = "color, text, watermarks, bad anatomy, corrupted pixels, artifacts, 3d render"
            
            with torch.no_grad():
                image = pipeline(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=15).images[0]
            
            grayscale_image = image.convert("L")
            generated_frames.append(np.array(grayscale_image).astype(np.float32))
            print(f"   ↳ Slice {z+1}/{total_slices} rendered.")
            
        # Apply your 2.5D smoothing physics filter
        volume_matrix = np.stack(generated_frames, axis=0)
        smoothed_volume = gaussian_filter1d(volume_matrix, sigma=1.0, axis=0)
        
        # Save out to your local folder
        for z in range(total_slices):
            target_path = os.path.join(pt["dir"], f"slice_{z:03d}.dcm")
            save_single_dicom_slice(
                pixel_array_2d=smoothed_volume[z, :, :],
                output_path=target_path,
                slice_idx=z,
                patient_sex=pt["sex"],
                target_age=pt["age"]
            )
    print("\n✅ Local cohort data folders successfully written to disk!")

if __name__ == "__main__":
    build_local_cohort()
