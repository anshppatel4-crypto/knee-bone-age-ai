import os
import torch
import torch.nn as nn
from src.model import KneeBoneAgeMultiTaskNet

def inflate_2d_checkpoint_to_3d(pretrained_2d_path, output_3d_path):
    """Maps trained 2D convolutional layers into 3D equivalents by replicating

    the filter weights along the newly introduced volumetric depth axis.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate the fresh 3D MONAI Multi-Task architecture
    model_3d = KneeBoneAgeMultiTaskNet()
    state_dict_3d = model_3d.state_dict()
    
    # 2. Load your 2D pre-trained hand checkpoint state
    if not os.path.exists(pretrained_2d_path):
        raise FileNotFoundError(f"❌ Base 2D checkpoint not found at: {pretrained_2d_path}")
        
    print(f"🔄 Loading raw 2D pre-training vectors from: {pretrained_2d_path}")
    state_dict_2d = torch.load(pretrained_2d_path, map_location=device)
    
    inflated_counter = 0
    skipped_counter = 0
    
    # 3. Iterate through every single weight block inside your network
    print("🔒 Beginning architectural tensor inflation...")
    for key in state_dict_3d.keys():
        if key in state_dict_2d:
            param_2d = state_dict_2d[key]
            param_3d = state_dict_3d[key]
            
            # Match sizes: Check if this tensor is a convolutional filter layer
            if len(param_2d.shape) == 4 and len(param_3d.shape) == 5:
                # Shape change: (Out, In, H, W) -> (Out, In, Depth, H, W)
                depth_kernel_size = param_3d.shape[2]
                
                # Replicate the 2D filter layer uniformly along the depth axis
                # We divide by depth_kernel_size to normalize signal energy levels
                inflated_param = param_2d.unsqueeze(2).repeat(1, 1, depth_kernel_size, 1, 1) / depth_kernel_size
                state_dict_3d[key] = inflated_param
                inflated_counter += 1
            else:
                # Directly copy over layers that match exactly (like Linear layers or biases)
                if param_2d.shape == param_3d.shape:
                    state_dict_3d[key] = param_2d
                    inflated_counter += 1
                else:
                    skipped_counter += 1
        else:
            skipped_counter += 1

    # 4. Save the newly inflated 3D checkpoint
    model_3d.load_state_dict(state_dict_3d)
    torch.save(model_3d.state_dict(), output_3d_path)
    print("\n" + "="*50)
    print(f"✅ INFLATION SUCCESSFUL: Loaded {inflated_counter} layers | Preserved {skipped_counter} structural blocks.")
    print(f"💾 Fresh 3D backbone saved at: {output_3d_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    inflate_2d_checkpoint_to_3d(
        pretrained_2d_path="skeletal_maturity_backbone.pth", 
        output_3d_path="inflated_knee_backbone3d.pth"
    )
