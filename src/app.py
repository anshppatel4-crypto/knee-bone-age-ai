import os
import sys
from pathlib import Path

# PATH VISIBILITY ANCHOR
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.predict import load_and_sort_dicom_volume, generate_radiographic_description
from src.model import KneeBoneAgeMultiTaskNet

# Set web page metadata configurations
st.set_page_config(
    page_title="Pediatric 3D Knee MRI Bone Age Platform",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Pediatric 3D Knee MRI Bone Age Estimation Platform")
st.markdown("""
This production dashboard runs a live **Multi-Task Volumetric Neural Network** augmented with an 
**Anisotropic Spatial Attention Gate** and a **3D Cubic Spline Resampler**. It maps physical epiphyseal 
growth plate characteristics directly onto human chronological developmental indices.
""")

st.sidebar.header("🕹️ Clinical Parameters & Controls")

scan_dir = st.sidebar.text_input(
    "Target Scan Folder Path:",
    value="data/imported_patient_scan"
)

biological_sex = st.sidebar.selectbox(
    "Patient Biological Sex:",
    options=["Male", "Female"],
    index=0
)

weights_path = st.sidebar.text_input(
    "Calibrated Model Checkpoint (.pth):",
    value="final_knee_model_fine_tuned.pth"
)

execute_inference = st.sidebar.button("🚀 Compute Diagnostic Report")

col1, col2 = st.columns(2)

if execute_inference:
    if not os.path.exists(scan_dir):
        st.error(f"❌ Target directory path not found on disk storage array: {scan_dir}")
    elif not os.path.exists(weights_path):
        st.error(f"❌ Target weights checkpoint artifact missing from system path: {weights_path}")
    else:
        with st.spinner("⏳ Standardizing 3D Matrix & Executing Attention Inference Forward Pass..."):
            try:
                volume = load_and_sort_dicom_volume(scan_dir, target_depth=64, target_resolution=256)
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = KneeBoneAgeMultiTaskNet()
                model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
                model.to(device)
                model.eval()

                volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                sex_val = 1.0 if biological_sex == "Male" else 0.0
                sex_tensor = torch.tensor([sex_val], dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    image = volume_tensor.float()
                    sex_t = sex_tensor.float().view(-1, 1)
                    
                    spatial_features = model.backbone.features(image)
                    spatial_features = model.attention_gate(spatial_features)
                    spatial_features = model.channel_bridge(spatial_features)
                    spatial_features = model._collapse_spatial_features(spatial_features)
                    
                    sex_features = model.sex_encoder(sex_t)
                    fused_features = torch.cat([spatial_features, sex_features], dim=1)
                    shared_rep = model.fusion(fused_features)
                    
                    raw_age_logit = model.regression_head(shared_rep).squeeze(-1).item()
                    stage_logits = model.stage_head(shared_rep)
                    
                    probs = torch.softmax(stage_logits, dim=1).squeeze(0).cpu().numpy()
                    predicted_stage_tier = int(np.argmax(probs))
                    
                    if predicted_stage_tier == 0 and raw_age_logit > 2.0:
                        raw_age_logit = 2.0 + (raw_age_logit - 2.0) * 0.1
                        
                    calculated_bone_age = (1.0 / (1.0 + np.exp(-raw_age_logit))) * 20.0
                    
                    closure_pct, clinical_text = generate_radiographic_description(
                        calculated_bone_age, predicted_stage_tier, probs
                    )

                st.session_state['volume'] = volume
                st.session_state['age'] = calculated_bone_age
                st.session_state['stage'] = predicted_stage_tier
                st.session_state['closure_pct'] = closure_pct
                st.session_state['clinical_text'] = clinical_text
                st.session_state['sex'] = biological_sex
                st.session_state['ran'] = True

            except Exception as e:
                st.error(f"💥 Computational runtime pipeline disruption encountered: {str(e)}")

if st.session_state.get('ran', False):
    volume = st.session_state['volume']
    
    with col1:
        st.subheader("📊 Diagnostic Metric Results")
        
        m1, m2, m3 = st.columns(3)
        m1.metric(label="🎯 Calculated Bone Age", value=f"{st.session_state['age']:.2f} Years")
        m2.metric(label="📈 Growth Plate Closure", value=f"{st.session_state['closure_pct']:.1f}%")
        m3.metric(label="🦴 Structural Stage", value=f"Tier {st.session_state['stage']}")
        
        st.info(f"🧬 **Patient Demographics:** Identified Biological {st.session_state['sex'].upper()} Profile Mapping.")
        
        st.markdown("### 📋 Generated Radiographic Impressions")
        st.write(st.session_state['clinical_text'])

    with col2:
        st.subheader("🎞️ Volumetric Slices Viewer")
        
        slice_idx = st.slider(
            "Select 3D Matrix Depth Coordinate (Z-Axis Slices):",
            min_value=0,
            max_value=63,
            value=32
        )
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(volume[slice_idx, :, :], cmap='bone')
        ax.axis('off')
        st.pyplot(fig)
        st.caption(f"Displaying cubic spline upscaled structural cross-section layer slice #{slice_idx:02d} of 64.")
