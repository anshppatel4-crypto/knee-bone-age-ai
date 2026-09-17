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
from src.predict import generate_radiographic_description, predict_scan
from src.model import load_checkpoint
from src.preprocess import DEFAULT_INPUT_SHAPE, load_series

# Set web page metadata configurations
st.set_page_config(
    page_title="Pediatric 3D Knee MRI Bone Age Platform",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Pediatric 3D Knee MRI Bone Age Estimation Platform")
st.markdown("""
A **3D ResNet-34** (MedicalNet pretrained) reads the whole knee volume and predicts skeletal
maturity alongside a growth plate closure stage, using epiphyseal ossification, physeal width
and marrow signal as cues.
""")
st.warning("Research prototype trained on synthetic phantoms. Accuracy on real patient scans is "
           "unvalidated — not for clinical use.", icon="⚠️")

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
    value="final_knee_model_resnet34.pth"
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
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model, metadata = load_checkpoint(weights_path, device)

                volume = load_series(scan_dir, tuple(metadata.get("input_shape", DEFAULT_INPUT_SHAPE)))
                result = predict_scan(model, volume, biological_sex, device)

                closure_pct, clinical_text = generate_radiographic_description(
                    result["bone_age"], result["stage"], result["stage_probabilities"]
                )

                st.session_state['volume'] = volume
                st.session_state['age'] = result["bone_age"]
                st.session_state['uncertainty'] = result["uncertainty"]
                st.session_state['stage'] = result["stage"]
                st.session_state['closure_pct'] = closure_pct
                st.session_state['clinical_text'] = clinical_text
                st.session_state['sex'] = biological_sex
                st.session_state['val_mae'] = metadata.get("val_mae")
                st.session_state['ran'] = True

            except Exception as e:
                st.error(f"💥 Computational runtime pipeline disruption encountered: {str(e)}")

if st.session_state.get('ran', False):
    volume = st.session_state['volume']
    
    with col1:
        st.subheader("📊 Diagnostic Metric Results")
        
        m1, m2, m3 = st.columns(3)
        m1.metric(label="🎯 Calculated Bone Age", value=f"{st.session_state['age']:.2f} Years",
                  delta=f"± {st.session_state.get('uncertainty', 0.0):.2f} y", delta_color="off")
        m2.metric(label="📈 Growth Plate Closure", value=f"{st.session_state['closure_pct']:.1f}%")
        m3.metric(label="🦴 Structural Stage", value=f"Tier {st.session_state['stage']}")
        
        st.info(f"🧬 **Patient Demographics:** Identified Biological {st.session_state['sex'].upper()} Profile Mapping.")
        if st.session_state.get('val_mae') is not None:
            st.caption(f"Checkpoint validation MAE: {st.session_state['val_mae']:.2f} years (synthetic data).")
        
        st.markdown("### 📋 Generated Radiographic Impressions")
        st.write(st.session_state['clinical_text'])

    with col2:
        st.subheader("🎞️ Volumetric Slices Viewer")
        
        slice_idx = st.slider(
            "Select 3D Matrix Depth Coordinate (Z-Axis Slices):",
            min_value=0,
            max_value=volume.shape[0] - 1,
            value=volume.shape[0] // 2
        )
        
        # Render with streamlit directly: no matplotlib dependency, and far faster
        slice_image = volume[slice_idx, :, :]
        lo, hi = np.percentile(slice_image, (1.0, 99.0))
        st.image(np.clip((slice_image - lo) / (hi - lo + 1e-8), 0, 1), clamp=True, use_container_width=True)
        st.caption(f"Displaying structural cross-section slice #{slice_idx:02d} of {volume.shape[0]}.")
