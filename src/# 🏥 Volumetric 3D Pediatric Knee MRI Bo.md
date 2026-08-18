# 🏥 Volumetric 3D Pediatric Knee MRI Bone Age Estimation SaaS Infrastructure

An advanced, end-to-end multi-task deep learning architecture powered by **PyTorch** and **MONAI** designed to automate continuous continuous pediatric bone age calculation and growth plate closure tier classification from 3D Knee MRI scans. 

Targeting an elite clinical diagnostic accuracy threshold of **0.5–0.7 years Mean Absolute Error (MAE)**, this system operates entirely via privacy-compliant data pathways, bypassing institutional data blocks and HIPAA constraints.

---

## 🧬 Core Engineering Innovations

To resolve the severe shortage of labeled pediatric knee datasets without compromising anatomical truth, this project introduces a three-tiered algorithmic framework:

1. **2.5D Volumetric Latent Diffusion Engine (`src/generate_25d.py`)**  
   Leverages text-conditioned latent diffusion filters to generate micro-sequential cross-sectional image layers of the knee joint. It applies a **1D Gaussian Depth smoothing filter** across the Z-axis stack to enforce structural bone continuity, fusing independent 2D frames into a valid 3D spatial matrix. It then uses `pydicom` to inject realistic scanner properties (`SliceLocation`, `PatientSex`).

2. **2D-to-3D Convolutional Weight Inflation Matrix (`src/inflate_weights.py`)**  
   Prevents network initialization collapse. It extracts high-utility structural filters learned from **12,000 real-world public RSNA Hand/Wrist radiographs**, duplicates them layer-by-layer along the volumetric depth axis, and normalizes signal energy to inherit complex pediatric skeletal maturity physics before knee fine-tuning.

3. **Multi-Task Volumetric Neural Network (`src/model.py`)**  
   Utilizes a high-capacity **3D DenseNet121** backbone to extract structural features, embeds and concatenates the patient's categorical biological sex, and bifurcates into dual terminal heads: an **Age Regression Head** (Continuous Decimal Years via MSE Loss with a Sigmoid bounding activation layer mapped to 20.0 years) and a **Stage Classification Head** (Growth Plate Closure Tiers via Cross-Entropy Loss).

4. **Trilinear Volumetric Standardization Guard (`src/predict.py`)**  
   A deployable production inference CLI that dynamically standardizes arbitrary clinical scan dimensions to a uniform grid (64 slices, 256x256 resolution) via trilinear spline interpolation to prevent 3D neural feature collapse in deep downsampling blocks.

---

## 🚀 Execution & Command-Line Operations

### 1. Install Workspace Requirements
```bash
pip install torch diffusers transformers accelerate scipy pydicom numpy monai
```

### 2. Synthesize High-Density 3D Pediatric Evaluation Slices
```bash
python src/run_local_generation.py
```

### 3. Inflate Pre-Trained 2D Radiograph Weight Filters
```bash
python src/inflate_weights.py
```

### 4. Optimize Model Parameters Across Multi-Patient Cohorts
```bash
python src/train_knee.py
```

### 5. Execute the SaaS Production Inference Engine
```bash
python src/predict.py --dir data/cohort_pt_2/ --sex m --weights final_knee_model_fine_tuned.pth
```
