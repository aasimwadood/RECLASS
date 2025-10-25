# RECLASS: Redefining Classroom Behavior through Multi-Modal Unsupervised Student and Scene Tracking

**RECLASS** is a unified **multi-modal, unsupervised anomaly detection framework** for classroom scene understanding.  
It jointly models **spatio-temporal (RGB + optical flow)** dynamics and **social graph relationships** (students, interactions) to identify abnormal events such as distractions, inactivity, or unsafe actions — without requiring any labeled anomalies.

This repository provides a PyTorch reference implementation of the core RECLASS framework described in our paper  
> *“RECLASS: Redefining Classroom Behavior through Multi-Modal Unsupervised Student and Scene Tracking”* (2025).

---

## 🚀 Key Features

- **Multi-Modal Encoders:** Joint learning of visual (RGB) and motion (optical flow) streams using shared latent spaces.  
- **Dual Generators (G<sub>st</sub>, G<sub>g</sub>):**  
  - *Spatio-Temporal Generator* synthesizes future video clips from latent embeddings.  
  - *Graph Generator* reconstructs social interaction graphs from student embeddings.  
- **Adversarial Discriminators (D<sub>st</sub>, D<sub>g</sub>):** Distinguish real vs. generated visual/graph samples.  
- **Explainability via Grad-CAM:** Visual heatmaps indicate regions influencing anomaly decisions.  
- **Unsupervised Anomaly Scoring:** Combines reconstruction, adversarial, and graph-structure losses with EMA smoothing.  
- **Configurable & Modular:** Drop-in replacement for your own detectors, flow estimators, or tracking pipelines.

---

## 🧩 Repository Structure

```
RECLASS/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── reclass/
│   ├── __init__.py
│   ├── dataset.py
│   ├── models.py
│   ├── losses.py
│   ├── training.py
│   ├── inference.py
│   └── utils.py
├── scripts/
│   └── train.sh
└── examples/
    └── run_demo.ipynb
```

---

## ⚙️ Installation

Tested on **Python 3.10+**, **PyTorch ≥2.0**, **CUDA ≥11.7**.

```bash
git clone https://github.com/<yourname>/RECLASS.git
cd RECLASS
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 📦 Requirements

```
torch>=2.0
torchvision
numpy
opencv-python
pyyaml
tqdm
networkx
scikit-learn
matplotlib
tensorboard
```

---

## 🧰 Pre-processing Pipeline (External Tools)

To reproduce the full RECLASS data flow, integrate the following external modules:

| Component | Tool | Output | Paper Ref. |
|------------|------|---------|-------------|
| **Object Detection** | [YOLOv8](https://github.com/ultralytics/ultralytics) | Bounding boxes per frame | Sec. 3.1 |
| **Multi-Object Tracking** | [ByteTrack](https://github.com/ifzhang/ByteTrack) | Track IDs across frames | Sec. 3.1 |
| **Optical Flow** | [RAFT](https://github.com/princeton-vl/RAFT) or TV-L1 | Flow magnitude & direction | Eq. (3) |
| **Pose Extraction** | [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) | Skeleton keypoints | Sec. 3.2 |
| **Graph Construction** | Custom adjacency (distance + co-attention) | Graph G = (V,E) | Eq. (5)–(6) |

After preprocessing, each video clip should provide:
- RGB clip tensor: `[T, 3, H, W]`
- Flow tensor: `[T, 2, H, W]`
- Graph node features: `[N, F]`
- Adjacency matrix: `[N, N]`

Store paths to these in text files (`train.txt`, `val.txt`, `test.txt`) as in `configs/default.yaml`.

---

## 🧠 Model Overview

| Component | Symbol | Description |
|------------|---------|-------------|
| **Encoders** | E<sub>st</sub>, E<sub>flow</sub>, E<sub>g</sub> | Extract latent representations from video, flow, and graphs |
| **Generators** | G<sub>st</sub>, G<sub>g</sub> | Reconstruct or predict next frames / social graphs |
| **Discriminators** | D<sub>st</sub>, D<sub>g</sub> | Adversarial networks to enforce realism |
| **Explainability** | Grad-CAM | Highlights salient spatial-temporal regions influencing scores |
| **Anomaly Scoring** | A(t) | Combines reconstruction + adversarial + graph cues |

---

## 🧮 Training

```bash
bash scripts/train.sh
```

Example config: `configs/default.yaml`
```yaml
train:
  epochs: 60
  batch_size: 8
  lr: 3e-4
  betas: [0.5, 0.999]
  clip_len: 32
  image_h: 256
  image_w: 448
  ema_beta: 0.9
  fusion_alpha: 0.85
dataset:
  root: /data/SAVD/
  train_list: train.txt
  val_list: val.txt
  test_list: test.txt
```

### Monitoring
Use TensorBoard for visual loss curves:
```bash
tensorboard --logdir runs
```

---

## 🔍 Evaluation

During inference, each clip produces an anomaly score `A(t)`.  
A Gaussian Mixture Model (GMM) or percentile threshold on validation scores can be used to determine abnormal frames.

```bash
python -m reclass.inference --config configs/default.yaml --weights weights/best.pth
```

Metrics:
- **Frame-level AUC (FAUC)**
- **Pixel-level AUC (PAUC)**
- **Event-level F1**

---

## 🌈 Explainability

RECLASS integrates **Grad-CAM** on the video discriminator `D_st`.  
This provides saliency heatmaps highlighting the regions that contributed most to anomaly predictions.

```python
from reclass.utils import grad_cam_on_discriminator
cam = grad_cam_on_discriminator(Dst, input_clip, score)
```

Generated CAM maps can be overlaid on frames for visual inspection.

---

## 🧪 Example Usage

`examples/run_demo.ipynb` demonstrates:
```python
from reclass.inference import infer_step
from reclass.models import *

# Load pretrained weights
model_dict = {...}
outputs = infer_step(model_dict, batch, device='cuda')
A_t, A_ema, cam = outputs['A'], outputs['A_ema'], outputs['cam']
```


---

## 🧬 Dataset (SAVD)

The *Student Activity Video Dataset (SAVD)* consists of multiple classroom videos recorded under varied lighting and viewpoints.  
Each clip contains synchronized RGB, flow, and pose features.  
For research use, please contact the authors for dataset access.

Default preprocessing parameters (from paper Table 2):

| Parameter | Value |
|------------|--------|
| Frame size | 256×448 |
| Clip length | 32 frames |
| Flow window | 10 |
| Flow method | TV-L1 (λ=0.15, θ=0.3) |
| Sampling stride | 2 |
| FPS | 25 |

---

## 📊 Results (from paper)

| Dataset | FAUC | PAUC | F1 | AUPRC |
|----------|------|------|----|-------|
| SAVD | 0.923 | 0.902 | 0.881 | 0.874 |

---

## 🧱 Design Philosophy

RECLASS is built around **interpretable unsupervised learning** — it doesn’t just detect anomalies, it explains *why* an event is anomalous:
1. Reconstruct normal spatio-temporal and social dynamics.  
2. Highlight deviations in reconstruction, motion, and attention.  
3. Fuse multimodal evidence for robust decision making.

---

## 🛠 Troubleshooting

| Issue | Likely Cause | Fix |
|-------|---------------|-----|
| Losses diverge early | Discriminators too strong | Reduce `λ_adv` or pretrain encoders |
| CAM maps are blank | Gradients detached | Ensure `requires_grad=True` for discriminator inputs |
| Memory overflow | High clip length | Reduce `clip_len` to 16 or enable gradient checkpointing |

---

## 📜 Citation

If you use this code or dataset, please cite:

```
@article{reclass2025,
  title={RECLASS: Redefining Classroom Behavior through Multi-Modal Unsupervised Student and Scene Tracking},
  author={Asim Wadood et al.},
  journal={arXiv preprint arXiv:2501.xxxxx},
  year={2025}
}
```


---

## 📄 License

MIT License © 2025 Your Name / Research Lab  
Free for academic and non-commercial use. Contact authors for commercial licensing.

---

## 🌍 Acknowledgements
This work builds on open research from the visual anomaly detection community, including:
- *AnoGAN*, *MemAE*, *STCNet*, *GraphDeVI*, *GMAD*.
We thank the contributors of YOLOv8, RAFT, and OpenPose for making their tools available.

---

> **Note:** This repository provides a clean, research-ready codebase for academic and prototyping use.  
> For deployment or dataset access, please refer to institutional data-use policies.
