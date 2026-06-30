# TrustTour

**An AI-enabled Smart Tourism Trust and Decision Support Framework for Authenticity Verification of Tourism Visual Content**

TrustTour combines a **Dual-Stream Scaled Vision Transformer (DS-SViT)** for AI-generated image detection with a **trust estimation**, **risk quantification**, **human-in-the-loop reinforcement learning (HITL-RL)**, and **Stackelberg game-theoretic policy optimization** layer, producing an end-to-end, interpretable decision-support system rather than a standalone binary classifier.

This repository provides a full, runnable reference implementation:

- DS-SViT model (RGB semantic stream + SRM forensic residual stream + cross-attention fusion)
- Five baselines: Xception (XCP), F3Net (F3N), Swin Transformer (Swin), AIDE (CLIP-based), IAPL (prompt-learning VFM)
- Trust Assessment Module (Eq. 39–45), Adaptive Decision Module (risk thresholds, Eq. 46–51), Strategic Optimization Module (Stackelberg game, Eq. 52–59)
- HITL-RL policy-gradient threshold tuner
- Dataloaders for **CIFAKE** and **Synthbuster**
- Training / evaluation / inference scripts, metrics (ACC, PRE, REC, F1, MCC, AUC, HRR), and plotting utilities matching the paper's figures

> This code is a faithful open re-implementation of the architecture and equations described in the TrustTour paper. Exact reported numbers (96.8% accuracy, etc.) depend on the actual dataset splits, seeds, and compute used in the paper and are not guaranteed to reproduce bit-for-bit — use this as a research-grade starting point.

## Repository Structure

```
trusttour/
├── configs/                 # YAML experiment configs
│   ├── default.yaml
│   ├── baselines.yaml
│   └── trusttour.yaml
├── data/
│   ├── datasets.py          # CIFAKE / Synthbuster Dataset classes
│   ├── transforms.py        # augmentation / SRM-aware transforms
│   └── prepare_data.py      # folder-structure builder / sanity checks
├── models/
│   ├── ds_svit.py           # Dual-Stream Scaled Vision Transformer
│   ├── srm.py                # Spatial Rich Model filters
│   ├── trust_module.py       # Trust Assessment Module (Eq. 39-45)
│   ├── decision_module.py    # Adaptive Decision Module (Eq. 46-51)
│   └── baselines/
│       ├── xception.py
│       ├── f3net.py
│       ├── swin.py
│       ├── aide.py
│       └── iapl.py
├── rl/
│   └── hitl_rl.py            # HITL-RL threshold policy agent
├── game/
│   └── stackelberg.py        # Strategic Optimization Module (Eq. 52-59)
├── utils/
│   ├── metrics.py            # ACC/PRE/REC/F1/MCC/AUC/HRR
│   ├── losses.py
│   ├── seed.py
│   ├── logger.py
│   └── plotting.py           # reproduces Fig. 3-6 style plots
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── train_baselines.py
│   ├── run_hitl_rl.py
│   ├── run_game.py
│   └── infer.py
├── tests/
│   ├── test_models.py
│   ├── test_trust_decision.py
│   └── test_dataloader.py
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-org>/trusttour.git
cd trusttour
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Preparation

Download **CIFAKE** (Kaggle) and **Synthbuster** (Hugging Face / official release) and arrange them as:

```
data/CIFAKE/{train,test}/{REAL,FAKE}/*.png
data/Synthbuster/{train,test}/{real,fake}/*.png
```

Then run:

```bash
python data/prepare_data.py --root data/CIFAKE --dataset cifake --check
python data/prepare_data.py --root data/Synthbuster --dataset synthbuster --check
```

## Training

Train TrustTour (DS-SViT + trust fusion):

```bash
python scripts/train.py --config configs/trusttour.yaml --dataset cifake
```

Train all baselines:

```bash
python scripts/train_baselines.py --config configs/baselines.yaml --dataset cifake --model xception
python scripts/train_baselines.py --config configs/baselines.yaml --dataset cifake --model f3net
python scripts/train_baselines.py --config configs/baselines.yaml --dataset cifake --model swin
python scripts/train_baselines.py --config configs/baselines.yaml --dataset cifake --model aide
python scripts/train_baselines.py --config configs/baselines.yaml --dataset cifake --model iapl
```

## Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/trusttour_best.pth --dataset cifake --split test
```

Reports ACC / PRE / REC / F1 / MCC / AUC plus trust-aware stats (mean trust, mean risk, HRR, decision distribution) and saves ROC/PR curves to `outputs/`.

## HITL-RL Threshold Adaptation

```bash
python scripts/run_hitl_rl.py --checkpoint checkpoints/trusttour_best.pth --dataset cifake --episodes 200
```

## Strategic (Stackelberg) Optimization

```bash
python scripts/run_game.py --checkpoint checkpoints/trusttour_best.pth --iterations 60
```

## Inference on a Single Image

```bash
python scripts/infer.py --checkpoint checkpoints/trusttour_best.pth --image path/to/photo.jpg
```

Outputs authenticity score `A(I)`, trust score `T(I)`, risk score `R(I)`, and the platform decision (`Verified` / `Human Review` / `Rejected`).

## Citing

If you use this code, please cite the TrustTour paper:

```bibtex
@article{trusttour2026,
  title   = {TrustTour: An AI-enabled Smart Tourism Trust and Decision Support Framework for Authenticity Verification of Tourism Visual Content},
  author  = {Mishra, Pankaj and co-authors},
  year    = {2026}
}
```

## License

MIT License, see [LICENSE](LICENSE).
