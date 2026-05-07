# TED_DCMIS

The official implementation of our paper
["Boosting Knowledge Diversity, Accuracy, and Stability via Tri-Enhanced Distillation for Domain Continual Medical Image Segmentation"](https://www.sciencedirect.com/science/article/abs/pii/S1361841524000379),
published in **Medical Image Analysis (MedIA), 2024**.

TED is the first work in our domain continual medical image segmentation series. It
focuses on a fundamental question:

> How can we better preserve and learn old knowledge when adapting a segmentation
> model to continuously arriving medical domains?

To address this problem, TED proposes a **Tri-Enhanced Distillation** framework that
improves old knowledge from three complementary perspectives: **diversity**,
**accuracy**, and **stability**.

## 📖 Table of Contents

- [TED\_DCMIS](#ted_dcmis)
  - [📖 Table of Contents](#-table-of-contents)
  - [Overview](#overview)
  - [What's New](#whats-new)
  - [Research Motivation](#research-motivation)
  - [Method Evolution](#method-evolution)
  - [Key Idea](#key-idea)
  - [Framework](#framework)
  - [Requirements](#requirements)
  - [Project Structure](#project-structure)
  - [Data Preparation](#data-preparation)
  - [Run](#run)
  - [Analysis](#analysis)
  - [Ablation Study](#ablation-study)
  - [Acknowledgement](#acknowledgement)
  - [Citation](#citation)

## Overview

Domain continual medical image segmentation aims to sequentially adapt a segmentation
model to multiple medical domains without revisiting previous-domain data. This setting
is clinically important because medical images are often collected from different
centers, scanners, protocols, and patient populations, while privacy and storage
constraints make joint training difficult.

However, continual adaptation suffers from **catastrophic forgetting**: when the model
learns a new domain, its performance on previously learned domains may degrade
substantially.

TED addresses this problem by enhancing the way old knowledge is distilled, transferred,
and fused during continual learning. Rather than treating knowledge distillation as a
single retention operation, TED decomposes the process into three stages and improves
each stage explicitly.

## What's New

Our follow-up works further extend this research line from new perspectives:

- **CauAug_DCMIS** studies domain continual segmentation from a causal perspective and
  mitigates knowledge bias in both new and old knowledge.
  - 📄 [Paper (IEEE JBHI)](https://ieeexplore.ieee.org/document/11054328)
  - 💻 [Code (GitHub)](https://github.com/PerceptionComputingLab/CauAug_DCMIS)

- **TKRL_DCMIS** further focuses on teacher-originated defects and rectifies knowledge
  gaps and biases inherited from older teacher models.
  - 📄 [Paper (IEEE JBHI)](https://ieeexplore.ieee.org/document/11359675)
  - 💻 [Code (GitHub)](https://github.com/PerceptionComputingLab/TKRL_DCMIS)

## Research Motivation

Existing distillation-based continual segmentation methods preserve old knowledge by
using the previous model as a teacher and the current model as a student. Although this
strategy is effective, it still has important limitations in domain continual medical
image segmentation.

Specifically, old knowledge may be poorly learned because of three issues:

1. **Limited diversity of distilled old knowledge**
   - Current-domain data provide only a narrow view of the teacher model.
   - Distilled knowledge can become repetitive and insufficient.

2. **Inaccurate transfer of old knowledge**
   - Not all teacher predictions are equally useful for the student model.
   - Irrelevant or noisy old knowledge may be transferred without selection.

3. **Unstable fusion of old and new knowledge**
   - Old and new knowledge may conflict during optimization.
   - Sample-level uncertainty can be biased and may destabilize continual training.

Therefore, TED is designed to enhance old knowledge learning by improving knowledge
**diversity**, **accuracy**, and **stability** at different stages of the distillation
process.

## Method Evolution

TED is the starting point of our research line on domain continual medical image
segmentation.

```text
TED
└── How to better retain old knowledge?
    ├── Stochastic Knowledge Augmentation: improve diversity
    ├── Adaptive Knowledge Transfer: improve accuracy
    └── Global Uncertainty-Guided Fusion: improve stability
```

This work is further extended by our subsequent studies:

```text
TED: Old knowledge retention
  ↓
CauAug: Causal learning of both new and old knowledge
  ↓
TKRL: Rectification of teacher-originated defects
```

In this series, TED focuses on **how to enhance old knowledge learning**. CauAug then
asks whether both old and new knowledge are affected by causal bias. TKRL further asks
whether the old teacher model itself already contains knowledge gaps and biases that
should be rectified before distillation.

## Key Idea

The core idea of TED is:

> Catastrophic forgetting can be mitigated more effectively if old knowledge is not only
> retained, but also enhanced in diversity, transfer accuracy, and fusion stability.

To this end, TED introduces three modules:

- **Stochastic Knowledge Augmentation (SKA)**
  - uses noise images to extract more diverse knowledge from the old model;
  - expands the coverage of old knowledge beyond the current-domain distribution.

- **Adaptive Knowledge Transfer (AKT)**
  - selectively transfers critical old knowledge;
  - emphasizes informative regions related to domain shift and reduces irrelevant
    transfer.

- **Global Uncertainty-Guided Fusion (GUGF)**
  - introduces a global view of uncertainty across batch samples;
  - stabilizes the fusion between old and new knowledge during continual learning.

## Framework

TED follows a distillation-based domain continual learning pipeline. At each continual
step, the previous model is frozen as the old teacher model, while the current model is
trained to learn the new domain and preserve old knowledge.

The overall objective consists of:

- a segmentation loss for learning the current domain;
- a standard knowledge distillation loss for retaining previous knowledge;
- an SKA loss for diverse old knowledge distillation;
- an AKT-GUGF loss for accurate and stable old-new knowledge fusion.

<p align="center">
  <img src="figures/framework.png" width="95%">
</p>

## Requirements

- Python 3.8.15
- PyTorch
- CUDA

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
--ablation_study/
--analysis/
--data_prep/
--mp/
--storage/
--README.md
--requirements.txt
--main.py
--get.py
--args.py
--command
```

## Data Preparation

Please refer to the data preparation instructions:

```bash
cat data_prep/readme.md
python data_prep/prostate_prepare.py
python data_prep/cardiacmm_prepare.py
```

## Run

Please check the example commands:

```bash
cat command
```

Example for prostate continual segmentation:

```bash
python main.py --dataset prostate --approach ted --epochs 50 \
  --experiment-name prostate-tedgugf-unet --backbone unet --device-ids 4 --gugf
```

## Analysis

```bash
python analysis/eval_dataset.py   # evaluate each dataset and each approach
python analysis/table_figure.py   # generate tables and figures in the paper
python analysis/save_images.py    # save segmentation results
```

## Ablation Study

```bash
# Ablation study of SKA
python ablation_study/ska.py
python ablation_study/ska_plot.py

# Ablation study of AKT
python ablation_study/akt.py
python ablation_study/akt_plot.py

# Ablation study of GUGF / GVU
python ablation_study/gvu_prostate_plot.py
python ablation_study/gvu_mm_plot.py
```

## Acknowledgement

Our code is inspired by
<a href="https://github.com/MECLabTUDA/ACS">ACS</a>.

## Citation

```bibtex
@article{ZHU2024103112,
  title = {Boosting knowledge diversity, accuracy, and stability via tri-enhanced distillation for domain continual medical image segmentation},
  journal = {Medical Image Analysis},
  volume = {94},
  pages = {103112},
  year = {2024},
  issn = {1361-8415},
  author = {Zhanshi Zhu and Xinghua Ma and Wei Wang and Suyu Dong and Kuanquan Wang and Lianming Wu and Gongning Luo and Guohua Wang and Shuo Li},
}
```