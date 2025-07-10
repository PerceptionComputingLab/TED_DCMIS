# TED_DCMIS
The implementation of our paper ["Boosting Knowledge Diversity, Accuracy, and Stability via Tri-Enhanced Distillation for
Domain Continual Medical Image Segmentation"](https://www.sciencedirect.com/science/article/abs/pii/S1361841524000379),  Medical Image Analysis (MedIA).

A repository for TED: a tri-enhanced distillation framework to improve knowledge diversity, accuracy, and stability in domain continual medical image segmentation.

## 📖 Table of Contents
- [Overview](#overview)
- [What’s New](#whats-new)
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Run](#run)
- [Analysis](#analysis)
- [Ablation Study](#ablation-study)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## 🔔 What’s New

We are excited to share that our new work on domain continual learning from a causal perspective has been published! It proposes a novel framework for causality-adjusted data augmentation to further mitigate knowledge bias.

👉 Check out our new work: Causality-Adjusted Data Augmentation for Domain Continual Medical Image Segmentation
- 📄 [Paper (IEEE JBHI)](https://ieeexplore.ieee.org/document/11054328)
- 💻 [Code (GitHub)](https://github.com/PerceptionComputingLab/CauAug_DCMIS)

## 🔍 Introduction

Domain continual medical image segmentation plays a crucial role in clinical settings. This approach enables segmentation models to continually learn from a sequential data stream across multiple domains. However, it faces the challenge of catastrophic forgetting. Existing methods based on knowledge distillation show potential to address this challenge via a three-stage process: distillation, transfer, and fusion. Yet, each stage presents its unique issues that, collectively, amplify the problem of catastrophic forgetting. To address these issues at each stage, we propose a tri-enhanced distillation framework. (1) Stochastic Knowledge Augmentation reduces redundancy in knowledge, thereby increasing both the diversity and volume of knowledge derived from the old network. (2) Adaptive Knowledge Transfer selectively captures critical information from the old knowledge, facilitating a more accurate knowledge transfer. (3) Global Uncertainty-Guided Fusion introduces a global uncertainty view of the dataset to fuse the old and new knowledge with reduced bias, promoting a more stable knowledge fusion. Our experimental results not only validate the feasibility of our approach, but also demonstrate its superior performance compared to state-of-the-art methods. We suggest that our innovative tri-enhanced distillation framework may establish a robust benchmark for domain continual medical image segmentation.

## 🛠️ Requirements
- Python 3.8.15
- pip install -r requirements.txt

## 🗂️ Project Structure
```
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

## 📂 Data Preparation
```
    cat data_prep/readme.md
    python data_prep/prostate_prepare.py
    python data_prep/cardiacmm_prepare.py
```

## ▶️ Run
```
    cat command
    python main.py --dataset prostate --approach ted --epochs 50 \
    --experiment-name prostate-tedgugf-unet --backbone unet --device-ids 4 --gugf
```

## 📊 Analysis
```
    python analysis/eval_dataset.py # evaluate the performance of each dataset and each approach
    python analysis/table_figure.py # generate the table and figure in the paper
    python analysis/save_images.py # save the segmentation results
```

## 🧪 Ablation Study
```
    # ablation study of the ska
    python ablation_study/ska.py
    python ablataion_study/ska_plot.py
    # ablation study of the akt
    python ablation_study/akt.py
    python ablation_study/akt_plot.py
    # ablation study of the gugf (gvu)
    python ablation_study/gvu_prostate_plot.py
    python ablation_study/gvu_mm_plot.py
    
```

## 🙏 Acknowledgement

Our code is inspired from <a href="https://github.com/MECLabTUDA/ACS
">ACS</a>.


## 📑 Citation

```bash 
@article{ZHU2024103112,
title = {Boosting knowledge diversity, accuracy, and stability via tri-enhanced distillation for domain continual medical image segmentation},
journal = {Medical Image Analysis},
volume = {94},
pages = {103112},
year = {2024},
issn = {1361-8415},
author = {Zhanshi Zhu and Xinghua Ma and Wei Wang and Suyu Dong and Kuanquan Wang and Lianming Wu and Gongning Luo and Guohua Wang and Shuo Li},
}```