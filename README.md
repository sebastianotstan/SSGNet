# SSGNet

This repository provides the official implementation of the paper:  

**"Towards Data-Efficient Medical Imaging: A Generative and Semi-Supervised Framework" (BMVC 2025)**  

[ArXiv Preprint (https://arxiv.org/abs/2508.04534)]() • [Conference Paper (BMVC Proceedings)]()

<p align="center">
  <img src="images/diagram_BMVC.pdf" width="800px">
</p>

---

## Abstract
Deep learning in medical imaging is often limited by scarce and imbalanced annotated data. We present **SSGNet**, a unified framework that combines class–specific generative modeling with iterative semi–supervised pseudo–labeling to enhance both classification and segmentation. Rather than functioning as a standalone model, SSGNet augments existing baselines by expanding training data with StyleGAN3–generated images and refining labels through iterative pseudo–labeling. Experiments across multiple medical imaging benchmarks demonstrate consistent gains in classification and segmentation performance, while Fréchet Inception Distance analysis confirms the high quality of generated samples. These results highlight SSGNet as a practical strategy to mitigate annotation bottlenecks and improve robustness in medical image analysis.

---

## Repository Structure (planned)
```bash
SSGNet/
│
├── classification/       # Classification experiments (ResNet50, EfficientNet, etc.)
├── segmentation/         # Segmentation experiments (VM-UNet, Adaptive t-vMF Dice Loss)
├── stylegan3_training/   # Scripts for StyleGAN3 class-specific training
├── pseudo_labeling/      # Iterative pseudo-labeling and refinement
├── results/              # Sample outputs, logs, and comparison grids
├── data/                 # Instructions for dataset preparation (CBIS-DDSM, ISIC, Kvasir, etc.)
│
├── requirements.txt      # Dependencies
├── README.md             # This file
└── LICENSE
