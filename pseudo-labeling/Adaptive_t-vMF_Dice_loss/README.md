# Iterative Training with Adaptive t‑VMF Dice Loss

This README describes how to run **iterative pseudo‑labeling** and dataset versioning for a segmentation model trained with **Adaptive t‑VMF Dice loss**. The pipeline mirrors our VM‑UNet iterative loop but uses a t‑VMF (t‑von Mises–Fisher) + Dice composite loss that adapts per‑round to noisy pseudo‑labels.

The core idea:
1) Package each dataset **version** (original, +GAN images, +pseudo‑labels round k, …) into compact **NumPy `.npy` bundles** for fast IO.
2) Train with Adaptive t‑VMF Dice loss on version *k*.
3) Predict masks on the unlabeled pool → create version *k+1* by updating pseudo‑labels.
4) Repeat until validation performance converges.




## 1. Dataset Packaging to `.npy`

Create **NumPy archives** for each dataset version to speed up training and keep iterations reproducible.

### 1.1 Folder Layout (source images & masks)

dataset_root/
├── images/               # all images (PNG/JPG/TIF)
└── masks/                # GT masks for labeled subset (PNG, same names)

### 1.2 Build `.npy` packs
Use `data_merge.ipynb` (example below) to create memory‑mappable arrays:
- `images.npy`




## 2. End‑to‑End Workflow


# 2.1 Build v0 from original data
data_merge.ipynb

# 2.2 (Optional) Merge GAN images + initial pseudo‑labels → build v1
#    Update images/masks on disk, then:
data_merge.ipynb

# 2.3 Iterative training with Adaptive t‑VMF Dice
train.sh




## 3. Summary

- Package each dataset **version** into `.npy` arrays.
- Train with **Adaptive t‑VMF Dice** on version *k*.
- Predict masks to construct version *k+1*.
- Iterate until **convergence** by your stopping rule (Dice plateau, patience, etc.).

