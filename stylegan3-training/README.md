# Medical Imaging Datasets

This repository contains multiple medical imaging datasets prepared for **segmentation** and **classification** tasks.  
Some datasets are used in both tasks and require additional preprocessing steps.

---

## Overview of Datasets

- **CBIS-DDSM**  
  - Segmentation: Lesion segmentation masks  
  - Classification: Benign vs Malignant  
  - Requires preprocessing (DICOM to PNG, ROI extraction)

- **Kvasir-SEG**  
  - Segmentation: Polyp masks  
  - Classification: Polyp vs Non-Polyp  
  - Requires preprocessing (resize, normalization)

- **ISIC 2017**  
  - Segmentation: Skin lesion segmentation

- **ISIC 2018**  
  - Segmentation: Skin lesion segmentation

- **Chest X-Ray (NIH / Kaggle)**  
  - Classification: Pneumonia detection

- **BreastMNIST (MedMNIST v2)**  
  - Classification: Benign vs Malignant breast tumors

---

## Task-Specific Grouping

### Segmentation Datasets
- CBIS-DDSM
- Kvasir-SEG
- ISIC 2017
- ISIC 2018

### Classification Datasets
- CBIS-DDSM
- Kvasir-SEG
- Chest X-Ray
- BreastMNIST

---

## Preprocessing Notes

- **CBIS-DDSM**  
  Convert raw DICOM images to PNG/JPG. Extract ROI patches for both segmentation and classification tasks.  
  Crop converted image and have the tumor region centered for easier analysis.
  Normalization recommended before training.

- **Kvasir-SEG**  
  Cut all images and masks to multiple fixed size patches (e.g., 256×256).  
  Use binary masks for segmentation. For classification, convert to polyp vs non-polyp labels.

- **ISIC 2017 & 2018**  
  Images are already in JPG/PNG format with corresponding masks.  
  Standard resizing and normalization sufficient.

- **Chest X-Ray**  
  Use pneumonia vs normal labels. Preprocessing includes resizing and grayscale normalization.

- **BreastMNIST**  
  Provided in NumPy / PNG format through MedMNIST.  
  Split into train/val/test as provided by the official dataset.

---

## Folder Structure (Example)

```text
datasets/
├── CBIS-DDSM/
│   ├── images/
│   ├── masks/                 # segmentation masks
│   ├── classification/        # classification-ready splits
│   └── preprocessing/         # scripts for dicom2png, ROI extraction etc.
├── Kvasir-SEG/
│   ├── images/
│   ├── masks/
│   ├── classification/
│   └── preprocessing/
├── ISIC2017/
│   ├── images/
│   └── masks/
├── ISIC2018/
│   ├── images/
│   └── masks/
├── ChestX-Ray/
│   ├── train/
│   ├── val/
│   └── test/
└── BreastMNIST/
    ├── train/
    ├── val/
    └── test/



---

## Usage

- To prepare segmentation datasets:  
Run preprocessing scripts inside each dataset folder to generate `images/` and `masks/` pairs.

- To prepare classification datasets:  
Use provided preprocessing to split data into `train/`, `val/`, and `test/` directories with class subfolders.

---

## Citation

If you use these datasets, please cite their original sources:

- [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)  
- [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)  
- [ISIC 2017](https://challenge.isic-archive.com/landing/2017/)  
- [ISIC 2018](https://challenge.isic-archive.com/landing/2018/)  
- [Chest X-Ray](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- [BreastMNIST](https://medmnist.com/)

---
