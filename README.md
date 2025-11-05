# RP2K Balanced Classification – Notebook Pipeline

This repository contains a single Jupyter/Colab notebook that builds a **balanced image classification** pipeline on the [RP2K dataset](https://www.kaggle.com/datasets/khyeh0719/rp2k-dataset).  

---

## Update notebook paths (if needed)

```python
DATASET_PATH = './data/rp2k-dataset'
IMG_ROOT = os.path.join(DATASET_PATH, 'rp2k_dataset')
META_PATH = os.path.join(DATASET_PATH, 'meta.csv')
```

---

## Environment

### Python
- Python ≥ 3.9

### Dependencies
From the notebook imports:

- numpy  
- pandas  
- matplotlib  
- tqdm  
- pillow  
- scikit-learn  
- opencv-python (cv2)  
- albumentations, albumentations[imgaug] (optional extras)  
- torch, torchvision, timm  

### Install (example)

```bash
pip install --upgrade pip
pip install numpy pandas matplotlib tqdm pillow scikit-learn opencv-python albumentations timm torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> Adjust PyTorch index URL to match CUDA/CPU: https://pytorch.org/get-started/locally/

---

## Reproducibility

The notebook seeds Python, NumPy, and PyTorch via:

```python
set_seed(42)
```

- `random_state=42` for dataset splits  
- Augmentation stochastic unless seeding Albumentations + NumPy/OpenCV  

---

## Workflow (Notebook Sections)

### 1. Imports, Seed, Configs, Logger
- Deterministic seeds  
- Detects `DEVICE`  
- Sets paths  

### 2. Loading & Analyzing Metadata
- Reads `meta.csv`  
- Plot: `class_distribution_all.png`  

### 3. Filter Classes (≥ 40 images)
- Keeps classes with ≥ 40 images → `meta_filtered`  
- Plot: `class_distribution_filtered.png`  

### 4. Balanced Sampling (cap 200 per class)
- Downsamples >200 images  
- Prepares augmentation list  
- Plot: `class_distribution_sampled.png`  

### 5. Augmentation Plan & Generation
- Albumentations heavy augmentations  
- Balanced dataset saved & plotted  
- Sample grid: `sample_balanced_class.png`  

### 6. Train/Validation Split
- `train_test_split(..., stratify=...)`  
- Plot: `class_distribution_trainval.png`  

### 7. Transforms
- `get_train_transform()` / `get_val_transform()`  
- Visual samples: `augmentation_examples.png`  

### 8. Dataset & Dataloaders
- `BalancedRP2KDataset`  
- `DataLoader` with `pin_memory=True`, `num_workers=4`  

### 9. Model
- EfficientNetV2 (`timm`) pretrained  

### 10. Training
- AMP (`autocast`, `GradScaler`)  
- AdamW, CosineLR  
- Checkpoints + Early stopping  
- Curves saved per epoch  

### 11. Evaluation
- Loads `best_model_balanced.pth`  
- `classification_report.csv`  
- `confusion_matrix.png`  
- `correct_incorrect_examples.png`  

---

## Hardware Notes

- GPU recommended  
- Reduce `BATCH_SIZE` if OOM  

---

## Troubleshooting

- Verify dataset paths if `cv2.imread` returns None  
- Reduce augmentation intensity if overfitting/slow  
- Use fewer workers on low-RAM systems  

---

## Dataset

Kaggle: https://www.kaggle.com/datasets/khyeh0719/rp2k-dataset

---

## References

- PyTorch Dataloader Tutorial  
- Albumentations Docs  
- sklearn train_test_split  
- Matplotlib histogram docs  

---

## License

Dataset follows Kaggle license.  
Notebook provided for research/education use.
