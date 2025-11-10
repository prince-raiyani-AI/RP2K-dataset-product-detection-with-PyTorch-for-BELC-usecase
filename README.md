# RP2K Balanced Classification – Notebook Pipeline

This repository contains a single Jupyter/Colab notebook that builds a **balanced image classification** pipeline on the [RP2K dataset](https://www.kaggle.com/datasets/khyeh0719/rp2k-dataset).  

---

## Update notebook paths as required

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
- albumentations, albumentations[imgaug]
- torch, torchvision, timm  

### Install

```bash
pip install --upgrade pip
pip install -r requirements.txt
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


## Results and visuals
<img width="1800" height="600" alt="class_image_distribution" src="https://github.com/user-attachments/assets/cd86a336-bd20-4a78-82a9-c678891d6311" />
<img width="973" height="718" alt="do1" src="https://github.com/user-attachments/assets/8c7c0883-0262-4778-a0d0-4832cc1d3059" />
<img width="1477" height="467" alt="clsd" src="https://github.com/user-attachments/assets/c75fe4c2-9d98-408f-87a8-4fb674cf56de" />
<img width="1447" height="455" alt="cld1" src="https://github.com/user-attachments/assets/b93c0a58-edcb-4bab-bdda-f5762cfe434c" />
<img width="1126" height="347" alt="te-rp2k" src="https://github.com/user-attachments/assets/ae4a7a99-6448-4e4f-86cb-ca6af26acc99" />
<img width="1200" height="500" alt="train_val_curves_epoch2" src="https://github.com/user-attachments/assets/3d47488b-1ea9-4fd6-895e-d07f04fa4304" />
<img width="1200" height="1000" alt="confusion_matrix" src="https://github.com/user-attachments/assets/e727540e-187d-467b-a2e9-fe1f8a7908c5" />
<img width="2000" height="440" alt="correct_incorrect_examples" src="https://github.com/user-attachments/assets/d6b76062-01db-4b6b-a555-2cfe102d322c" />


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
- https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader
- https://www.geeksforgeeks.org/deep-learning/reproducibility-in-pytorch/
- https://www.geeksforgeeks.org/machine-learning/how-to-set-up-and-run-cuda-operations-in-pytorch/
- https://www.geeksforgeeks.org/python/how-to-access-the-metadata-of-a-tensor-in-pytorch/
- https://www.geeksforgeeks.org/deep-learning/handling-class-imbalance-in-pytorch/
- https://seaborn.pydata.org/tutorial/distributions.html
- https://www.geeksforgeeks.org/python/apply-operations-to-groups-in-pandas/
- https://www.geeksforgeeks.org/python/python-pandas-dataframe-sample/
- https://www.w3schools.com/python/matplotlib_histograms.asp
- https://albumentations.ai/docs/
- https://www.geeksforgeeks.org/python/random-choices-method-in-python/
- https://www.geeksforgeeks.org/python/python-directory-management/
- https://www.geeksforgeeks.org/python/python-shutil-copy2-method/
- https://stackoverflow.com/questions/26681756/how-to-convert-a-python-numpy-array-to-an-rgb-image-with-opencv-2-4
- https://www.w3schools.com/python/matplotlib_subplot.asp
- https://www.geeksforgeeks.org/machine-learning/how-to-split-a-dataset-into-train-and-test-sets-using-python/
- https://www.datacamp.com/tutorial/complete-guide-data-augmentation
- https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
- https://www.geeksforgeeks.org/python/datasets-and-dataloaders-in-pytorch/
- https://dataloop.ai/library/model/timm_efficientnetv2_rw_sra2_in1k/
- https://machinelearningmastery.com/managing-a-pytorch-training-process-with-checkpoints-and-early-stopping/
- https://medium.com/biased-algorithms/a-practical-guide-to-implementing-early-stopping-in-pytorch-for-model-training-99a7cbd46e9d
- https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
  
---

## License

Dataset follows Kaggle license.  
