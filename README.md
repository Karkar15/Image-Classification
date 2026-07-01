# COMP30027 Assignment 2

This repository contains the Task 1 and Task 2 image-classification experiments:

- Classical models: Random Forest, kNN, SVM
- CNN models: pre-trained ResNet-18 and scratch ResNet-18
- Task 2 ensemble: Random Forest + SVM + CNN voting ensemble

All commands below should be run from the project root

## 1. Environment Setup

Create or activate a Python environment, then install the required packages:

```bash
python -m pip install numpy pandas matplotlib scikit-learn torch torchvision opencv-python mahotas scikit-image pillow
```

On Apple Silicon, the CNN scripts automatically use `mps` when available. Otherwise they fall back to CUDA or CPU.

## 2. Expected Data Layout

The scripts expect these folders to exist:

```text
task1_data/
task2_data/
```

Each task folder should contain the assignment metadata and feature files, including:

```text
train_metadata.csv
test_metadata.csv
color_histogram.csv
hog_pca.csv
additional_features.csv
images/train/
images/test/
```

Task 2 may also include `class_mapping.csv`.

## 3. Optional Feature Precomputation

Random Forest, SVM, kNN, and the ensemble use cached texture/keypoint features. The model scripts can generate these automatically, but precomputing them first makes later runs cleaner:

```bash
python precompute_features.py all
```

To precompute only one task:

```bash
python precompute_features.py task1
python precompute_features.py task2
```

## 4. Run Task 1 Models

### Task 1 Random Forest

```bash
python task1_random_forest.py
```

Outputs include validation metrics, feature-importance plots, and:

```text
plots/task1_rf_submission.csv
```

### Task 1 kNN

```bash
python task1_knn.py
```

Outputs include validation reports, tuning plots, predictions, and:

```text
plots/task1_knn_submission.csv
```

### Task 1 SVM

```bash
python task1_svm.py
```

Outputs include validation reports, tuning plots, predictions, and:

```text
plots/task1_svm_submission.csv
```

### Task 1 Pre-trained CNN

```bash
python task1_pretrained_cnn.py
```

This runs learning-rate tuning, final 80/20 training, validation evaluation, t-SNE, and test prediction.

Outputs include:

```text
plots/task1_resnet18_best.pth
plots/task1_resnet18_submission.csv
plots/task1_resnet18_submission_classid.csv
```

### Task 1 Scratch CNN

```bash
python task1_scratch_cnn.py
```

This trains ResNet-18 from random weights. It is much slower than the pre-trained CNN.

Outputs include:

```text
plots/task1_scratchcnn_best.pth
plots/task1_scratchcnn_submission.csv
plots/task1_scratchcnn_submission_classid.csv
```

## 5. Run Task 2 Models

### Task 2 Random Forest

```bash
python task2_random_forest.py
```

Outputs include validation metrics, feature-importance plots, and:

```text
plots/task2_rf_submission.csv
```

### Task 2 SVM

```bash
python task2_svm.py
```

Outputs include validation reports, tuning plots, predictions, and:

```text
plots/task2_svm_submission.csv
```

### Task 2 Pre-trained CNN

```bash
python task2_pretrained_cnn.py
```

This runs learning-rate tuning, final 80/20 training, validation evaluation, t-SNE, and test prediction.

Outputs include:

```text
plots/task2_resnet18_best.pth
plots/task2_resnet18_submission.csv
plots/task2_resnet18_submission_classid.csv
```

### Task 2 Ensemble

```bash
python ensemble.py
```

This trains and compares a Random Forest, SVM, and CNN ensemble using hard and soft voting.

Outputs include:

```text
plots/task2_ensemble_submission.csv
plots/task2_ensemble_predictions.csv
plots/task2_ensemble_summary.txt
```

## 6. Generate Summary Plots

After running the models, generate combined comparison plots with:

```bash
python plot.py
```

Outputs are written to:

```text
plots_allmodels/
```

## 7. Recommended Run Order

For a complete run:

```bash
python precompute_features.py all

python task1_random_forest.py
python task1_knn.py
python task1_svm.py
python task1_pretrained_cnn.py
python task1_scratch_cnn.py

python task2_random_forest.py
python task2_svm.py
python task2_pretrained_cnn.py
python ensemble.py

python plot.py
```

## 8. Notes

- All main outputs are saved under `plots/`.
- Combined model-comparison plots are saved under `plots_allmodels/`.
- Classical models retrain their final estimator on 100% of the labelled training data before Kaggle prediction.
- Standalone CNN submissions use the best validation checkpoint from the final 80/20 training split.
- CNN scripts can take a long time, especially `task1_scratch_cnn.py` and `ensemble.py`.
