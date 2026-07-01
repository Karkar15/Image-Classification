from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from data_preprocessing import (
    MetadataImageDataset,
    RandomForestFeatureConfig,
    load_full_feature_frames,
    task2_pretrained_transforms,
)


DATA_DIR = Path("task2_data")
OUTPUT_DIR = Path("plots")
RANDOM_STATE = 42
NUM_CLASSES = 10

RF_PARAMS = {
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "n_estimators": 300,
}

SVM_PARAMS = {
    "C": 3.0,
    "gamma": 0.003,
}

CNN_LR = 1e-4
BATCH_SIZE = 16
CV_FOLDS = 5
CNN_CV_EPOCHS = 15
CNN_FINAL_EPOCHS = 50
WEIGHT_DECAY = 1e-4
CV_PATIENCE = 7
VAL_SIZE = 0.2

FEATURE_CONFIG = RandomForestFeatureConfig(
    data_dir=DATA_DIR,
    image_size=(128, 128),
    texture_output_csv="task2_texture_features.csv",
    keypoint_output_csv="task2_keypoint_features.csv",
    test_texture_output_csv="task2_test_texture_features.csv",
    test_keypoint_output_csv="task2_test_keypoint_features.csv",
    include_fine_color=True,
    include_hog_contrast=True,
    gabor_frequencies=(0.1, 0.25, 0.4, 0.6),
    gabor_thetas=(0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6),
)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_rf() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    **RF_PARAMS,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_svm() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    **SVM_PARAMS,
                ),
            ),
        ]
    )


def build_cnn(device: torch.device) -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        if "layer1" in name:
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate_cnn(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)

            running_loss += loss.item() * images.size(0)
            correct += (probs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, np.vstack(all_probs), np.array(all_labels)


def train_cnn_fold(train_meta, val_meta, device, train_transforms, val_transforms):
    model = build_cnn(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CNN_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CNN_CV_EPOCHS, eta_min=1e-6
    )

    train_loader = DataLoader(
        MetadataImageDataset(train_meta, DATA_DIR, train_transforms),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        MetadataImageDataset(val_meta, DATA_DIR, val_transforms),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    best_loss = float("inf")
    best_weights = None
    patience = 0

    for epoch in range(1, CNN_CV_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate_cnn(model, val_loader, criterion, device)
        scheduler.step()
        print(
            f"    CNN epoch {epoch:02d}/{CNN_CV_EPOCHS} | "
            f"train acc {train_acc:.4f} | val acc {val_acc:.4f} | val loss {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= CV_PATIENCE:
                print(f"    CNN early stopping at epoch {epoch}")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model


def train_cnn_full(train_meta, device, train_transforms):
    model = build_cnn(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CNN_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CNN_FINAL_EPOCHS, eta_min=1e-6
    )
    train_loader = DataLoader(
        MetadataImageDataset(train_meta, DATA_DIR, train_transforms),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    for epoch in range(1, CNN_FINAL_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        print(
            f"    CNN full epoch {epoch:02d}/{CNN_FINAL_EPOCHS} | "
            f"train acc {train_acc:.4f} | train loss {train_loss:.4f}"
        )

    torch.save(model.state_dict(), OUTPUT_DIR / "task2_ensemble_cnn_full.pth")
    return model


def predict_cnn_proba(model, meta, device, val_transforms):
    has_labels = "class_id" in meta.columns
    loader = DataLoader(
        MetadataImageDataset(meta, DATA_DIR, val_transforms, has_labels=has_labels),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    all_probs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)


def hard_vote(predictions: list[np.ndarray], tie_break_probs: np.ndarray, classes: np.ndarray) -> np.ndarray:
    stacked = np.vstack(predictions)
    out = []
    for col_idx in range(stacked.shape[1]):
        counts = np.bincount(stacked[:, col_idx].astype(int), minlength=len(classes))
        winners = np.where(counts == counts.max())[0]
        if len(winners) == 1:
            out.append(winners[0])
        else:
            best = winners[np.argmax(tie_break_probs[col_idx, winners])]
            out.append(best)
    return np.array(out, dtype=int)


def class_id_mapping(train_frame: pd.DataFrame) -> dict[int, str]:
    return (
        train_frame[["class_id", "class_name"]]
        .drop_duplicates()
        .sort_values("class_id")
        .set_index("class_id")["class_name"]
        .to_dict()
    )


def save_classification_outputs(prefix, y_true, y_pred, class_id_to_name):
    import matplotlib.pyplot as plt

    labels = sorted(class_id_to_name)
    class_names = [class_id_to_name[label] for label in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_path = OUTPUT_DIR / f"{prefix}_classification_report.csv"
    pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "class"}).to_csv(
        report_path, index=False
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_path = OUTPUT_DIR / f"{prefix}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues"
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.set_title("Task 2 Ensemble Confusion Matrix (Held-out Validation)")
    plt.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    return report_path, cm_path


def save_model_comparison(prefix, validation_scores, best_voting):
    import matplotlib.pyplot as plt

    rows = []
    for name in ["rf", "svm", "cnn", best_voting]:
        label = f"ensemble_{name}" if name in ("hard", "soft") else name
        rows.append(
            {
                "model": label,
                "validation_accuracy": float(validation_scores[name]),
            }
        )
    table = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / f"{prefix}_model_comparison.csv"
    table.to_csv(csv_path, index=False)

    png_path = OUTPUT_DIR / f"{prefix}_model_comparison.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        table["model"],
        table["validation_accuracy"],
        color=["#4C72B0", "#DD8452", "#55A868", "#8172B2"],
    )
    for bar, value in zip(bars, table["validation_accuracy"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Held-out Validation Accuracy")
    ax.set_title("Task 2: Individual Models vs Ensemble")
    ax.set_ylim(0, min(1, float(table["validation_accuracy"].max()) + 0.08))
    plt.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return csv_path, png_path


def save_soft_vote_heatmap(prefix, soft_probs, model_probs, y_true, train_frame, class_id_to_name):
    import matplotlib.pyplot as plt

    classes = np.array(sorted(class_id_to_name))
    soft_pred = classes[np.argmax(soft_probs, axis=1)]
    misclassified = np.where(soft_pred != y_true)[0]
    sample_indices = misclassified[:5] if len(misclassified) else np.argsort(soft_probs.max(axis=1))[:5]
    model_names = ["rf", "svm", "cnn", "ensemble"]
    class_names = [class_id_to_name[int(class_id)] for class_id in classes]

    rows = []
    for sample_idx in sample_indices:
        image_id = train_frame.iloc[sample_idx]["image_id"]
        for model_name in model_names:
            probs = soft_probs[sample_idx] if model_name == "ensemble" else model_probs[model_name][sample_idx]
            for class_id, class_name, prob in zip(classes, class_names, probs):
                rows.append(
                    {
                        "image_id": image_id,
                        "true_class_id": int(y_true[sample_idx]),
                        "true_class_name": class_id_to_name[int(y_true[sample_idx])],
                        "soft_ensemble_pred_class_id": int(soft_pred[sample_idx]),
                        "soft_ensemble_pred_class_name": class_id_to_name[int(soft_pred[sample_idx])],
                        "model": model_name,
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "probability": prob,
                    }
                )

    csv_path = OUTPUT_DIR / f"{prefix}_soft_vote_probability_heatmap.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    png_path = OUTPUT_DIR / f"{prefix}_soft_vote_probability_heatmap.png"

    fig, axes = plt.subplots(
        len(sample_indices),
        1,
        figsize=(max(13, 1.15 * len(class_names)), max(7, 2.15 * len(sample_indices) + 1.5)),
        squeeze=False,
    )
    image = None
    for row_number, (ax, sample_idx) in enumerate(zip(axes[:, 0], sample_indices)):
        matrix = np.vstack([
            model_probs["rf"][sample_idx],
            model_probs["svm"][sample_idx],
            model_probs["cnn"][sample_idx],
            soft_probs[sample_idx],
        ])
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_yticklabels(model_names)
        ax.set_xticks(np.arange(len(class_names)))
        if row_number == len(sample_indices) - 1:
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("Class")
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        image_id = train_frame.iloc[sample_idx]["image_id"]
        true_name = class_id_to_name[int(y_true[sample_idx])]
        pred_name = class_id_to_name[int(soft_pred[sample_idx])]
        ax.set_title(f"{image_id}: true={true_name}, soft ensemble={pred_name}", fontsize=10)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if value >= 0.15:
                    ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white")
    if image is not None:
        cbar_ax = fig.add_axes([0.88, 0.14, 0.025, 0.72])
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.set_label("Predicted probability", labelpad=12)
    fig.suptitle("Task 2 Soft Vote Probability Distribution on Misclassified Samples", y=0.985)
    fig.subplots_adjust(left=0.11, right=0.84, top=0.93, bottom=0.16, hspace=0.85)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return csv_path, png_path


def main():
    parser = argparse.ArgumentParser(description="Task 2 RF + SVM + CNN voting ensemble.")
    parser.parse_args()

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    OUTPUT_DIR.mkdir(exist_ok=True)
    prefix = "task2_ensemble"

    print("Task 2 Ensemble: RF + SVM + CNN")
    print(f"RF params : {RF_PARAMS}")
    print(f"SVM params: {SVM_PARAMS}")
    print(f"CNN LR    : {CNN_LR}")
    print(f"CV folds  : {CV_FOLDS}")

    device = get_device()
    print(f"Device    : {device}")

    train_transforms, val_transforms = task2_pretrained_transforms()
    train_frame, test_frame, feature_cols, _ = load_full_feature_frames(FEATURE_CONFIG)
    train_paths = pd.read_csv(DATA_DIR / "train_metadata.csv")[["image_id", "image_path"]]
    test_paths = pd.read_csv(DATA_DIR / "test_metadata.csv")[["image_id", "image_path"]]
    if "image_path" not in train_frame.columns:
        train_frame = train_frame.merge(train_paths, on="image_id", how="left", validate="one_to_one")
    if "image_path" not in test_frame.columns:
        test_frame = test_frame.merge(test_paths, on="image_id", how="left", validate="one_to_one")
    class_id_to_name = class_id_mapping(train_frame)

    X = train_frame[feature_cols].to_numpy(dtype=np.float32)
    y = train_frame["class_id"].to_numpy(dtype=np.int64)
    X_test = test_frame[feature_cols].to_numpy(dtype=np.float32)
    classes = np.array(sorted(class_id_to_name))
    final_train_size = len(y)

    print(f"Training samples: {final_train_size}")
    print(f"Held-out validation split: {VAL_SIZE:.0%}")
    print(f"Voting mode is selected with {CV_FOLDS}-fold CV on the training split.")
    print("Final RF, SVM, and CNN are retrained on 100% of Task 2 training data.")

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_frame)),
        test_size=max(VAL_SIZE, len(class_id_to_name) / len(train_frame)),
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    train_split_frame = train_frame.iloc[train_idx].reset_index(drop=True)
    valid_frame = train_frame.iloc[valid_idx].reset_index(drop=True)

    print(f"Training split samples  : {len(y_train)}")
    print(f"Validation split samples: {len(y_valid)}")

    fold_rows = []

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    total_start = time.time()
    for fold, (inner_train_idx, inner_val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        print("\n" + "=" * 60)
        print(f"Voting CV fold {fold}/{CV_FOLDS}")
        print("=" * 60)

        rf = build_rf()
        svm = build_svm()
        fold_train_meta = train_split_frame.iloc[inner_train_idx].reset_index(drop=True)
        fold_val_meta = train_split_frame.iloc[inner_val_idx].reset_index(drop=True)

        print("  Training RF...")
        rf.fit(X_train[inner_train_idx], y_train[inner_train_idx])
        print("  Training SVM...")
        svm.fit(X_train[inner_train_idx], y_train[inner_train_idx])
        print("  Training CNN...")
        cnn = train_cnn_fold(fold_train_meta, fold_val_meta, device, train_transforms, val_transforms)

        rf_probs = rf.predict_proba(X_train[inner_val_idx])
        svm_probs = svm.predict_proba(X_train[inner_val_idx])
        cnn_probs = predict_cnn_proba(cnn, fold_val_meta, device, val_transforms)
        avg_probs = (rf_probs + svm_probs + cnn_probs) / 3.0

        rf_pred = classes[np.argmax(rf_probs, axis=1)]
        svm_pred = classes[np.argmax(svm_probs, axis=1)]
        cnn_pred = classes[np.argmax(cnn_probs, axis=1)]
        hard_pred = hard_vote(
            [rf_pred, svm_pred, cnn_pred],
            avg_probs,
            classes,
        )
        soft_pred = classes[np.argmax(avg_probs, axis=1)]

        row = {
            "fold": fold,
            "rf": (rf_pred == y_train[inner_val_idx]).mean(),
            "svm": (svm_pred == y_train[inner_val_idx]).mean(),
            "cnn": (cnn_pred == y_train[inner_val_idx]).mean(),
            "hard": (hard_pred == y_train[inner_val_idx]).mean(),
            "soft": (soft_pred == y_train[inner_val_idx]).mean(),
        }
        fold_rows.append(row)
        print(
            "  Accuracies | "
            f"RF {row['rf']:.4f} | SVM {row['svm']:.4f} | CNN {row['cnn']:.4f} | "
            f"hard {row['hard']:.4f} | soft {row['soft']:.4f}"
        )

    fold_df = pd.DataFrame(fold_rows)
    voting_rows = []
    for voting in ("hard", "soft"):
        row = {
            "voting": voting,
            "mean_accuracy": fold_df[voting].mean(),
            "std_accuracy": fold_df[voting].std(ddof=0),
        }
        for fold_idx, fold_accuracy in enumerate(fold_df[voting], start=1):
            row[f"fold_{fold_idx}_accuracy"] = fold_accuracy
        voting_rows.append(row)
    voting_results = (
        pd.DataFrame(voting_rows)
        .sort_values("mean_accuracy", ascending=False)
        .reset_index(drop=True)
    )
    best_voting = voting_results.loc[0, "voting"]

    voting_path = OUTPUT_DIR / f"{prefix}_voting_cv_results.csv"
    voting_results.to_csv(voting_path, index=False)

    print("\n" + "=" * 60)
    print(f"Best voting mode from training-split CV: {best_voting}")
    print("Training RF, SVM, and CNN on the 80% split for held-out validation...")
    print("=" * 60)

    rf_valid_model = build_rf()
    svm_valid_model = build_svm()
    print("  Training RF...")
    rf_valid_model.fit(X_train, y_train)
    print("  Training SVM...")
    svm_valid_model.fit(X_train, y_train)
    print("  Training CNN...")
    cnn_valid_model = train_cnn_fold(train_split_frame, valid_frame, device, train_transforms, val_transforms)

    rf_valid_probs = rf_valid_model.predict_proba(X_valid)
    svm_valid_probs = svm_valid_model.predict_proba(X_valid)
    cnn_valid_probs = predict_cnn_proba(cnn_valid_model, valid_frame, device, val_transforms)
    valid_soft_probs = (rf_valid_probs + svm_valid_probs + cnn_valid_probs) / 3.0

    rf_valid_pred = classes[np.argmax(rf_valid_probs, axis=1)]
    svm_valid_pred = classes[np.argmax(svm_valid_probs, axis=1)]
    cnn_valid_pred = classes[np.argmax(cnn_valid_probs, axis=1)]
    hard_valid_pred = hard_vote([rf_valid_pred, svm_valid_pred, cnn_valid_pred], valid_soft_probs, classes)
    soft_valid_pred = classes[np.argmax(valid_soft_probs, axis=1)]
    selected_pred = hard_valid_pred if best_voting == "hard" else soft_valid_pred

    validation_scores = {
        "rf": (rf_valid_pred == y_valid).mean(),
        "svm": (svm_valid_pred == y_valid).mean(),
        "cnn": (cnn_valid_pred == y_valid).mean(),
        "hard": (hard_valid_pred == y_valid).mean(),
        "soft": (soft_valid_pred == y_valid).mean(),
    }
    print(
        "  Held-out accuracies | "
        f"RF {validation_scores['rf']:.4f} | "
        f"SVM {validation_scores['svm']:.4f} | "
        f"CNN {validation_scores['cnn']:.4f} | "
        f"hard {validation_scores['hard']:.4f} | "
        f"soft {validation_scores['soft']:.4f}"
    )

    report_path, cm_path = save_classification_outputs(prefix, y_valid, selected_pred, class_id_to_name)
    comparison_csv, comparison_png = save_model_comparison(prefix, validation_scores, best_voting)
    heatmap_csv, heatmap_png = save_soft_vote_heatmap(
        prefix,
        valid_soft_probs,
        {"rf": rf_valid_probs, "svm": svm_valid_probs, "cnn": cnn_valid_probs},
        y_valid,
        valid_frame,
        class_id_to_name,
    )

    print("\n" + "=" * 60)
    print(f"Best voting mode: {best_voting}")
    print(f"Retraining RF, SVM, and CNN on all {final_train_size} Task 2 training samples...")
    print("=" * 60)

    rf_final = build_rf()
    svm_final = build_svm()
    rf_final.fit(X, y)
    svm_final.fit(X, y)
    cnn_final = train_cnn_full(train_frame.reset_index(drop=True), device, train_transforms)

    test_meta = test_frame.reset_index(drop=True)
    rf_test_probs = rf_final.predict_proba(X_test)
    svm_test_probs = svm_final.predict_proba(X_test)
    cnn_test_probs = predict_cnn_proba(cnn_final, test_meta, device, val_transforms)
    test_avg_probs = (rf_test_probs + svm_test_probs + cnn_test_probs) / 3.0

    rf_test_pred = classes[np.argmax(rf_test_probs, axis=1)]
    svm_test_pred = classes[np.argmax(svm_test_probs, axis=1)]
    cnn_test_pred = classes[np.argmax(cnn_test_probs, axis=1)]

    if best_voting == "hard":
        test_pred = hard_vote([rf_test_pred, svm_test_pred, cnn_test_pred], test_avg_probs, classes)
    else:
        test_pred = classes[np.argmax(test_avg_probs, axis=1)]

    predictions = pd.DataFrame(
        {
            "image_id": test_frame["image_id"].values,
            "class_id": test_pred.astype(int),
        }
    )
    predictions["class_name"] = predictions["class_id"].map(class_id_to_name)
    predictions_path = OUTPUT_DIR / f"{prefix}_predictions.csv"
    submission_path = OUTPUT_DIR / f"{prefix}_submission.csv"
    predictions.to_csv(predictions_path, index=False)
    predictions[["image_id", "class_id"]].to_csv(submission_path, index=False)

    summary_path = OUTPUT_DIR / f"{prefix}_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "Task: task2",
                "Models: random_forest, svm, cnn",
                f"RF params: {RF_PARAMS}",
                f"SVM params: {SVM_PARAMS}",
                f"CNN LR: {CNN_LR}",
                f"CV folds: {CV_FOLDS}",
                f"Held-out validation fraction: {VAL_SIZE}",
                f"Training split samples: {len(y_train)}",
                f"Validation split samples: {len(y_valid)}",
                f"Best voting: {best_voting}",
                f"Held-out RF accuracy: {float(validation_scores['rf']):.6f}",
                f"Held-out SVM accuracy: {float(validation_scores['svm']):.6f}",
                f"Held-out CNN accuracy: {float(validation_scores['cnn']):.6f}",
                f"Held-out hard ensemble accuracy: {float(validation_scores['hard']):.6f}",
                f"Held-out soft ensemble accuracy: {float(validation_scores['soft']):.6f}",
                f"Final retrain samples: {final_train_size}",
                "Final retrain data: 100% of Task 2 training data",
                f"Hard training-split CV mean accuracy: {float(voting_results.loc[voting_results['voting'] == 'hard', 'mean_accuracy'].iloc[0]):.6f}",
                f"Soft training-split CV mean accuracy: {float(voting_results.loc[voting_results['voting'] == 'soft', 'mean_accuracy'].iloc[0]):.6f}",
                f"Runtime minutes: {(time.time() - total_start) / 60:.2f}",
                f"Voting CV: {voting_path}",
                f"Classification report: {report_path}",
                f"Confusion matrix: {cm_path}",
                f"Model comparison: {comparison_png}",
                f"Soft vote heatmap: {heatmap_png}",
                f"Predictions: {predictions_path}",
                f"Submission: {submission_path}",
            ]
        )
    )

    print("\nEnsemble outputs:")
    print(f"  Voting CV          : {voting_path}")
    print(f"  Classification CSV : {report_path}")
    print(f"  Confusion matrix   : {cm_path}")
    print(f"  Model comparison   : {comparison_png}")
    print(f"  Soft vote heatmap  : {heatmap_png}")
    print(f"  Submission         : {submission_path}")


if __name__ == "__main__":
    main()
