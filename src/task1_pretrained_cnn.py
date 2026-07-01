# Rebuilt from Task1 (3).ipynb: Task 1 Pre-trained CNN.

import os
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.models as models
from torchvision.models import ResNet18_Weights

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from data_preprocessing import MetadataImageDataset, make_tsne, task1_pretrained_transforms


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = "task1_data"
TRAIN_META = os.path.join(DATA_DIR, "train_metadata.csv")
TEST_META  = os.path.join(DATA_DIR, "test_metadata.csv")
IMAGE_ROOT = DATA_DIR
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_CLASSES  = 10
BATCH_SIZE   = 64
CV_EPOCHS  = 5    
NUM_EPOCHS = 30
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.2
RANDOM_STATE = 42
CV_PATIENCE    = 5   # aggressive early stopping during CV (speed)
PATIENCE       = 15  # more patient for final training (accuracy)
N_FOLDS      = 2     # 2-fold CV for LR tuning (keeps runtime reasonable)

# Learning rate candidates to evaluate
LR_CANDIDATES = [1e-3, 1e-4, 1e-5]

# Output paths
MODEL_SAVE_PATH  = str(OUTPUT_DIR / "task1_resnet18_best.pth")
SUBMISSION_PATH  = str(OUTPUT_DIR / "task1_resnet18_submission.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device     : {DEVICE}")
print(f"LR candidates    : {LR_CANDIDATES}")
print(f"CV folds         : {N_FOLDS}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

train_transforms, val_test_transforms = task1_pretrained_transforms()

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────────────────────────────────────

AnimalDataset = MetadataImageDataset

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes, device):
    """Load ImageNet-pretrained ResNet-18 and replace the final layer."""
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features   # 512
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN / EVALUATE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct      += (logits.argmax(dim=1) == labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0
    all_preds    = []
    all_labels   = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits    = model(images)
            loss      = criterion(logits, labels)
            predicted = logits.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            correct      += (predicted == labels).sum().item()
            total        += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_labels, all_preds

# ─────────────────────────────────────────────────────────────────────────────
# 6. SINGLE TRAINING RUN (one fold, one LR)
# ─────────────────────────────────────────────────────────────────────────────

def train_run(train_meta, val_meta, learning_rate, device, verbose=True):
    """Train for up to NUM_EPOCHS with early stopping. Returns best val acc."""
    model     = build_model(NUM_CLASSES, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    train_loader = DataLoader(
        AnimalDataset(train_meta, IMAGE_ROOT, train_transforms),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        AnimalDataset(val_meta, IMAGE_ROOT, val_test_transforms),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    patience_count = 0

    for epoch in range(1, CV_EPOCHS + 1):
        train_loss, train_acc   = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if verbose:
            print(f"    Epoch {epoch:02d} | train acc {train_acc:.4f} | "
                  f"val acc {val_acc:.4f} | val loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_val_acc   = val_acc
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= CV_PATIENCE:
                if verbose:
                    print(f"    Early stopping at epoch {epoch}")
                break

    return best_val_acc

# ─────────────────────────────────────────────────────────────────────────────
# 7. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

train_df    = pd.read_csv(TRAIN_META)
classes     = sorted(train_df["class_name"].unique())
id_to_class = dict(zip(train_df["class_id"], train_df["class_name"]))
print(f"\nClasses ({NUM_CLASSES}): {classes}")
print(f"Total training samples : {len(train_df)}")

train_meta_split, val_meta_split = train_test_split(
    train_df,
    test_size=VAL_SPLIT,
    stratify=train_df["class_id"],
    random_state=RANDOM_STATE,
)
train_meta_split = train_meta_split.reset_index(drop=True)
val_meta_split = val_meta_split.reset_index(drop=True)
print(f"LR tuning/training pool : {len(train_meta_split)} ({1 - VAL_SPLIT:.0%})")
print(f"Held-out validation    : {len(val_meta_split)} ({VAL_SPLIT:.0%})")

# ─────────────────────────────────────────────────────────────────────────────
# 8. LEARNING RATE TUNING — STRATIFIED K-FOLD CV
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Learning Rate Tuning — Stratified K-Fold CV on Training Pool")
print("=" * 60)

skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=RANDOM_STATE)
cv_results = {lr: [] for lr in LR_CANDIDATES}
total_start = time.time()

for lr in LR_CANDIDATES:
    print(f"\n── LR = {lr} ──────────────────────────────────────────")
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_meta_split, train_meta_split["class_id"]), start=1
    ):
        fold_start = time.time()
        fold_train = train_meta_split.iloc[train_idx].reset_index(drop=True)
        fold_val   = train_meta_split.iloc[val_idx].reset_index(drop=True)

        print(f"  Starting fold {fold}/{N_FOLDS}...")
        best_acc = train_run(fold_train, fold_val, lr, DEVICE, verbose=True)
        cv_results[lr].append(best_acc)

        print(f"  Fold {fold}/{N_FOLDS}  val acc: {best_acc:.4f}  "
              f"({time.time() - fold_start:.0f}s)")

    mean_acc = np.mean(cv_results[lr])
    std_acc  = np.std(cv_results[lr])
    print(f"  → Mean: {mean_acc:.4f} ± {std_acc:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LR Tuning Summary")
print("=" * 60)
print(f"{'LR':<10} {'Mean Acc':>10} {'Std':>8}  Fold accuracies")
print("-" * 60)

best_lr   = None
best_mean = -1

for lr, accs in cv_results.items():
    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    fold_str = "  ".join([f"{a:.4f}" for a in accs])
    print(f"{lr:<10} {mean_acc:>10.4f} {std_acc:>8.4f}  {fold_str}")
    if mean_acc > best_mean:
        best_mean = mean_acc
        best_lr   = lr

print(f"\n  Best LR: {best_lr}  (mean CV acc: {best_mean:.4f})")
print(f"  Total CV time: {(time.time() - total_start) / 60:.1f} min")

# ── LR tuning bar chart  →  task1_pretrainedcnn_lr_tuning.png ────────────────
means    = [np.mean(cv_results[lr]) for lr in LR_CANDIDATES]
stds     = [np.std(cv_results[lr])  for lr in LR_CANDIDATES]
x_labels = [str(lr) for lr in LR_CANDIDATES]
colors   = ["#4C72B0", "#DD8452", "#55A868"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(x_labels, means, yerr=stds, capsize=6,
              color=colors[:len(LR_CANDIDATES)], width=0.5)
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2,
            mean + std + 0.008,
            f"{mean:.4f}\n±{std:.4f}",
            ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Mean CV Accuracy")
ax.set_title(f"Task 1 — Pre-trained ResNet-18: LR Tuning ({N_FOLDS}-Fold CV)")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task1_pretrainedcnn_lr_tuning.png", dpi=150)
plt.close()
print(f"  LR tuning chart saved to: {OUTPUT_DIR / 'task1_pretrainedcnn_lr_tuning.png'}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. FINAL TRAINING WITH BEST LR
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"Final Training — best LR = {best_lr}")
print("=" * 60)

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
print(f"Train samples: {len(train_meta_split)}, Val samples: {len(val_meta_split)}")

# Build final model, criterion, optimiser, scheduler
model         = build_model(NUM_CLASSES, DEVICE)
criterion     = nn.CrossEntropyLoss()
optimizer     = optim.Adam(model.parameters(), lr=best_lr,
                           weight_decay=WEIGHT_DECAY)
scheduler     = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3,
)

train_loader = DataLoader(
    AnimalDataset(train_meta_split, IMAGE_ROOT, train_transforms),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)
val_loader = DataLoader(
    AnimalDataset(val_meta_split, IMAGE_ROOT, val_test_transforms),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)

history        = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_loss  = float("inf")
best_weights   = None
best_epoch     = 0
patience_count = 0
start_time     = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc   = train_one_epoch(model, train_loader, criterion,
                                               optimizer, DEVICE)
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    elapsed = time.time() - start_time
    print(f"  Epoch {epoch:02d}/{NUM_EPOCHS} | "
          f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
          f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f} | "
          f"Time: {elapsed:.0f}s")

    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        best_weights   = copy.deepcopy(model.state_dict())
        best_epoch     = epoch
        patience_count = 0
        torch.save(best_weights, MODEL_SAVE_PATH)
        print(f"    ✓ New best val loss: {best_val_loss:.4f} — model saved")
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\n  Early stopping triggered after {epoch} epochs.")
            break

if best_weights is not None:
    model.load_state_dict(best_weights)
print(f"\nTraining complete. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. LOSS & ACCURACY CURVES  →  task1_pretrainedcnn_loss_accuracy_curves.png
# ─────────────────────────────────────────────────────────────────────────────

epochs_ran = range(1, len(history["train_loss"]) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs_ran, history["train_loss"], label="Train loss")
ax1.plot(epochs_ran, history["val_loss"],   label="Val loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title(f"Task 1 Pre-trained ResNet-18 — Loss (LR={best_lr})")
ax1.legend()

ax2.plot(epochs_ran, history["train_acc"], label="Train acc")
ax2.plot(epochs_ran, history["val_acc"],   label="Val acc")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title(f"Task 1 Pre-trained ResNet-18 — Accuracy (LR={best_lr})")
ax2.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task1_pretrainedcnn_loss_accuracy_curves.png", dpi=150)
plt.close()
print(f"Loss/accuracy curves saved to: {OUTPUT_DIR / 'task1_pretrainedcnn_loss_accuracy_curves.png'}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. EVALUATION ON VALIDATION SET
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Evaluation on validation set")
print("=" * 60)

_, val_acc_final, true_labels, pred_labels = evaluate(
    model, val_loader, criterion, DEVICE
)

print(f"\n  Validation accuracy : {val_acc_final:.4f}  ({val_acc_final*100:.1f}%)")
print("\n  Per-class classification report:")
print(classification_report(true_labels, pred_labels,
                             target_names=classes, digits=4))

# ── Confusion matrix  →  task1_pretrainedcnn_cm.png ──────────────────────────
cm = confusion_matrix(true_labels, pred_labels)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title(f"Task 1 — Pre-trained ResNet-18 Confusion Matrix\n"
             f"Validation accuracy: {val_acc_final:.2%}  (LR={best_lr})",
             fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task1_pretrainedcnn_cm.png", dpi=150)
plt.close()
print(f"  Confusion matrix saved to: {OUTPUT_DIR / 'task1_pretrainedcnn_cm.png'}")

# ── Classification report CSV  →  task1_pretrainedcnn_classification_report.csv
report_dict = classification_report(
    true_labels, pred_labels,
    target_names=classes, digits=4, output_dict=True,
)
report_df = pd.DataFrame(report_dict).transpose().reset_index()
report_df.rename(columns={"index": "class"}, inplace=True)
report_df.to_csv(OUTPUT_DIR / "task1_pretrainedcnn_classification_report.csv", index=False)
print(f"  Classification report saved to: {OUTPUT_DIR / 'task1_pretrainedcnn_classification_report.csv'}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. t-SNE VISUALISATION  →  task1_pretrainedcnn_tsne.png
# ─────────────────────────────────────────────────────────────────────────────
# We extract the 512-dim feature vector from ResNet-18's penultimate layer
# (just before the final fc) for every validation image.
# t-SNE then projects these 512-dim vectors down to 2-D for plotting.
# Tight, well-separated clusters indicate the backbone has learned
# discriminative features for each class.

print("\n" + "=" * 60)
print("t-SNE Feature Visualisation")
print("=" * 60)

# ── Hook to capture the 512-dim backbone output ───────────────────────────────
# We register a forward hook on model.avgpool, which outputs the
# (batch, 512, 1, 1) tensor that feeds into the fc layer.
features_store = {}

def hook_fn(module, input, output):
    # Flatten (batch, 512, 1, 1) → (batch, 512)
    features_store["feats"] = output.squeeze(-1).squeeze(-1).detach().cpu()

hook = model.avgpool.register_forward_hook(hook_fn)

# ── Collect features + labels for the entire validation set ──────────────────
model.eval()
all_feats  = []
all_labels_tsne = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        model(images)                            # forward pass triggers hook
        all_feats.append(features_store["feats"])
        all_labels_tsne.extend(labels.numpy())

hook.remove()   # clean up hook after use

all_feats = torch.cat(all_feats, dim=0).numpy()   # (n_val, 512)
all_labels_tsne = np.array(all_labels_tsne)

print(f"  Extracted features: {all_feats.shape}")
print("  Running t-SNE (this may take ~30s)...")

tsne      = make_tsne(n_components=2, perplexity=30, random_state=RANDOM_STATE,
                      max_iter=1000)
feats_2d  = tsne.fit_transform(all_feats)        # (n_val, 2)

# ── Plot ──────────────────────────────────────────────────────────────────────
# Each point is one validation image; colour = true class.
# A good backbone produces tight, separated blobs.

palette = plt.cm.get_cmap("tab10", NUM_CLASSES)

fig, ax = plt.subplots(figsize=(10, 8))

for class_id, class_name in sorted(id_to_class.items()):
    mask = all_labels_tsne == class_id
    ax.scatter(
        feats_2d[mask, 0], feats_2d[mask, 1],
        label=class_name,
        color=palette(class_id),
        s=30, alpha=0.75, edgecolors="none",
    )

ax.set_title(
    f"Task 1 — t-SNE of Pre-trained ResNet-18 Features\n"
    f"(validation set, 512-dim → 2-D,  val acc={val_acc_final:.2%})",
    fontsize=13,
)
ax.set_xlabel("t-SNE dimension 1")
ax.set_ylabel("t-SNE dimension 2")
ax.legend(loc="best", fontsize=9, markerscale=1.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task1_pretrainedcnn_tsne.png", dpi=150)
plt.close()
print(f"  t-SNE plot saved to: {OUTPUT_DIR / 'task1_pretrainedcnn_tsne.png'}")

# ─────────────────────────────────────────────────────────────────────────────
# 13. TEST-SET PREDICTION  →  task1_resnet18_submission.csv
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Generating test predictions for Kaggle submission")
print("=" * 60)

if not os.path.exists(TEST_META):
    print("  test_metadata.csv not found — skipping.")
else:
    test_df = pd.read_csv(TEST_META)

    test_loader = DataLoader(
        AnimalDataset(test_df, IMAGE_ROOT, val_test_transforms, has_labels=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    model.eval()
    all_image_ids    = []
    all_pred_classes = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images    = images.to(DEVICE)
            predicted = model(images).argmax(dim=1).cpu().numpy()
            all_image_ids.extend(image_ids)
            all_pred_classes.extend([id_to_class[p] for p in predicted])

    # Save class_name version
    submission = pd.DataFrame({
        "image_id"  : all_image_ids,
        "class_name": all_pred_classes,
    })
    submission.to_csv(SUBMISSION_PATH, index=False)

    # Save class_id version (Kaggle format)
    class_to_id = (
        train_df[["class_name", "class_id"]]
        .drop_duplicates()
        .set_index("class_name")["class_id"]
        .to_dict()
    )
    submission_id = pd.DataFrame({
        "image_id": all_image_ids,
        "class_id": [class_to_id[n] for n in all_pred_classes],
    })
    submission_id.to_csv(
        SUBMISSION_PATH.replace(".csv", "_classid.csv"), index=False
    )

    print(f"  class_name submission : {SUBMISSION_PATH}")
    print(f"  class_id   submission : {SUBMISSION_PATH.replace('.csv', '_classid.csv')}")
    print(f"  Total predictions     : {len(submission_id)}")
    print(f"  Class distribution:\n{submission['class_name'].value_counts()}")
    print(f"\n  Best LR used: {best_lr}")
