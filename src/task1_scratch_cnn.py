# Rebuilt from Task1 (3).ipynb: Task 1 CNN trained from scratch.

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
# CHANGED: removed ResNet18_Weights import — not needed for scratch training

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from data_preprocessing import MetadataImageDataset, make_tsne, task1_scratch_transforms


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
# CHANGED: 5 → 15
# Scratch model needs more epochs per CV fold to learn meaningful features
# from random weights. Pre-trained model already has useful features after
# 1-2 epochs; scratch model needs many more passes just to learn basic edges.
CV_EPOCHS    = 15
# CHANGED: 30 → 100
# Training from scratch is fundamentally slower — the model must learn
# everything from random weights up through edges, textures, shapes, and
# finally class-specific features. Pre-trained only needed ~15-20 epochs;
# scratch needs up to 100 (early stopping will cut this short in practice).
NUM_EPOCHS   = 100
# CHANGED: 1e-4 → 1e-3
# Stronger L2 regularisation for scratch training. Without ImageNet weights
# as a stable starting point, the model is much more prone to overfitting
# on the ~3,000 training images. Higher weight decay penalises large weights
# more aggressively, preventing the model from memorising training data.
WEIGHT_DECAY = 1e-3
VAL_SPLIT    = 0.2
RANDOM_STATE = 42
# CHANGED: 5 → 7
# Scratch training loss curves are noisier — the model oscillates more
# before converging. More patience prevents premature stopping during CV.
CV_PATIENCE  = 7
# CHANGED: 15 → 20
# Final training needs more patience — scratch convergence is slower and
# noisier. Stopping too early would underestimate the model's true capacity.
PATIENCE     = 20
N_FOLDS      = 2

# CHANGED: LR range shifted higher — [1e-3, 1e-4, 1e-5] → [1e-2, 1e-3, 1e-4]
# Pre-trained fine-tuning uses small LR to avoid destroying ImageNet weights.
# Scratch training has no pre-trained weights to protect, so larger LR is
# needed to drive learning from random initialisation. 1e-5 would be far
# too slow for scratch; 1e-2 is the natural starting point.
LR_CANDIDATES = [1e-2, 1e-3, 1e-4]

# Output paths — CHANGED: task1_pretrainedcnn → task1_scratchcnn
MODEL_SAVE_PATH = str(OUTPUT_DIR / "task1_scratchcnn_best.pth")
SUBMISSION_PATH = str(OUTPUT_DIR / "task1_scratchcnn_submission.csv")

# CHANGED: added MPS support for Apple Silicon iMacs
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device     : {DEVICE}")
print(f"Training mode    : FROM SCRATCH (no pre-trained weights)")
print(f"LR candidates    : {LR_CANDIDATES}")
print(f"CV folds         : {N_FOLDS}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
# Normalisation kept identical to pre-trained version (ImageNet stats).
# Even without pre-trained weights, normalising to ~[-2, 2] keeps gradients
# numerically stable and makes LR choices more consistent.
#
# CHANGED: augmentation slightly strengthened vs pre-trained Task 1.
# Scratch model is more prone to overfitting because it has no prior
# knowledge — stronger augmentation increases effective dataset diversity.
# RandomGrayscale added to force learning of shape-based rather than
# purely colour-based features from the start.

train_transforms, val_test_transforms = task1_scratch_transforms()

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────────────────────────────────────
# Unchanged from pre-trained version.

AnimalDataset = MetadataImageDataset

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes, device):
    """
    Build ResNet-18 with RANDOM weights (no pre-training).

    CHANGED from pre-trained version:
      - weights=None instead of ResNet18_Weights.IMAGENET1K_V1
        All 11M parameters are randomly initialised using PyTorch's default
        He/Kaiming initialisation (designed for ReLU networks).
        The model must learn EVERYTHING from the training data:
          layer1 → must learn to detect edges/colours from scratch
          layer2 → must learn textures/shapes from scratch
          layer3 → must learn object parts from scratch
          layer4 → must learn class-specific features from scratch
          fc     → must learn classification from scratch

      - No layers are frozen (nothing to freeze — all weights are random,
        freezing random weights would just prevent learning entirely)

      - Dropout(0.5) added before fc layer (same as Task 2 pre-trained).
        Without ImageNet features as a stable foundation, overfitting risk
        is higher. Dropout forces robust feature combinations.

    This model serves as an ablation study: comparing its accuracy to the
    pre-trained version isolates how much performance comes from ImageNet
    weights vs the ResNet-18 architecture itself.
    """
    # CHANGED: weights=None → random He/Kaiming initialisation
    # No download, no ImageNet weights, pure random start
    model = models.resnet18(weights=None)

    # CHANGED: added Dropout(0.5) before final layer
    # Higher regularisation needed — no pre-trained features to stabilise training
    in_features = model.fc.in_features   # 512
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    return model.to(device)

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN / EVALUATE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
# Unchanged from pre-trained version.

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

def train_run(train_meta, val_meta, learning_rate, device, verbose=False):
    """Train for up to CV_EPOCHS with early stopping. Returns best val acc."""
    model     = build_model(NUM_CLASSES, device)
    criterion = nn.CrossEntropyLoss()

    # CHANGED: model.parameters() instead of filter(requires_grad)
    # No layers are frozen so all parameters need to be optimised.
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    # CHANGED: CosineAnnealingLR instead of ReduceLROnPlateau
    # Scratch training benefits from smooth LR decay:
    #   - Large LR early helps escape random-init local minima quickly
    #   - Gradually smaller LR later makes fine-grained adjustments
    # ReduceLROnPlateau can trigger too early on the noisy loss curves
    # typical of scratch training, stunting learning prematurely.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CV_EPOCHS, eta_min=1e-6
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
        scheduler.step()   # CHANGED: .step() not .step(val_loss) for cosine annealing

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
# Unchanged from pre-trained version.

train_df    = pd.read_csv(TRAIN_META)
classes     = sorted(train_df["class_name"].unique())
id_to_class = dict(zip(train_df["class_id"], train_df["class_name"]))
print(f"\nTask 1 — CNN Trained From Scratch")
print(f"Classes ({NUM_CLASSES}): {classes}")
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
# LR tuning is restricted to the training pool so the held-out validation
# split remains unseen during hyperparameter selection.

print("\n" + "=" * 60)
print("Task 1 Scratch CNN — LR Tuning — Stratified K-Fold CV on Training Pool")
print("=" * 60)

skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE)
cv_results  = {lr: [] for lr in LR_CANDIDATES}
total_start = time.time()

for lr in LR_CANDIDATES:
    print(f"\n── LR = {lr} ──────────────────────────────────────────")
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_meta_split, train_meta_split["class_id"]), start=1
    ):
        fold_start = time.time()
        fold_train = train_meta_split.iloc[train_idx].reset_index(drop=True)
        fold_val   = train_meta_split.iloc[val_idx].reset_index(drop=True)

        best_acc = train_run(fold_train, fold_val, lr, DEVICE, verbose=False)
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

# ── LR tuning bar chart  →  task1_scratchcnn_lr_tuning.png ───────────────────
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
ax.set_title(f"Task 1 — Scratch ResNet-18: LR Tuning ({N_FOLDS}-Fold CV)")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task1_scratchcnn_lr_tuning.png", dpi=150)
plt.close()
print(f"  LR tuning chart saved to: {OUTPUT_DIR / 'task1_scratchcnn_lr_tuning.png'}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. FINAL TRAINING WITH BEST LR
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"Task 1 Scratch — Final Training — best LR = {best_lr}")
print("=" * 60)

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
print(f"Train samples: {len(train_meta_split)}, Val samples: {len(val_meta_split)}")

model     = build_model(NUM_CLASSES, DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=best_lr,
    weight_decay=WEIGHT_DECAY,
)
# CHANGED: CosineAnnealingLR for final training (same reasoning as CV runs)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
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
    scheduler.step()   # CHANGED: no argument for CosineAnnealingLR

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    elapsed = time.time() - start_time
    print(f"  Epoch {epoch:03d}/{NUM_EPOCHS} | "
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
# 10. LOSS & ACCURACY CURVES  →  task1_scratchcnn_loss_accuracy_curves.png
# ─────────────────────────────────────────────────────────────────────────────
# Unchanged from pre-trained version (structure identical).
# Expect scratch curves to show:
#   - Slower initial improvement (no head start from ImageNet)
#   - Larger train/val gap (more overfitting without pre-trained regularisation)
#   - Lower final accuracy

epochs_ran = range(1, len(history["train_loss"]) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs_ran, history["train_loss"], label="Train loss")
ax1.plot(epochs_ran, history["val_loss"],   label="Val loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title(f"Task 1 Scratch ResNet-18 — Loss (LR={best_lr})")
ax1.legend()

ax2.plot(epochs_ran, history["train_acc"], label="Train acc")
ax2.plot(epochs_ran, history["val_acc"],   label="Val acc")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title(f"Task 1 Scratch ResNet-18 — Accuracy (LR={best_lr})")
ax2.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task1_scratchcnn_loss_accuracy_curves.png", dpi=150)
plt.close()
print(f"Loss/accuracy curves saved to: {OUTPUT_DIR / 'task1_scratchcnn_loss_accuracy_curves.png'}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. EVALUATION ON VALIDATION SET
# ─────────────────────────────────────────────────────────────────────────────
# Unchanged from pre-trained version.

print("\n" + "=" * 60)
print("Task 1 Scratch — Evaluation on validation set")
print("=" * 60)

_, val_acc_final, true_labels, pred_labels = evaluate(
    model, val_loader, criterion, DEVICE
)

print(f"\n  Validation accuracy : {val_acc_final:.4f}  ({val_acc_final*100:.1f}%)")
print("\n  Per-class classification report:")
print(classification_report(true_labels, pred_labels,
                             target_names=classes, digits=4))

# ── Confusion matrix  →  task1_scratchcnn_cm.png ─────────────────────────────
cm = confusion_matrix(true_labels, pred_labels)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title(f"Task 1 — Scratch ResNet-18 Confusion Matrix\n"
             f"Validation accuracy: {val_acc_final:.2%}  (LR={best_lr})",
             fontsize=13)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task1_scratchcnn_cm.png", dpi=150)
plt.close()
print(f"  Confusion matrix saved to: {OUTPUT_DIR / 'task1_scratchcnn_cm.png'}")

# ── Classification report CSV  →  task1_scratchcnn_classification_report.csv ─
report_dict = classification_report(
    true_labels, pred_labels,
    target_names=classes, digits=4, output_dict=True,
)
report_df = pd.DataFrame(report_dict).transpose().reset_index()
report_df.rename(columns={"index": "class"}, inplace=True)
report_df.to_csv(OUTPUT_DIR / "task1_scratchcnn_classification_report.csv", index=False)
print(f"  Classification report saved to: {OUTPUT_DIR / 'task1_scratchcnn_classification_report.csv'}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. t-SNE VISUALISATION  →  task1_scratchcnn_tsne.png
# ─────────────────────────────────────────────────────────────────────────────
# Unchanged from pre-trained version.
# Compare this t-SNE to the pre-trained version:
#   Pre-trained: expect tight, well-separated clusters (rich ImageNet features)
#   Scratch:     expect looser, more overlapping clusters (self-learned features)
# The difference visually explains the accuracy gap between the two models.

print("\n" + "=" * 60)
print("Task 1 Scratch — t-SNE Feature Visualisation")
print("=" * 60)

features_store = {}

def hook_fn(module, input, output):
    features_store["feats"] = output.squeeze(-1).squeeze(-1).detach().cpu()

hook = model.avgpool.register_forward_hook(hook_fn)

model.eval()
all_feats       = []
all_labels_tsne = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        model(images)
        all_feats.append(features_store["feats"])
        all_labels_tsne.extend(labels.numpy())

hook.remove()

all_feats       = torch.cat(all_feats, dim=0).numpy()
all_labels_tsne = np.array(all_labels_tsne)

print(f"  Extracted features: {all_feats.shape}")
print("  Running t-SNE (this may take ~30s)...")

tsne     = make_tsne(n_components=2, perplexity=30, random_state=RANDOM_STATE,
                     max_iter=1000)
feats_2d = tsne.fit_transform(all_feats)

palette = plt.get_cmap("tab10", NUM_CLASSES)

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
    f"Task 1 — t-SNE of Scratch ResNet-18 Features\n"
    f"(validation set, 512-dim → 2-D,  val acc={val_acc_final:.2%})",
    fontsize=13,
)
ax.set_xlabel("t-SNE dimension 1")
ax.set_ylabel("t-SNE dimension 2")
ax.legend(loc="best", fontsize=9, markerscale=1.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task1_scratchcnn_tsne.png", dpi=150)
plt.close()
print(f"  t-SNE plot saved to: {OUTPUT_DIR / 'task1_scratchcnn_tsne.png'}")

# ─────────────────────────────────────────────────────────────────────────────
# 13. TEST-SET PREDICTION  →  task1_scratchcnn_submission.csv
# ─────────────────────────────────────────────────────────────────────────────
# Unchanged from pre-trained version.

print("\n" + "=" * 60)
print("Task 1 Scratch — Generating test predictions for Kaggle")
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
