# Rebuilt from Task2 (2).ipynb: Task 2 Pre-trained CNN.

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

from data_preprocessing import MetadataImageDataset, make_tsne, task2_pretrained_transforms


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = "task2_data"
TRAIN_META = os.path.join(DATA_DIR, "train_metadata.csv")
TEST_META  = os.path.join(DATA_DIR, "test_metadata.csv")
IMAGE_ROOT = DATA_DIR
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_CLASSES  = 10
# CHANGED: 64 → 16
# Task 2 has ~350-400 images total (~35-40 per class).
# Smaller batch size gives more gradient updates per epoch,
# which is important when each epoch sees so few images.
BATCH_SIZE   = 16
# CHANGED: 5 → 15
# More CV epochs needed — with only ~280 training images per fold,
# the model needs more passes to show meaningful accuracy signal.
CV_EPOCHS    = 15
# CHANGED: 30 → 50
# Small dataset converges slowly from pre-trained weights; more
# epochs allowed (early stopping will cut this short in practice).
NUM_EPOCHS   = 50
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.2
RANDOM_STATE = 42
# CHANGED: 5 → 7
# Small-dataset training is noisier — need more patience before
# declaring no improvement during CV runs.
CV_PATIENCE  = 7
# CHANGED: 15 → 10
# Final training patience kept reasonable to avoid very long runs.
PATIENCE     = 10
# CHANGED: 2 → 5
# 5 folds → each training fold uses 80% of ~350 = ~280 images (~28/class).
# 2 folds would only give ~175 images per fold — too thin for 10 classes.
N_FOLDS      = 5

# Learning rate candidates to evaluate
LR_CANDIDATES = [1e-3, 1e-4, 1e-5]

# Output paths
MODEL_SAVE_PATH  = str(OUTPUT_DIR / "task2_resnet18_best.pth")
SUBMISSION_PATH  = str(OUTPUT_DIR / "task2_resnet18_submission.csv")

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device     : {DEVICE}")
print(f"LR candidates    : {LR_CANDIDATES}")
print(f"CV folds         : {N_FOLDS}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
# CHANGED: Stronger augmentation throughout for Task 2.
# With only ~35 images per class, augmentation is the primary tool
# for preventing memorisation. Each change is justified below.

train_transforms, val_test_transforms = task2_pretrained_transforms()

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────────────────────────────────────

BirdDataset = MetadataImageDataset

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes, device):
    """
    Load ImageNet-pretrained ResNet-18 adapted for fine-grained bird classification.

    CHANGED from Task 1:
      - layer1 frozen only (Task 1 froze nothing).
        Bird species differ in subtle mid-level features (feather texture,
        wing patterns) that require layer2+ to adapt. Only layer1 (basic
        edges/colours) is safe to freeze as these are universal features.
      - Dropout(0.5) added before final layer (Task 1 had no dropout).
        With only ~35 images per class, the model can easily memorise
        training data. Dropout forces robust feature combinations.
    """
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # CHANGED: freeze only layer1 — layer2+ must adapt to bird features
    # Layer roles:
    #   layer1 → basic edges, colours     (frozen — universal)
    #   layer2 → textures, simple shapes  (trainable — feather patterns)
    #   layer3 → object parts             (trainable — beaks, wings)
    #   layer4 → high-level combinations  (trainable — species features)
    for name, param in model.named_parameters():
        if "layer1" in name:
            param.requires_grad = False

    # CHANGED: Dropout(0.5) + Linear instead of plain Linear
    in_features = model.fc.in_features   # 512
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
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

def train_run(train_meta, val_meta, learning_rate, device, verbose=False):
    """Train for up to CV_EPOCHS with early stopping. Returns best val acc."""
    model     = build_model(NUM_CLASSES, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    # CHANGED: CosineAnnealingLR instead of ReduceLROnPlateau
    # Cosine annealing smoothly decays LR from initial value to eta_min
    # over T_max epochs. Better for small datasets because:
    #   - No plateau detection needed — reduction is predictable
    #   - Smooth decay helps escape local minima early in training
    #   - ReduceLROnPlateau can be too aggressive on noisy small-data loss curves
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CV_EPOCHS, eta_min=1e-6
    )

    train_loader = DataLoader(
        BirdDataset(train_meta, IMAGE_ROOT, train_transforms),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        BirdDataset(val_meta, IMAGE_ROOT, val_test_transforms),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    patience_count = 0

    for epoch in range(1, CV_EPOCHS + 1):
        train_loss, train_acc   = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

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
print(f"\nTask 2 — Fine-Grained Bird Species Classification")
print(f"Classes ({NUM_CLASSES}): {classes}")
print(f"Total training samples : {len(train_df)}")
print(f"Approx per class       : {len(train_df) // NUM_CLASSES}")

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
print("Task 2 — Learning Rate Tuning — Stratified K-Fold CV on Training Pool")
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

# ── LR tuning bar chart  →  task2_pretrainedcnn_lr_tuning.png ────────────────
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
ax.set_title(f"Task 2 — Pre-trained ResNet-18: LR Tuning ({N_FOLDS}-Fold CV)")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task2_pretrainedcnn_lr_tuning.png", dpi=150)
plt.close()
print(f"  LR tuning chart saved to: {OUTPUT_DIR / 'task2_pretrainedcnn_lr_tuning.png'}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. FINAL TRAINING WITH BEST LR
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"Task 2 — Final Training — best LR = {best_lr}")
print("=" * 60)

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
print(f"Train samples: {len(train_meta_split)}, Val samples: {len(val_meta_split)}")

model     = build_model(NUM_CLASSES, DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=best_lr,
    weight_decay=WEIGHT_DECAY,
)
# CHANGED: CosineAnnealingLR for final training (same reasoning as CV runs)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
)

train_loader = DataLoader(
    BirdDataset(train_meta_split, IMAGE_ROOT, train_transforms),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)
val_loader = DataLoader(
    BirdDataset(val_meta_split, IMAGE_ROOT, val_test_transforms),
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
    scheduler.step()

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
# 10. LOSS & ACCURACY CURVES  →  task2_pretrainedcnn_loss_accuracy_curves.png
# ─────────────────────────────────────────────────────────────────────────────

epochs_ran = range(1, len(history["train_loss"]) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs_ran, history["train_loss"], label="Train loss")
ax1.plot(epochs_ran, history["val_loss"],   label="Val loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title(f"Task 2 Pre-trained ResNet-18 — Loss (LR={best_lr})")
ax1.legend()

ax2.plot(epochs_ran, history["train_acc"], label="Train acc")
ax2.plot(epochs_ran, history["val_acc"],   label="Val acc")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title(f"Task 2 Pre-trained ResNet-18 — Accuracy (LR={best_lr})")
ax2.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task2_pretrainedcnn_loss_accuracy_curves.png", dpi=150)
plt.close()
print(f"Loss/accuracy curves saved to: {OUTPUT_DIR / 'task2_pretrainedcnn_loss_accuracy_curves.png'}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. EVALUATION ON VALIDATION SET
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Task 2 — Evaluation on validation set")
print("=" * 60)

_, val_acc_final, true_labels, pred_labels = evaluate(
    model, val_loader, criterion, DEVICE
)

print(f"\n  Validation accuracy : {val_acc_final:.4f}  ({val_acc_final*100:.1f}%)")
print("\n  Per-class classification report:")
print(classification_report(true_labels, pred_labels,
                             target_names=classes, digits=4))

# ── Confusion matrix  →  task2_pretrainedcnn_cm.png ──────────────────────────
cm = confusion_matrix(true_labels, pred_labels)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title(f"Task 2 — Pre-trained ResNet-18 Confusion Matrix\n"
             f"Validation accuracy: {val_acc_final:.2%}  (LR={best_lr})",
             fontsize=13)
plt.xticks(rotation=45, ha="right", fontsize=9)  # ← rotates x-axis labels
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task2_pretrainedcnn_cm.png", dpi=150)
plt.close()
print(f"  Confusion matrix saved to: {OUTPUT_DIR / 'task2_pretrainedcnn_cm.png'}")

# ── Classification report CSV  →  task2_pretrainedcnn_classification_report.csv
report_dict = classification_report(
    true_labels, pred_labels,
    target_names=classes, digits=4, output_dict=True,
)
report_df = pd.DataFrame(report_dict).transpose().reset_index()
report_df.rename(columns={"index": "class"}, inplace=True)
report_df.to_csv(OUTPUT_DIR / "task2_pretrainedcnn_classification_report.csv", index=False)
print(f"  Classification report saved to: {OUTPUT_DIR / 'task2_pretrainedcnn_classification_report.csv'}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. t-SNE VISUALISATION  →  task2_pretrainedcnn_tsne.png
# ─────────────────────────────────────────────────────────────────────────────
# Extract 512-dim backbone features from ResNet-18's avgpool layer,
# then project to 2D with t-SNE.
#
# CHANGED: perplexity 30 → 15
# Task 2 val set is much smaller (~70-80 images vs ~750 in Task 1).
# Lower perplexity works better with fewer points — prevents t-SNE
# from creating artificially separated clusters on sparse data.

print("\n" + "=" * 60)
print("Task 2 — t-SNE Feature Visualisation")
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

# CHANGED: perplexity 30 → 15 for small validation set
tsne     = make_tsne(n_components=2, perplexity=15, random_state=RANDOM_STATE,
                     max_iter=1000)
feats_2d = tsne.fit_transform(all_feats)

palette = plt.cm.get_cmap("tab10", NUM_CLASSES)

fig, ax = plt.subplots(figsize=(10, 8))

for class_id, class_name in sorted(id_to_class.items()):
    mask = all_labels_tsne == class_id
    ax.scatter(
        feats_2d[mask, 0], feats_2d[mask, 1],
        label=class_name,
        color=palette(class_id),
        s=40, alpha=0.75, edgecolors="none",
    )

ax.set_title(
    f"Task 2 — t-SNE of Pre-trained ResNet-18 Features\n"
    f"(validation set, 512-dim → 2-D,  val acc={val_acc_final:.2%})",
    fontsize=13,
)
ax.set_xlabel("t-SNE dimension 1")
ax.set_ylabel("t-SNE dimension 2")
ax.legend(loc="best", fontsize=9, markerscale=1.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task2_pretrainedcnn_tsne.png", dpi=150)
plt.close()
print(f"  t-SNE plot saved to: {OUTPUT_DIR / 'task2_pretrainedcnn_tsne.png'}")

# ─────────────────────────────────────────────────────────────────────────────
# 13. TEST-SET PREDICTION  →  task2_resnet18_submission.csv
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Task 2 — Generating test predictions for Kaggle submission")
print("=" * 60)

if not os.path.exists(TEST_META):
    print("  test_metadata.csv not found — skipping.")
else:
    test_df = pd.read_csv(TEST_META)

    test_loader = DataLoader(
        BirdDataset(test_df, IMAGE_ROOT, val_test_transforms, has_labels=False),
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
