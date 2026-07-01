# Rebuilt from Task2 (2).ipynb: Task 2 Random Forest.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    RandomForestFeatureConfig,
    build_random_forest_test_features,
    build_random_forest_training_features,
)

DATA_DIR = Path("task2_data")
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

RF_CONFIG = RandomForestFeatureConfig(
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

train_df, features, feature_cols = build_random_forest_training_features(RF_CONFIG)

RANDOM_STATE = 42
VAL_SIZE = 0.2

X = features[feature_cols].values
y = features["class_name"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y,
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. BASELINE RANDOM FOREST WITH CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

baseline_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf",     RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=1,
        n_jobs=-1, random_state=RANDOM_STATE,
    )),
])

# CHANGED: n_splits 5 → 10
# With ~300-340 training samples after the split, 5-fold CV gives only
# ~240-272 training images per fold (~24-27 per class) — dangerously thin.
# 10-fold CV gives ~270-306 training images per fold (~27-30 per class),
# better utilising the limited data. More folds also give a more reliable
# CV estimate by averaging over more validation subsets.
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

cv_scores = cross_val_score(
    baseline_pipeline, X_train, y_train,
    cv=cv, scoring="accuracy", n_jobs=-1
)
print(f"\nBaseline CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING (GridSearchCV)
# ─────────────────────────────────────────────────────────────────────────────

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf",     RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE)),
])

# CHANGED: param_grid adjusted for small dataset
# n_estimators: added 700 — more trees help more with small noisy datasets
# max_depth: removed 40, added 10 — shallower trees regularise better when
#   only ~27-30 images per class per fold; deep trees will overfit easily
# min_samples_leaf: removed 1, added 7 — higher minimum leaf sizes enforce
#   stronger regularisation, preventing trees from fitting single samples
# max_features: unchanged — both still worth evaluating
param_grid = {
    "rf__n_estimators"    : [300, 500, 700],      # CHANGED: added 700, removed 100
    "rf__max_depth"       : [None, 10, 20],        # CHANGED: 40→10 (more regularisation)
    "rf__min_samples_leaf": [3, 5, 7],             # CHANGED: removed 1, added 7
    "rf__max_features"    : ["sqrt", "log2"],
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=cv,
    scoring="accuracy", n_jobs=-1, verbose=1, refit=True,
)
grid_search.fit(X_train, y_train)

print(f"\n  Best parameters : {grid_search.best_params_}")
print(f"  Best CV accuracy: {grid_search.best_score_:.4f}")
best_model = grid_search.best_estimator_

# ─────────────────────────────────────────────────────────────────────────────
# 6a. FINAL EVALUATION ON HELD-OUT VALIDATION SET
# ─────────────────────────────────────────────────────────────────────────────

y_pred = best_model.predict(X_val)

val_accuracy = accuracy_score(y_val, y_pred)
print(f"\n  Validation accuracy : {val_accuracy:.4f}  ({val_accuracy*100:.1f}%)")
print("\n  Per-class classification report:")
print(classification_report(y_val, y_pred, digits=4))

final_model = clone(grid_search.best_estimator_)
final_model.fit(X, y)
print(f"\n  Final RF retrained on all {len(y)} Task 2 training samples.")

# ─────────────────────────────────────────────────────────────────────────────
# 6b. MODEL COMPARISON: Baseline RF vs Tuned RF
# ─────────────────────────────────────────────────────────────────────────────

baseline_mean = cv_scores.mean()
baseline_std  = cv_scores.std()
tuned_cv_mean = grid_search.best_score_
best_idx      = grid_search.best_index_
tuned_cv_std  = grid_search.cv_results_["std_test_score"][best_idx]

labels = ["Baseline RF\n(CV)", "Tuned RF\n(CV)", "Tuned RF\n(Val)"]
means  = [baseline_mean, tuned_cv_mean, val_accuracy]
stds   = [baseline_std,  tuned_cv_std,  0]
colors = ["#4C72B0", "#DD8452", "#55A868"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors, width=0.5)
for bar, mean, std in zip(bars, means, stds):
    label = f"{mean:.4f}" if std == 0 else f"{mean:.4f} ± {std:.4f}"
    ax.text(bar.get_x() + bar.get_width() / 2,
            mean + std + 0.005, label,
            ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Accuracy")
ax.set_title("Task 2 — Baseline RF vs Tuned RF")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task2_rf_model_comparison.png", dpi=150)
plt.close()
print(f"  Model comparison chart saved to: {OUTPUT_DIR / 'task2_rf_model_comparison.png'}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. CONFUSION MATRIX  →  task2_rf_confusion_matrix.png
# ─────────────────────────────────────────────────────────────────────────────

classes = sorted(np.unique(y))
cm = confusion_matrix(y_val, y_pred, labels=classes)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title(f"Task 2 — Random Forest Confusion Matrix\n"
             f"Validation accuracy: {val_accuracy:.2%}", fontsize=13)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task2_rf_confusion_matrix.png", dpi=150)
plt.close()
print(f"  Confusion matrix saved to: {OUTPUT_DIR / 'task2_rf_confusion_matrix.png'}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. CV ACCURACY BAR CHART  →  task2_rf_cv_accuracy.csv  +  .png
# ─────────────────────────────────────────────────────────────────────────────

cv_results_df = pd.DataFrame(grid_search.cv_results_)
param_cols    = [c for c in cv_results_df.columns if c.startswith("param_")]
cv_summary    = cv_results_df[param_cols + ["mean_test_score", "std_test_score"]].copy()
cv_summary.columns = (
    [c.replace("param_rf__", "") for c in param_cols]
    + ["mean_accuracy", "std_accuracy"]
)
cv_summary = cv_summary.sort_values("mean_accuracy", ascending=False).reset_index(drop=True)
cv_summary.to_csv(OUTPUT_DIR / "task2_rf_cv_accuracy.csv", index=False)
print(f"  CV accuracy table saved to: {OUTPUT_DIR / 'task2_rf_cv_accuracy.csv'}")

top_n = min(20, len(cv_summary))
top   = cv_summary.head(top_n)

param_label_cols = [c for c in cv_summary.columns
                    if c not in ("mean_accuracy", "std_accuracy")]
labels_chart = [
    "\n".join(f"{col}={row[col]}" for col in param_label_cols)
    for _, row in top.iterrows()
]

fig, ax = plt.subplots(figsize=(max(12, top_n * 0.9), 6))
x = np.arange(top_n)
ax.bar(x, top["mean_accuracy"], yerr=top["std_accuracy"],
       capsize=4, color="#4C72B0", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels_chart, fontsize=7, rotation=45, ha="right")
ax.set_ylabel("Mean CV Accuracy (10-fold)")
ax.set_title("Task 2 — Random Forest GridSearchCV: Top Hyperparameter Combinations")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_ylim(0, 1)
for xi, (mean, std) in enumerate(zip(top["mean_accuracy"], top["std_accuracy"])):
    ax.text(xi, mean + std + 0.005, f"{mean:.3f}", ha="center", fontsize=7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task2_rf_cv_accuracy.png", dpi=150)
plt.close()
print(f"  CV accuracy bar chart saved to: {OUTPUT_DIR / 'task2_rf_cv_accuracy.png'}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. FEATURE IMPORTANCE PLOT  →  task2_rf_best_features.png
# ─────────────────────────────────────────────────────────────────────────────

importances = final_model.named_steps["rf"].feature_importances_

def get_category(col_name):
    if col_name.startswith("color_"):
        return "Color Histogram"
    elif col_name.startswith("hog_pca_"):
        return "HOG (PCA)"
    elif col_name.startswith("feat_"):
        return "Additional Features"
    elif col_name.startswith("texture_feature_"):
        idx = int(col_name.split("_")[-1])
        # CHANGED: offsets updated to match new texture_vec layout:
        # multiscale_lbp(54) + haralick(13) + gabor(48) + hsv(48) + fine_color(96)
        if idx < 54:
            return "LBP (Multiscale)"
        elif idx < 67:    # 54 + 13
            return "Haralick Texture"
        elif idx < 115:   # 67 + 48  CHANGED: was 91 (24 gabor) now 115 (48 gabor)
            return "Gabor Filters"
        elif idx < 163:   # 115 + 48
            return "HSV Histogram"
        else:             # 163 + 96
            return "Fine Color Histogram"    # CHANGED: new category
    elif col_name.startswith("keypoint_feature_"):
        idx = int(col_name.split("_")[-1])
        if idx < 64:
            return "ORB"
        else:
            return "SIFT"
    elif col_name == "colour_entropy":
        return "Colour Entropy"
    elif col_name == "hog_energy":
        return "HOG Energy"
    elif col_name == "hog_contrast":   # CHANGED: new engineered feature
        return "HOG Contrast"
    else:
        return "Other"

category_importance = {}
for col, imp in zip(feature_cols, importances):
    cat = get_category(col)
    category_importance.setdefault(cat, []).append(imp)

category_means = {cat: np.mean(vals) for cat, vals in category_importance.items()}
sorted_cats    = sorted(category_means.items(), key=lambda x: x[1], reverse=True)
top5_cats      = sorted_cats[:5]
top5_names     = [c[0] for c in top5_cats]
top5_vals      = [c[1] for c in top5_cats]

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
ax.barh(top5_names[::-1], top5_vals[::-1], color=colors[::-1])
ax.set_xlabel("Mean Feature Importance (mean decrease in impurity)")
ax.set_title("Task 2 — Random Forest: Top 5 Most Important Feature Categories")
for i, (name, val) in enumerate(zip(top5_names[::-1], top5_vals[::-1])):
    ax.text(val + 0.0001, i, f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task2_rf_best_features.png", dpi=150)
plt.close()
print(f"  Feature importance plot saved to: {OUTPUT_DIR / 'task2_rf_best_features.png'}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. CLASSIFICATION REPORT  →  task2_rf_classification_report.csv
# ─────────────────────────────────────────────────────────────────────────────

report_dict = classification_report(y_val, y_pred, digits=4, output_dict=True)
report_df   = pd.DataFrame(report_dict).transpose().reset_index()
report_df.rename(columns={"index": "class"}, inplace=True)
report_df.to_csv(OUTPUT_DIR / "task2_rf_classification_report.csv", index=False)
print(f"  Classification report saved to: {OUTPUT_DIR / 'task2_rf_classification_report.csv'}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. TEST PREDICTIONS  →  task2_rf_submission.csv
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("11. Generating test predictions for Kaggle submission")
print("=" * 60)

test_df, test_features = build_random_forest_test_features(RF_CONFIG)

# Predict
X_test = test_features[feature_cols].values
pred_class_names = final_model.predict(X_test)

# Map class_name to class_id
name_to_id = (
    train_df[["class_name", "class_id"]]
    .drop_duplicates()
    .set_index("class_name")["class_id"]
    .to_dict()
)

submission = pd.DataFrame({
    "image_id": test_features["image_id"].values,
    "class_id": [name_to_id[n] for n in pred_class_names],
})
submission.to_csv(OUTPUT_DIR / "task2_rf_submission.csv", index=False)

print(f"  Saved to: {OUTPUT_DIR / 'task2_rf_submission.csv'}")
print(f"  Total predictions : {len(submission)}")
print(f"  Preview:\n{submission.head()}")
