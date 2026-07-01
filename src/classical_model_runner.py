from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_preprocessing import load_classical_feature_frames


ModelConfigFn = Callable[[int], dict[str, object]]
FeatureLoaderFn = Callable[[Path], tuple[pd.DataFrame, pd.DataFrame, list[str], dict[int, str]]]


def _report_frame(report: dict[str, Any], accuracy: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "class": label,
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1-score"),
                "support": metrics.get("support"),
                "accuracy": accuracy,
            }
            for label, metrics in report.items()
            if isinstance(metrics, dict)
        ]
    )


def _feature_category(col_name: str, has_fine_color: bool) -> str:
    if col_name.startswith("color_"):
        return "Color Histogram"
    if col_name.startswith("hog_pca_"):
        return "HOG (PCA)"
    if col_name.startswith("feat_"):
        return "Additional Features"
    if col_name.startswith("texture_feature_"):
        idx = int(col_name.split("_")[-1])
        if idx < 54:
            return "LBP (Multiscale)"
        if idx < 67:
            return "Haralick Texture"
        if has_fine_color:
            if idx < 115:
                return "Gabor Filters"
            if idx < 163:
                return "HSV Histogram"
            return "Fine Color Histogram"
        if idx < 91:
            return "Gabor Filters"
        return "HSV Histogram"
    if col_name.startswith("keypoint_feature_"):
        idx = int(col_name.split("_")[-1])
        if idx < 64:
            return "ORB"
        return "SIFT"
    if col_name == "colour_entropy":
        return "Colour Entropy"
    if col_name == "hog_energy":
        return "HOG Energy"
    if col_name == "hog_contrast":
        return "HOG Contrast"
    return "Other"


def _save_category_validation_importance(
    *,
    output_dir: Path,
    file_prefix: str,
    display_name: str,
    search: GridSearchCV,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    feature_cols: list[str],
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    category_indices: dict[str, list[int]] = {}
    texture_count = sum(col_name.startswith("texture_feature_") for col_name in feature_cols)
    has_fine_color = texture_count > 139
    for index, col_name in enumerate(feature_cols):
        category_indices.setdefault(_feature_category(col_name, has_fine_color), []).append(index)

    rows: list[dict[str, object]] = []
    for category, indices in category_indices.items():
        estimator = clone(search.best_estimator_)
        scores = cross_val_score(
            estimator,
            X_train[:, indices],
            y_train,
            cv=search.cv,
            scoring="accuracy",
            n_jobs=1,
        )
        rows.append(
            {
                "feature_category": category,
                "mean_validation_accuracy": scores.mean(),
                "std_validation_accuracy": scores.std(),
                "feature_count": len(indices),
            }
        )

    table = (
        pd.DataFrame(rows)
        .sort_values("mean_validation_accuracy", ascending=False)
        .reset_index(drop=True)
    )
    csv_path = output_dir / f"{file_prefix}_feature_category_accuracy.csv"
    table.to_csv(csv_path, index=False)

    top5 = table.head(5)
    png_path = output_dir / f"{file_prefix}_top5_feature_categories.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        top5["feature_category"].iloc[::-1],
        top5["mean_validation_accuracy"].iloc[::-1],
        color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"][:len(top5)][::-1],
    )
    ax.set_xlabel("Mean Validation Accuracy")
    ax.set_title(f"{display_name}: Top 5 Feature Categories")
    for i, value in enumerate(top5["mean_validation_accuracy"].iloc[::-1]):
        ax.text(value + 0.002, i, f"{value:.4f}", va="center", fontsize=9)
    score_min = float(top5["mean_validation_accuracy"].min())
    score_max = float(top5["mean_validation_accuracy"].max())
    padding = max((score_max - score_min) * 0.2, 0.02)
    ax.set_xlim(max(0, score_min - padding), min(1, score_max + padding))
    plt.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return {"feature_category_accuracy_csv": str(csv_path), "top5_feature_categories": str(png_path)}


def _save_plots(
    output_dir: Path,
    file_prefix: str,
    display_name: str,
    search: GridSearchCV,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    valid_pred: np.ndarray,
    feature_cols: list[str],
    class_names: list[str],
    random_state: int,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    path = output_dir / f"{file_prefix}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_valid, valid_pred),
        display_labels=class_names,
    ).plot(ax=ax, cmap="Blues")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["confusion_matrix"] = str(path)

    results = pd.DataFrame(search.cv_results_)
    if {"param_classifier__C", "param_classifier__gamma"}.issubset(results.columns):
        pivot = results.pivot_table(
            index="param_classifier__C",
            columns="param_classifier__gamma",
            values="mean_test_score",
            aggfunc="mean",
        )
        path_csv = output_dir / f"{file_prefix}_hyperparameter_heatmap.csv"
        pivot.to_csv(path_csv)
        path = output_dir / f"{file_prefix}_hyperparameter_heatmap.png"
        fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1]), max(5, pivot.shape[0] * 0.35)))
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels([str(column) for column in pivot.columns], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels([str(index) for index in pivot.index], fontsize=8)
        ax.set_xlabel("gamma")
        ax.set_ylabel("C")
        ax.set_title(f"{display_name} hyperparameter tuning heatmap")
        fig.colorbar(image, ax=ax, label="Mean CV accuracy")
        plt.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["hyperparameter_heatmap"] = str(path)
    else:
        param_cols = [column for column in results.columns if column.startswith("param_")]
        summary = results[param_cols + ["mean_test_score", "std_test_score"]].copy()
        path_csv = output_dir / f"{file_prefix}_cv_results.csv"
        if "param_classifier__n_neighbors" in summary.columns:
            summary = summary.sort_values(
                ["param_classifier__weights", "param_classifier__p", "param_classifier__n_neighbors"]
            ).reset_index(drop=True)
            summary.to_csv(path_csv, index=False)

            path = output_dir / f"{file_prefix}_cv_results.png"
            fig, ax = plt.subplots(figsize=(9, 5.5))
            group_cols = [
                column
                for column in ("param_classifier__weights", "param_classifier__p")
                if column in summary.columns
            ]
            for group_values, group in summary.groupby(group_cols, dropna=False):
                if not isinstance(group_values, tuple):
                    group_values = (group_values,)
                label = ", ".join(
                    f"{column.replace('param_classifier__', '')}={value}"
                    for column, value in zip(group_cols, group_values)
                )
                ax.plot(
                    group["param_classifier__n_neighbors"].astype(int),
                    group["mean_test_score"],
                    marker="o",
                    linewidth=1.8,
                    label=label,
                )

            score_min = float(summary["mean_test_score"].min())
            score_max = float(summary["mean_test_score"].max())
            padding = max((score_max - score_min) * 0.15, 0.01)
            ax.set_ylim(max(0, score_min - padding), min(1, score_max + padding))
            ax.set_xticks(sorted(summary["param_classifier__n_neighbors"].astype(int).unique()))
            ax.set_xlabel("n_neighbors")
            ax.set_ylabel("Mean CV Accuracy")
            ax.set_title(f"{display_name} hyperparameter tuning results")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths["cv_results"] = str(path)
        else:
            summary = summary.sort_values("mean_test_score", ascending=False).reset_index(drop=True)
            summary.to_csv(path_csv, index=False)
            top_n = min(20, len(summary))
            top = summary.head(top_n)
            labels = [
                "\n".join(f"{column.replace('param_classifier__', '')}={row[column]}" for column in param_cols)
                for _, row in top.iterrows()
            ]
            path = output_dir / f"{file_prefix}_cv_results.png"
            fig, ax = plt.subplots(figsize=(max(10, top_n * 0.8), 6))
            x = np.arange(top_n)
            ax.bar(x, top["mean_test_score"], yerr=top["std_test_score"], capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Mean CV Accuracy")
            ax.set_title(f"{display_name} hyperparameter tuning results")
            ax.set_ylim(0, 1)
            plt.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths["cv_results"] = str(path)

    paths.update(
        _save_category_validation_importance(
            output_dir=output_dir,
            file_prefix=file_prefix,
            display_name=display_name,
            search=search,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            feature_cols=feature_cols,
        )
    )
    return paths


def run_tabular_model(
    *,
    data_dir: Path,
    output_dir: Path,
    file_prefix: str,
    display_name: str,
    get_model_config: ModelConfigFn,
    feature_loader: FeatureLoaderFn = load_classical_feature_frames,
    val_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame, test_frame, feature_cols, class_id_to_name = feature_loader(data_dir)
    X = train_frame[feature_cols].to_numpy(dtype=np.float32)
    y = train_frame["class_id"].to_numpy(dtype=np.int64)
    X_test = test_frame[feature_cols].to_numpy(dtype=np.float32)
    train_idx, valid_idx = train_test_split(
        np.arange(len(train_frame)),
        test_size=max(val_size, len(class_id_to_name) / len(train_frame)),
        stratify=y,
        random_state=random_state,
    )
    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    effective_cv = min(cv_folds, int(pd.Series(y_train).value_counts().min()))
    config = get_model_config(random_state=random_state)
    search = GridSearchCV(
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", config["model"]),
            ]
        ),
        config["params"],
        cv=StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=random_state),
        scoring="accuracy",
        n_jobs=1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    valid_pred = search.predict(X_valid)
    accuracy = accuracy_score(y_valid, valid_pred)
    class_labels = sorted(class_id_to_name)
    class_names = [class_id_to_name[class_id] for class_id in class_labels]
    report = classification_report(
        y_valid,
        valid_pred,
        labels=class_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_valid,
        valid_pred,
        labels=class_labels,
        target_names=class_names,
        zero_division=0,
    )
    report_path = output_dir / f"{file_prefix}_validation_report.txt"
    report_path.write_text(
        "\n".join(
            [
                f"Best CV accuracy: {search.best_score_:.6f}",
                f"Validation accuracy: {accuracy:.6f}",
                f"Best params: {json.dumps(search.best_params_, sort_keys=True)}",
                f"Final retrain samples: {len(y)}",
                "Final predictions use the best estimator refit on 100% of training data.",
                "",
                report_text,
            ]
        )
    )
    report_csv = output_dir / f"{file_prefix}_classification_report.csv"
    _report_frame(report, accuracy).to_csv(report_csv, index=False)
    plot_paths = _save_plots(
        output_dir,
        file_prefix,
        display_name,
        search,
        X_train,
        y_train,
        X_valid,
        y_valid,
        valid_pred,
        feature_cols,
        class_names,
        random_state,
    )
    best_model = clone(search.best_estimator_)
    best_model.fit(X, y)
    print(f"{display_name}: final model retrained on all {len(y)} training samples.")
    pred = best_model.predict(X_test)
    prediction = pd.DataFrame({"image_id": test_frame["image_id"], "class_id": pred.astype(int)})
    prediction["class_name"] = prediction["class_id"].map(class_id_to_name)
    prediction_path = output_dir / f"{file_prefix}_predictions.csv"
    submission_path = output_dir / f"{file_prefix}_submission.csv"
    prediction.to_csv(prediction_path, index=False)
    prediction[["image_id", "class_id"]].to_csv(submission_path, index=False)
    return {
        "best_cv_accuracy": float(search.best_score_),
        "validation_accuracy": float(accuracy),
        "best_params": search.best_params_,
        "files": {
            "validation_report": str(report_path),
            "classification_report_csv": str(report_csv),
            "predictions": str(prediction_path),
            "submission": str(submission_path),
            "plots": plot_paths,
        },
    }
