from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


PLOTS_DIR = Path("plots")
OUTPUT_DIR = Path("plots_allmodels")


TASK1_MODELS = [
    ("kNN", "task1_knn_classification_report.csv"),
    ("SVM", "task1_svm_classification_report.csv"),
    ("Random Forest", "task1_rf_classification_report.csv"),
    ("Scratch CNN", "task1_scratchcnn_classification_report.csv"),
    ("Pretrained CNN", "task1_pretrainedcnn_classification_report.csv"),
]

TASK2_MODELS = [
    ("SVM", "task2_svm_classification_report.csv"),
    ("Random Forest", "task2_rf_classification_report.csv"),
    ("Pretrained CNN", "task2_pretrainedcnn_classification_report.csv"),
    ("Ensemble", "task2_ensemble_classification_report.csv"),
]


def extract_accuracy(csv_path: Path) -> float:
    """Read a classification report CSV and return its overall accuracy."""
    report = pd.read_csv(csv_path)

    if "accuracy" in report.columns:
        values = report["accuracy"].dropna()
        if not values.empty:
            return float(values.iloc[0])

    if "class" in report.columns:
        accuracy_rows = report[
            report["class"].astype(str).str.strip().str.lower() == "accuracy"
        ]
        if not accuracy_rows.empty:
            row = accuracy_rows.iloc[0]
            for column in ("precision", "recall", "f1-score", "f1_score"):
                if column in report.columns and pd.notna(row[column]):
                    return float(row[column])

    raise ValueError(f"Could not find an accuracy value in {csv_path}")


def collect_accuracies(model_specs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for model_name, filename in model_specs:
        path = PLOTS_DIR / filename
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        rows.append(
            {
                "model": model_name,
                "accuracy": extract_accuracy(path),
                "source_csv": str(path),
            }
        )
    return pd.DataFrame(rows)


def plot_accuracy_comparison(table: pd.DataFrame, title: str, output_path: Path) -> None:
    if table.empty:
        raise ValueError(f"No model accuracies available for {title}")

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        table["model"],
        table["accuracy"],
        color=colors[: len(table)],
        width=0.62,
    )

    for bar, value in zip(bars, table["accuracy"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, min(1.0, max(table["accuracy"]) + 0.12))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    task1_table = collect_accuracies(TASK1_MODELS)
    task2_table = collect_accuracies(TASK2_MODELS)

    task1_output = OUTPUT_DIR / "task1_all_model_accuracies.png"
    task2_output = OUTPUT_DIR / "task2_all_model_accuracies.png"

    plot_accuracy_comparison(
        task1_table,
        "Task 1 Model Accuracy Comparison",
        task1_output,
    )
    plot_accuracy_comparison(
        task2_table,
        "Task 2 Model Accuracy Comparison",
        task2_output,
    )

    print("Saved model accuracy plots:")
    print(f"  {task1_output}")
    print(f"  {task2_output}")


if __name__ == "__main__":
    main()
