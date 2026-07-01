"""kNN model configuration for Task 1."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier

from classical_model_runner import run_tabular_model
from data_preprocessing import RandomForestFeatureConfig, load_full_feature_frames


def get_model_config(random_state: int = 42) -> dict[str, object]:
    """Return the kNN estimator and hyperparameter grid."""
    return {
        "model": KNeighborsClassifier(n_jobs=1),
        "params": {
            "classifier__n_neighbors": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        },
    }


def _load_frames(data_dir: Path):
    return load_full_feature_frames(
        RandomForestFeatureConfig(
            data_dir=data_dir,
            image_size=(64, 64),
            texture_output_csv="texture_features.csv",
            keypoint_output_csv="keypoint_features.csv",
            test_texture_output_csv="test_texture_features.csv",
            test_keypoint_output_csv="test_keypoint_features.csv",
        )
    )


def run(
    *,
    data_dir: Path = Path("task1_data"),
    output_dir: Path = Path("plots"),
    val_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, object]:
    return run_tabular_model(
        data_dir=data_dir,
        output_dir=output_dir,
        file_prefix="task1_knn",
        display_name="Task 1 kNN",
        get_model_config=get_model_config,
        feature_loader=_load_frames,
        val_size=val_size,
        cv_folds=cv_folds,
        random_state=random_state,
    )


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
