from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data_preprocessing import (
    RandomForestFeatureConfig,
    build_random_forest_test_features,
    build_random_forest_training_features,
)


TASK_CONFIGS = {
    "task1": RandomForestFeatureConfig(
        data_dir=Path("task1_data"),
        image_size=(64, 64),
        texture_output_csv="texture_features.csv",
        keypoint_output_csv="keypoint_features.csv",
        test_texture_output_csv="test_texture_features.csv",
        test_keypoint_output_csv="test_keypoint_features.csv",
    ),
    "task2": RandomForestFeatureConfig(
        data_dir=Path("task2_data"),
        image_size=(128, 128),
        texture_output_csv="task2_texture_features.csv",
        keypoint_output_csv="task2_keypoint_features.csv",
        test_texture_output_csv="task2_test_texture_features.csv",
        test_keypoint_output_csv="task2_test_keypoint_features.csv",
        include_fine_color=True,
        include_hog_contrast=True,
        gabor_frequencies=(0.1, 0.25, 0.4, 0.6),
        gabor_thetas=(0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6),
    ),
}


def precompute(task: str) -> None:
    config = TASK_CONFIGS[task]
    print(f"\n{task}: training features")
    build_random_forest_training_features(config)
    print(f"\n{task}: test features")
    build_random_forest_test_features(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute cached image-derived features for Random Forest models."
    )
    parser.add_argument(
        "task",
        nargs="?",
        choices=["task1", "task2", "all"],
        default="all",
        help="Which task cache to build. Defaults to all.",
    )
    args = parser.parse_args()

    tasks = TASK_CONFIGS if args.task == "all" else [args.task]
    for task in tasks:
        precompute(task)


if __name__ == "__main__":
    main()
