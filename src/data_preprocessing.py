from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import mahotas
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from skimage.filters import gabor

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - lets non-CNN scripts import this module.
    class Dataset:  # type: ignore[no-redef]
        pass


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class RandomForestFeatureConfig:
    data_dir: str | Path
    image_size: tuple[int, int]
    texture_output_csv: str
    keypoint_output_csv: str
    test_texture_output_csv: str | None = None
    test_keypoint_output_csv: str | None = None
    include_fine_color: bool = False
    include_hog_contrast: bool = False
    gabor_frequencies: tuple[float, ...] = (0.1, 0.25, 0.4)
    gabor_thetas: tuple[float, ...] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


class MetadataImageDataset(Dataset):
    def __init__(self, metadata_df, image_root, transform, has_labels=True):
        self.df = metadata_df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image

        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["image_path"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if self.has_labels:
            return image, int(row["class_id"])
        return image, row["image_id"]


def make_tsne(*, n_components: int, perplexity: float, random_state: int, max_iter: int):
    """Build a TSNE object across scikit-learn versions."""
    from sklearn.manifold import TSNE

    kwargs = {
        "n_components": n_components,
        "perplexity": perplexity,
        "random_state": random_state,
    }
    try:
        return TSNE(**kwargs, max_iter=max_iter)
    except TypeError as exc:
        if "max_iter" not in str(exc):
            raise
        return TSNE(**kwargs, n_iter=max_iter)


def task1_pretrained_transforms():
    import torchvision.transforms as transforms

    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transforms, val_test_transforms


def task1_scratch_transforms():
    import torchvision.transforms as transforms

    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transforms, val_test_transforms


def task2_pretrained_transforms():
    import torchvision.transforms as transforms

    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transforms, val_test_transforms


def extract_haralick_features(gray_image):
    haralick = mahotas.features.haralick(gray_image)
    return haralick.mean(axis=0)


def extract_orb_features(gray_image, n_features=50):
    orb = cv2.ORB_create(nfeatures=n_features)
    _, descriptors = orb.detectAndCompute(gray_image, None)
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(64)
    d = descriptors.astype(np.float32)
    return np.concatenate([d.mean(axis=0), d.std(axis=0)])


def extract_sift_features(gray_image, n_features=50):
    sift = cv2.SIFT_create(nfeatures=n_features)
    _, descriptors = sift.detectAndCompute(gray_image, None)
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(256)
    return np.concatenate([descriptors.mean(axis=0), descriptors.std(axis=0)])


def extract_multiscale_lbp(gray_image):
    features = []
    for radius, points in [(1, 8), (2, 16), (3, 24)]:
        lbp = local_binary_pattern(gray_image, P=points, R=radius, method="uniform")
        n_bins = points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                               range=(0, n_bins), density=True)
        features.append(hist)
    return np.concatenate(features)


def extract_hsv_histogram(bgr_image, bins=16):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    features = []
    for ch, (low, high) in enumerate([(0, 180), (0, 256), (0, 256)]):
        hist = cv2.calcHist([hsv], [ch], None, [bins], [low, high])
        hist = hist.flatten() / (hist.sum() + 1e-6)
        features.append(hist)
    return np.concatenate(features)


def extract_gabor_features(gray_image, frequencies=None, thetas=None):
    if frequencies is None:
        frequencies = (0.1, 0.25, 0.4)
    if thetas is None:
        thetas = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)

    features = []
    img_float = gray_image.astype(np.float32) / 255.0
    for freq in frequencies:
        for theta in thetas:
            real, imag = gabor(img_float, frequency=freq, theta=theta)
            magnitude = np.sqrt(real**2 + imag**2)
            features.extend([magnitude.mean(), magnitude.std()])
    return np.array(features)


def extract_fine_color_histogram(bgr_image, bins=32):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    features = []
    for ch, (low, high) in enumerate([(0, 180), (0, 256), (0, 256)]):
        hist = cv2.calcHist([hsv], [ch], None, [bins], [low, high])
        hist = hist.flatten() / (hist.sum() + 1e-6)
        features.append(hist)
    return np.concatenate(features)


def _texture_vector(image, gray, config: RandomForestFeatureConfig):
    feature_parts = [
        extract_multiscale_lbp(gray),
        extract_haralick_features(gray),
        extract_gabor_features(gray, config.gabor_frequencies, config.gabor_thetas),
        extract_hsv_histogram(image),
    ]
    if config.include_fine_color:
        feature_parts.append(extract_fine_color_histogram(image))
    return np.concatenate(feature_parts)


def extract_image_features(image_folder, split_prefix, config: RandomForestFeatureConfig):
    texture_rows = []
    keypoint_rows = []

    for class_folder in os.listdir(image_folder):
        class_path = os.path.join(image_folder, class_folder)

        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(class_path, filename)
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not load: {image_path}")
                    continue
                image = cv2.resize(image, config.image_size)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                image_id = f"train_{os.path.splitext(filename)[0]}"

                texture_vec = _texture_vector(image, gray, config)
                t_row = {"image_id": image_id}
                for i, v in enumerate(texture_vec):
                    t_row[f"texture_feature_{i}"] = v
                texture_rows.append(t_row)

                keypoint_vec = np.concatenate([
                    extract_orb_features(gray),
                    extract_sift_features(gray),
                ])
                k_row = {"image_id": image_id}
                for i, v in enumerate(keypoint_vec):
                    k_row[f"keypoint_feature_{i}"] = v
                keypoint_rows.append(k_row)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    return pd.DataFrame(texture_rows), pd.DataFrame(keypoint_rows)


def extract_image_features_test(image_folder, config: RandomForestFeatureConfig):
    texture_rows = []
    keypoint_rows = []

    for filename in os.listdir(image_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(image_folder, filename)
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load: {image_path}")
                continue
            image = cv2.resize(image, config.image_size)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            image_id = os.path.splitext(filename)[0]

            texture_vec = _texture_vector(image, gray, config)
            t_row = {"image_id": image_id}
            for i, v in enumerate(texture_vec):
                t_row[f"texture_feature_{i}"] = v
            texture_rows.append(t_row)

            keypoint_vec = np.concatenate([
                extract_orb_features(gray),
                extract_sift_features(gray),
            ])
            k_row = {"image_id": image_id}
            for i, v in enumerate(keypoint_vec):
                k_row[f"keypoint_feature_{i}"] = v
            keypoint_rows.append(k_row)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return pd.DataFrame(texture_rows), pd.DataFrame(keypoint_rows)


def _data_path(config: RandomForestFeatureConfig, *parts):
    return Path(config.data_dir).joinpath(*parts)


def _output_path(config: RandomForestFeatureConfig, filename):
    path = Path(filename)
    if path.is_absolute():
        return path
    return _data_path(config, filename)


def _cached_or_extract_image_features(
    *,
    image_folder: Path,
    split_prefix: str,
    texture_csv: Path,
    keypoint_csv: Path,
    config: RandomForestFeatureConfig,
):
    if texture_csv.exists() and keypoint_csv.exists():
        print(f"Loading cached {split_prefix} image features...")
        texture_df = pd.read_csv(texture_csv)
        keypoint_df = pd.read_csv(keypoint_csv)
    else:
        print(f"Extracting {split_prefix} image features...")
        if split_prefix == "training":
            texture_df, keypoint_df = extract_image_features(image_folder, "train", config)
        else:
            texture_df, keypoint_df = extract_image_features_test(image_folder, config)
        texture_csv.parent.mkdir(parents=True, exist_ok=True)
        keypoint_csv.parent.mkdir(parents=True, exist_ok=True)
        texture_df.to_csv(texture_csv, index=False)
        keypoint_df.to_csv(keypoint_csv, index=False)
    return texture_df, keypoint_df


def add_random_forest_engineered_features(features, include_hog_contrast=False):
    color_cols = [c for c in features.columns if c.startswith("color_")]
    hog_cols = [c for c in features.columns if c.startswith("hog_pca_")]

    eps = 1e-10
    color_vals = features[color_cols].values
    p = np.clip(color_vals, eps, None)
    p = p / p.sum(axis=1, keepdims=True)
    colour_entropy = -np.sum(p * np.log(p), axis=1)

    hog_energy = (features[hog_cols].values ** 2).sum(axis=1)

    engineered = {
        "colour_entropy": colour_entropy,
        "hog_energy": hog_energy,
    }
    if include_hog_contrast:
        engineered["hog_contrast"] = features[hog_cols].values.std(axis=1)

    return pd.concat(
        [features, pd.DataFrame(engineered, index=features.index)],
        axis=1,
    )


def random_forest_feature_columns(features, include_hog_contrast=False):
    color_cols = [c for c in features.columns if c.startswith("color_")]
    hog_cols = [c for c in features.columns if c.startswith("hog_pca_")]
    add_cols = [c for c in features.columns if c.startswith("feat_")]
    texture_cols = [c for c in features.columns if c.startswith("texture_feature_")]
    keypoint_cols = [c for c in features.columns if c.startswith("keypoint_feature_")]

    engineered_cols = ["colour_entropy", "hog_energy"]
    if include_hog_contrast:
        engineered_cols.append("hog_contrast")

    return color_cols + hog_cols + add_cols + texture_cols + keypoint_cols + engineered_cols


def build_random_forest_training_features(config: RandomForestFeatureConfig):
    train_df = pd.read_csv(_data_path(config, "train_metadata.csv"))
    color_hist = pd.read_csv(_data_path(config, "color_histogram.csv"))
    hog_pca = pd.read_csv(_data_path(config, "hog_pca.csv"))
    add_feat = pd.read_csv(_data_path(config, "additional_features.csv"))

    texture_df, keypoint_df = _cached_or_extract_image_features(
        image_folder=_data_path(config, "images", "train"),
        split_prefix="training",
        texture_csv=_output_path(config, config.texture_output_csv),
        keypoint_csv=_output_path(config, config.keypoint_output_csv),
        config=config,
    )
    print(f"  Texture features  : {texture_df.shape}")
    print(f"  Keypoint features : {keypoint_df.shape}")

    features = (
        train_df[["image_id", "class_name", "class_id"]]
        .merge(color_hist, on="image_id", how="inner")
        .merge(hog_pca, on="image_id", how="inner")
        .merge(add_feat, on="image_id", how="inner")
        .merge(texture_df, on="image_id", how="inner")
        .merge(keypoint_df, on="image_id", how="inner")
    )
    features = add_random_forest_engineered_features(
        features, include_hog_contrast=config.include_hog_contrast
    )
    feature_cols = random_forest_feature_columns(
        features, include_hog_contrast=config.include_hog_contrast
    )

    return train_df, features, feature_cols


def build_random_forest_test_features(config: RandomForestFeatureConfig):
    test_df = pd.read_csv(_data_path(config, "test_metadata.csv"))
    test_color_hist = pd.read_csv(_data_path(config, "color_histogram.csv"))
    test_hog_pca = pd.read_csv(_data_path(config, "hog_pca.csv"))
    test_add_feat = pd.read_csv(_data_path(config, "additional_features.csv"))

    test_texture_output_csv = config.test_texture_output_csv or f"test_{config.texture_output_csv}"
    test_keypoint_output_csv = config.test_keypoint_output_csv or f"test_{config.keypoint_output_csv}"
    test_texture_df, test_keypoint_df = _cached_or_extract_image_features(
        image_folder=_data_path(config, "images", "test"),
        split_prefix="test",
        texture_csv=_output_path(config, test_texture_output_csv),
        keypoint_csv=_output_path(config, test_keypoint_output_csv),
        config=config,
    )
    print(f"  Test texture features  : {test_texture_df.shape}")
    print(f"  Test keypoint features : {test_keypoint_df.shape}")

    test_features = (
        test_df[["image_id"]]
        .merge(test_color_hist, on="image_id", how="inner")
        .merge(test_hog_pca, on="image_id", how="inner")
        .merge(test_add_feat, on="image_id", how="inner")
        .merge(test_texture_df, on="image_id", how="inner")
        .merge(test_keypoint_df, on="image_id", how="inner")
    )
    test_features = add_random_forest_engineered_features(
        test_features, include_hog_contrast=config.include_hog_contrast
    )

    return test_df, test_features


def load_full_feature_frames(config: RandomForestFeatureConfig):
    train_df, train_frame, feature_cols = build_random_forest_training_features(config)
    _, test_frame = build_random_forest_test_features(config)

    mapping_path = Path(config.data_dir) / "class_mapping.csv"
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        class_id_to_name = (
            mapping.sort_values("class_id")
            .set_index("class_id")["class_name"]
            .to_dict()
        )
    else:
        class_id_to_name = (
            train_df[["class_id", "class_name"]]
            .drop_duplicates()
            .sort_values("class_id")
            .set_index("class_id")["class_name"]
            .to_dict()
        )

    return train_frame, test_frame, feature_cols, class_id_to_name


def load_classical_feature_frames(data_dir: str | Path):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train_metadata.csv")
    test = pd.read_csv(data_dir / "test_metadata.csv")
    features = pd.read_csv(data_dir / "hog_pca.csv")
    for filename in ("color_histogram.csv", "additional_features.csv"):
        features = features.merge(pd.read_csv(data_dir / filename), on="image_id", how="inner")

    color_cols = [column for column in features.columns if column.startswith("color_")]
    values = np.clip(features[color_cols].to_numpy(dtype=np.float64), 0.0, None)
    totals = values.sum(axis=1, keepdims=True)
    probabilities = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
    log_probabilities = np.zeros_like(probabilities)
    np.log(probabilities, out=log_probabilities, where=probabilities > 0)
    features["colour_entropy"] = -(probabilities * log_probabilities).sum(axis=1)

    train_frame = train.merge(features, on="image_id", how="inner", validate="one_to_one")
    test_frame = test.merge(features, on="image_id", how="inner", validate="one_to_one")
    feature_cols = [column for column in features.columns if column != "image_id"]

    mapping_path = data_dir / "class_mapping.csv"
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        class_id_to_name = (
            mapping.sort_values("class_id")
            .set_index("class_id")["class_name"]
            .to_dict()
        )
    else:
        class_id_to_name = (
            train[["class_id", "class_name"]]
            .drop_duplicates()
            .sort_values("class_id")
            .set_index("class_id")["class_name"]
            .to_dict()
        )

    return train_frame, test_frame, feature_cols, class_id_to_name
