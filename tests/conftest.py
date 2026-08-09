import csv
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

WIDTH = 64
HEIGHT = 48
FPS = 10.0


def frame_colour(index: int) -> tuple[int, int, int]:
    """Distinct RGB colour per frame index, robust to lossy compression."""
    return (20 * index, 255 - 20 * index, 10 * index)


def write_video(path: Path, frame_count: int) -> Path:
    """Write a short MJPG AVI whose frames differ by a flat colour."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV build cannot write MJPG AVI files")

    for index in range(frame_count):
        red, green, blue = frame_colour(index)
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :] = (blue, green, red)
        writer.write(frame)
    writer.release()

    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("OpenCV did not produce a readable video file")
    return path


@pytest.fixture
def make_video() -> Callable[[Path, int], Path]:
    return write_video


FEATURE_CLASSES = ("walking", "falling")
FEATURE_DIM = 8
CLIP_LENGTH = 4
INDEX_COLUMNS = (
    "split",
    "label",
    "label_index",
    "group",
    "video",
    "clip",
    "frame_indices",
    "feature_path",
)
SPLIT_SIZES = {"train": 10, "validation": 4, "test": 4}


def clip_features(label_index: int, seed: int) -> np.ndarray:
    """Features that are linearly separable per class, so a tiny model can learn them."""
    rng = np.random.default_rng(seed)
    features = rng.normal(scale=0.1, size=(CLIP_LENGTH, FEATURE_DIM)).astype(np.float32)
    half = FEATURE_DIM // 2
    if label_index == 0:
        features[:, :half] += 2.0
    else:
        features[:, half:] += 2.0
    return features


@pytest.fixture
def feature_cache(tmp_path) -> dict[str, object]:
    """A synthetic feature cache with configuration files, ready for training."""
    cache_root = tmp_path / "features"
    rows: list[dict[str, object]] = []
    seed = 0

    for split, recordings in SPLIT_SIZES.items():
        split_dir = cache_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for label_index, label in enumerate(FEATURE_CLASSES):
            for recording in range(recordings):
                seed += 1
                group = f"g{split[:2]}{recording:02d}"
                name = f"{label}__{group}__clip00.npy"
                feature_path = split_dir / name
                np.save(feature_path, clip_features(label_index, seed))
                rows.append(
                    {
                        "split": split,
                        "label": label,
                        "label_index": label_index,
                        "group": group,
                        "video": f"{label}/v_{label}_{group}_c01.avi",
                        "clip": 0,
                        "frame_indices": "0 1 2 3",
                        "feature_path": feature_path.as_posix(),
                    }
                )

    index_path = cache_root / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    dataset_config = tmp_path / "dataset.yaml"
    dataset_config.write_text(
        yaml.safe_dump(
            {
                "name": "feature_cache_test",
                "root": str(tmp_path / "raw"),
                "processed_root": str(tmp_path / "processed"),
                "classes": list(FEATURE_CLASSES),
                "behavior_map": {"normal": ["walking"], "abnormal": ["falling"]},
                "split": {"train": 0.6, "validation": 0.2, "test": 0.2, "seed": 42},
                "sampling": {"clip_length": CLIP_LENGTH, "frame_stride": 1, "image_size": 32},
            }
        ),
        encoding="utf-8",
    )

    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        yaml.safe_dump(
            {
                "architecture": "cnn_lstm",
                "backbone": {"name": "resnet18", "pretrained": False, "freeze": True},
                "temporal": {
                    "hidden_size": 8,
                    "num_layers": 1,
                    "bidirectional": False,
                    "dropout": 0.0,
                },
                "classifier": {"dropout": 0.0},
                "anomaly": {"suspicious_threshold": 0.5, "high_risk_threshold": 0.8},
            }
        ),
        encoding="utf-8",
    )

    training_config = tmp_path / "training.yaml"
    training_config.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "device": "cpu",
                "epochs": 4,
                "batch_size": 4,
                "num_workers": 0,
                "learning_rate": 0.01,
                "class_weighting": "balanced",
                "early_stopping": {"monitor": "val_macro_f1", "patience": 5},
                "feature_cache_root": str(cache_root),
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "log_dir": str(tmp_path / "logs"),
            }
        ),
        encoding="utf-8",
    )

    return {
        "cache_root": cache_root,
        "index": index_path,
        "rows": rows,
        "classes": FEATURE_CLASSES,
        "feature_dim": FEATURE_DIM,
        "dataset_config": dataset_config,
        "model_config": model_config,
        "training_config": training_config,
        "checkpoint_dir": tmp_path / "checkpoints",
        "log_dir": tmp_path / "logs",
        "split_sizes": {split: size * len(FEATURE_CLASSES) for split, size in SPLIT_SIZES.items()},
    }


@pytest.fixture
def video_dataset(tmp_path) -> Path:
    """Two classes, four recordings each, two clips per recording."""
    root = tmp_path / "raw"
    for label in ("walking", "falling"):
        for group in (1, 2, 3, 4):
            for clip in (1, 2):
                name = f"v_{label}_g{group:02d}_c{clip:02d}.avi"
                write_video(root / label / name, frame_count=12)
    return root
