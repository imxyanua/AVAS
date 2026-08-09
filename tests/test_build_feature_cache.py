import csv

import numpy as np
import pytest
import yaml

from src.features.build_feature_cache import feature_filename, main
from src.preprocessing.splits import discover_videos, split_videos, write_manifest
from src.utils.config import DatasetConfig

CLASSES = ("walking", "falling")
GROUP_PATTERN = r"_(g\d+)_"


@pytest.fixture
def prepared_dataset(tmp_path, video_dataset):
    """A split manifest plus the config files needed by the cache builder."""
    dataset_payload = {
        "name": "cache_test",
        "root": str(video_dataset),
        "processed_root": str(tmp_path / "processed"),
        "group_pattern": GROUP_PATTERN,
        "classes": list(CLASSES),
        "behavior_map": {"normal": ["walking"], "abnormal": ["falling"]},
        "split": {"train": 0.5, "validation": 0.25, "test": 0.25, "seed": 42},
        "sampling": {
            "clip_length": 2,
            "frame_stride": 2,
            "strategy": "uniform",
            "image_size": 32,
        },
    }
    dataset_config_path = tmp_path / "dataset.yaml"
    dataset_config_path.write_text(yaml.safe_dump(dataset_payload), encoding="utf-8")

    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text(
        yaml.safe_dump(
            {
                "architecture": "cnn_lstm",
                "backbone": {"name": "resnet18", "pretrained": False, "freeze": True},
            }
        ),
        encoding="utf-8",
    )

    training_config_path = tmp_path / "training.yaml"
    training_config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "device": "cpu",
                "num_workers": 0,
                "feature_cache_root": str(tmp_path / "features"),
            }
        ),
        encoding="utf-8",
    )

    config = DatasetConfig.from_yaml(dataset_config_path)
    samples = discover_videos(video_dataset, CLASSES, group_pattern=GROUP_PATTERN)
    splits = split_videos(samples, config.split.as_ratios(), seed=config.split.seed)
    manifest = write_manifest(splits, config.processed_root / "splits.csv", relative_to=config.root)

    return {
        "dataset_config": dataset_config_path,
        "model_config": model_config_path,
        "training_config": training_config_path,
        "manifest": manifest,
        "cache_root": tmp_path / "features",
        "sample_count": len(samples),
    }


def cli_args(prepared, *extra):
    return [
        "--dataset-config",
        str(prepared["dataset_config"]),
        "--model-config",
        str(prepared["model_config"]),
        "--training-config",
        str(prepared["training_config"]),
        "--manifest",
        str(prepared["manifest"]),
        "--device",
        "cpu",
        "--batch-size",
        "2",
        *extra,
    ]


def read_index(cache_root):
    with (cache_root / "index.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_cache_builder_writes_one_feature_file_per_clip(prepared_dataset):
    exit_code = main(cli_args(prepared_dataset))

    cache_root = prepared_dataset["cache_root"]
    rows = read_index(cache_root)

    assert exit_code == 0
    assert len(rows) == prepared_dataset["sample_count"]
    for row in rows:
        features = np.load(row["feature_path"])
        assert features.shape == (2, 512)
        assert features.dtype == np.float32


def test_cached_features_are_grouped_by_split(prepared_dataset):
    main(cli_args(prepared_dataset))

    rows = read_index(prepared_dataset["cache_root"])

    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    for row in rows:
        assert f"/{row['split']}/" in row["feature_path"]


def test_index_records_provenance(prepared_dataset):
    main(cli_args(prepared_dataset))

    rows = read_index(prepared_dataset["cache_root"])

    for row in rows:
        assert row["label"] in CLASSES
        assert row["label_index"] in {"0", "1"}
        assert row["group"].startswith("g")
        assert len(row["frame_indices"].split()) == 2


def test_existing_features_are_reused_unless_overwritten(prepared_dataset):
    main(cli_args(prepared_dataset))
    rows = read_index(prepared_dataset["cache_root"])
    target = rows[0]["feature_path"]
    np.save(target, np.zeros((2, 512), dtype=np.float32))

    main(cli_args(prepared_dataset))
    assert not np.load(target).any()

    main(cli_args(prepared_dataset, "--overwrite"))
    assert np.load(target).any()


def test_only_requested_splits_are_processed(prepared_dataset):
    main(cli_args(prepared_dataset, "--splits", "train"))

    rows = read_index(prepared_dataset["cache_root"])

    assert {row["split"] for row in rows} == {"train"}


def test_unknown_split_is_reported(prepared_dataset):
    with pytest.raises(SystemExit, match="no splits named"):
        main(cli_args(prepared_dataset, "--splits", "holdout"))


def test_multiple_clips_require_random_sampling(prepared_dataset):
    with pytest.raises(SystemExit, match="random"):
        main(cli_args(prepared_dataset, "--clips-per-video", "2"))


def test_clips_per_video_must_be_positive(prepared_dataset):
    with pytest.raises(SystemExit, match="clips-per-video"):
        main(cli_args(prepared_dataset, "--clips-per-video", "0"))


def test_unfrozen_backbone_is_rejected(prepared_dataset, tmp_path):
    model_config = tmp_path / "unfrozen.yaml"
    model_config.write_text(
        yaml.safe_dump({"backbone": {"name": "resnet18", "pretrained": False, "freeze": False}}),
        encoding="utf-8",
    )
    args = cli_args(prepared_dataset)
    args[args.index("--model-config") + 1] = str(model_config)

    with pytest.raises(SystemExit, match="freeze"):
        main(args)


def test_feature_filename_is_unique_per_class_and_clip(tmp_path):
    root = tmp_path / "raw"
    walking = root / "walking" / "v_clip_g01_c01.avi"
    falling = root / "falling" / "v_clip_g01_c01.avi"

    assert feature_filename(walking, root, 0) != feature_filename(falling, root, 0)
    assert feature_filename(walking, root, 0) != feature_filename(walking, root, 1)
    assert feature_filename(walking, root, 3).endswith("__clip03.npy")


def test_feature_filename_falls_back_to_the_file_name(tmp_path):
    outside = tmp_path / "elsewhere" / "clip.avi"

    assert feature_filename(outside, tmp_path / "raw", 0) == "clip__clip00.npy"
