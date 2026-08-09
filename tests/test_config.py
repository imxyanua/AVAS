from pathlib import Path

import pytest
import yaml

from src.utils.config import ConfigError, DatasetConfig, SamplingConfig, SplitConfig, load_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def base_payload() -> dict:
    return {
        "name": "unit_test_dataset",
        "root": "data/raw/unit_test",
        "processed_root": "data/processed/unit_test",
        "video_extensions": [".avi", ".MP4"],
        "group_pattern": r"_(g\d+)_",
        "classes": ["walking", "falling"],
        "behavior_map": {"normal": ["walking"], "abnormal": ["falling"]},
        "split": {"train": 0.7, "validation": 0.15, "test": 0.15, "seed": 7},
        "sampling": {
            "clip_length": 8,
            "frame_stride": 2,
            "strategy": "center",
            "image_size": 112,
        },
    }


def test_repository_dataset_config_is_valid():
    config = DatasetConfig.from_yaml(REPO_ROOT / "configs" / "dataset.yaml")

    assert config.classes
    assert config.split.as_ratios()["train"] > 0
    assert set(config.normal_classes) | set(config.abnormal_classes) == set(config.classes)


def test_from_mapping_normalises_paths_and_extensions():
    config = DatasetConfig.from_mapping(base_payload())

    assert config.root == Path("data/raw/unit_test")
    assert config.processed_root == Path("data/processed/unit_test")
    assert config.video_extensions == (".avi", ".mp4")
    assert config.class_to_index == {"walking": 0, "falling": 1}
    assert config.is_abnormal("falling") is True
    assert config.is_abnormal("walking") is False


def test_is_abnormal_rejects_unknown_class():
    config = DatasetConfig.from_mapping(base_payload())

    with pytest.raises(KeyError):
        config.is_abnormal("dancing")


def test_split_ratios_must_sum_to_one():
    with pytest.raises(ConfigError, match=r"must sum to 1\.0"):
        SplitConfig(train=0.7, validation=0.2, test=0.2)


def test_split_ratio_must_be_a_fraction():
    with pytest.raises(ConfigError, match="between 0 and 1"):
        SplitConfig(train=1.0, validation=0.15, test=0.15)


@pytest.mark.parametrize(
    ("field", "value"),
    [("clip_length", 0), ("frame_stride", 0), ("image_size", 0)],
)
def test_sampling_rejects_non_positive_values(field, value):
    with pytest.raises(ConfigError, match=field):
        SamplingConfig(**{field: value})


def test_sampling_rejects_unknown_strategy():
    with pytest.raises(ConfigError, match="strategy"):
        SamplingConfig(strategy="stochastic")


def test_behavior_map_must_cover_every_class():
    payload = base_payload()
    payload["behavior_map"] = {"normal": ["walking"], "abnormal": []}

    with pytest.raises(ConfigError, match="missing classes"):
        DatasetConfig.from_mapping(payload)


def test_behavior_map_rejects_unknown_class():
    payload = base_payload()
    payload["behavior_map"]["abnormal"] = ["falling", "arson"]

    with pytest.raises(ConfigError, match="unknown classes"):
        DatasetConfig.from_mapping(payload)


def test_behavior_map_rejects_class_in_both_groups():
    payload = base_payload()
    payload["behavior_map"] = {
        "normal": ["walking", "falling"],
        "abnormal": ["falling"],
    }

    with pytest.raises(ConfigError, match="normal and abnormal"):
        DatasetConfig.from_mapping(payload)


def test_duplicate_classes_are_rejected():
    payload = base_payload()
    payload["classes"] = ["walking", "walking", "falling"]

    with pytest.raises(ConfigError, match="duplicates"):
        DatasetConfig.from_mapping(payload)


def test_group_pattern_requires_capturing_group():
    payload = base_payload()
    payload["group_pattern"] = r"_g\d+_"

    with pytest.raises(ConfigError, match="capturing group"):
        DatasetConfig.from_mapping(payload)


def test_invalid_group_pattern_is_reported():
    payload = base_payload()
    payload["group_pattern"] = "_(g\\d+_"

    with pytest.raises(ConfigError, match="valid regex"):
        DatasetConfig.from_mapping(payload)


def test_missing_required_keys_are_listed():
    payload = base_payload()
    del payload["classes"]
    del payload["split"]

    with pytest.raises(ConfigError, match="missing required keys"):
        DatasetConfig.from_mapping(payload)


def test_extensions_must_start_with_dot():
    payload = base_payload()
    payload["video_extensions"] = ["avi"]

    with pytest.raises(ConfigError, match="must start with"):
        DatasetConfig.from_mapping(payload)


def test_load_yaml_reports_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_yaml(tmp_path / "absent.yaml")


def test_load_yaml_reports_empty_file(tmp_path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError, match="empty"):
        load_yaml(config_path)


def test_from_yaml_round_trip(tmp_path):
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(base_payload()), encoding="utf-8")

    config = DatasetConfig.from_yaml(config_path)

    assert config.name == "unit_test_dataset"
    assert config.sampling.strategy == "center"
    assert config.split.seed == 7
