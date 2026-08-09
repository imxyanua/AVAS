import csv

import numpy as np
import pytest

from src.features.cache_index import (
    FeatureCacheError,
    assert_no_group_leakage,
    build_dataset,
    clips_for_split,
    feature_dimension,
    read_feature_index,
    validate_against_config,
)
from src.utils.config import DatasetConfig


def rewrite_index(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def test_read_feature_index_returns_every_clip(feature_cache):
    clips = read_feature_index(feature_cache["index"])

    assert len(clips) == len(feature_cache["rows"])
    assert {clip.split for clip in clips} == {"train", "validation", "test"}
    assert clips[0].feature_path.is_file()
    assert clips[0].group_key.startswith(clips[0].label)


def test_clips_for_split_selects_one_split(feature_cache):
    clips = read_feature_index(feature_cache["index"])

    train = clips_for_split(clips, "train")

    assert len(train) == feature_cache["split_sizes"]["train"]
    assert {clip.split for clip in train} == {"train"}


def test_clips_for_split_reports_available_splits(feature_cache):
    clips = read_feature_index(feature_cache["index"])

    with pytest.raises(FeatureCacheError, match="available: \\['test', 'train', 'validation'\\]"):
        clips_for_split(clips, "holdout")


def test_cache_without_leakage_passes_the_check(feature_cache):
    clips = read_feature_index(feature_cache["index"])

    assert_no_group_leakage(clips)


def test_recording_in_two_splits_is_detected(feature_cache):
    rows = list(feature_cache["rows"])
    duplicated = dict(rows[0])
    duplicated["split"] = "test"
    rewrite_index(feature_cache["index"], [*rows, duplicated], rows[0].keys())

    clips = read_feature_index(feature_cache["index"])

    with pytest.raises(FeatureCacheError, match="shared between splits"):
        assert_no_group_leakage(clips)


def test_index_missing_columns_is_reported(tmp_path):
    index_path = tmp_path / "index.csv"
    index_path.write_text("split,label\ntrain,walking\n", encoding="utf-8")

    with pytest.raises(FeatureCacheError, match="missing columns"):
        read_feature_index(index_path)


def test_empty_index_is_reported(feature_cache):
    rewrite_index(feature_cache["index"], [], feature_cache["rows"][0].keys())

    with pytest.raises(FeatureCacheError, match="empty"):
        read_feature_index(feature_cache["index"])


def test_missing_index_is_reported(tmp_path):
    with pytest.raises(FeatureCacheError, match="not found"):
        read_feature_index(tmp_path / "absent.csv")


def test_validate_against_config_accepts_a_matching_cache(feature_cache):
    config = DatasetConfig.from_yaml(feature_cache["dataset_config"])
    clips = read_feature_index(feature_cache["index"])

    validate_against_config(clips, config)


def test_validate_against_config_rejects_unknown_labels(feature_cache):
    rows = [dict(row) for row in feature_cache["rows"]]
    rows[0]["label"] = "arson"
    rewrite_index(feature_cache["index"], rows, rows[0].keys())
    config = DatasetConfig.from_yaml(feature_cache["dataset_config"])

    with pytest.raises(FeatureCacheError, match="not in the configured classes"):
        validate_against_config(read_feature_index(feature_cache["index"]), config)


def test_validate_against_config_rejects_shifted_label_indices(feature_cache):
    rows = [dict(row) for row in feature_cache["rows"]]
    rows[0]["label_index"] = 1 - int(rows[0]["label_index"])
    rewrite_index(feature_cache["index"], rows, rows[0].keys())
    config = DatasetConfig.from_yaml(feature_cache["dataset_config"])

    with pytest.raises(FeatureCacheError, match="config expects"):
        validate_against_config(read_feature_index(feature_cache["index"]), config)


def test_feature_dimension_is_read_from_the_cache(feature_cache):
    clips = read_feature_index(feature_cache["index"])

    assert feature_dimension(clips) == feature_cache["feature_dim"]


def test_feature_dimension_reports_a_missing_file(feature_cache):
    clips = read_feature_index(feature_cache["index"])
    clips[0].feature_path.unlink()

    with pytest.raises(FeatureCacheError, match="not found"):
        feature_dimension(clips)


def test_feature_dimension_rejects_arrays_without_a_time_axis(feature_cache):
    clips = read_feature_index(feature_cache["index"])
    np.save(clips[0].feature_path, np.zeros(8, dtype=np.float32))

    with pytest.raises(FeatureCacheError, match=r"shape \(T, D\)"):
        feature_dimension(clips)


def test_build_dataset_serves_features_and_labels(feature_cache):
    clips = clips_for_split(read_feature_index(feature_cache["index"]), "train")

    dataset = build_dataset(clips, expected_dim=feature_cache["feature_dim"])
    features, label = dataset[0]

    assert len(dataset) == len(clips)
    assert features.shape == (4, feature_cache["feature_dim"])
    assert label in {0, 1}


def test_build_dataset_reports_missing_feature_files(feature_cache):
    clips = clips_for_split(read_feature_index(feature_cache["index"]), "train")
    clips[2].feature_path.unlink()

    with pytest.raises(FeatureCacheError, match="cached feature files are missing"):
        build_dataset(clips)
