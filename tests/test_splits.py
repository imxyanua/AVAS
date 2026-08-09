from pathlib import Path

import pytest

from src.preprocessing.splits import (
    DatasetLayoutError,
    SplitError,
    SplitLeakageError,
    VideoSample,
    assert_no_group_leakage,
    discover_from_config,
    discover_videos,
    read_manifest,
    split_from_config,
    split_videos,
    summarize_splits,
    write_manifest,
)
from src.utils.config import DatasetConfig

GROUP_PATTERN = r"_(g\d+)_"
CLASSES = ("walking", "falling")
RATIOS = {"train": 0.7, "validation": 0.15, "test": 0.15}


def build_dataset(root, *, classes=CLASSES, groups_per_class=10, clips_per_group=4, suffix=".avi"):
    for label in classes:
        class_dir = root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for group in range(1, groups_per_class + 1):
            for clip in range(1, clips_per_group + 1):
                name = f"v_{label}_g{group:02d}_c{clip:02d}{suffix}"
                (class_dir / name).touch()
    return root


def dataset_config(root, **overrides):
    payload = {
        "name": "split_test",
        "root": str(root),
        "group_pattern": GROUP_PATTERN,
        "classes": list(CLASSES),
        "behavior_map": {"normal": ["walking"], "abnormal": ["falling"]},
        "split": {"train": 0.7, "validation": 0.15, "test": 0.15, "seed": 42},
    }
    payload.update(overrides)
    return DatasetConfig.from_mapping(payload)


def test_discover_videos_reads_labels_and_groups(tmp_path):
    build_dataset(tmp_path, groups_per_class=3, clips_per_group=2)

    samples = discover_videos(tmp_path, CLASSES, group_pattern=GROUP_PATTERN)

    assert len(samples) == 12
    assert {sample.label for sample in samples} == set(CLASSES)
    assert {sample.group for sample in samples} == {"g01", "g02", "g03"}
    assert samples[0].group_key.startswith("walking/")


def test_discover_videos_ignores_unrelated_files(tmp_path):
    build_dataset(tmp_path, groups_per_class=2, clips_per_group=1)
    (tmp_path / "walking" / "notes.txt").touch()
    (tmp_path / "walking" / "thumbnail.png").touch()

    samples = discover_videos(tmp_path, CLASSES, group_pattern=GROUP_PATTERN)

    assert all(sample.path.suffix == ".avi" for sample in samples)
    assert len(samples) == 4


def test_discover_videos_matches_extensions_case_insensitively(tmp_path):
    build_dataset(tmp_path, groups_per_class=2, clips_per_group=1, suffix=".MP4")

    samples = discover_videos(tmp_path, CLASSES, extensions=(".mp4",))

    assert len(samples) == 4


def test_without_group_pattern_each_file_is_its_own_group(tmp_path):
    build_dataset(tmp_path, groups_per_class=2, clips_per_group=3)

    samples = discover_videos(tmp_path, CLASSES)

    assert len({sample.group for sample in samples}) == len(samples)


def test_unmatched_group_pattern_falls_back_to_file_stem(tmp_path):
    class_dir = tmp_path / "walking"
    class_dir.mkdir(parents=True)
    (class_dir / "clip_without_group.avi").touch()
    (tmp_path / "falling").mkdir()
    (tmp_path / "falling" / "v_falling_g01_c01.avi").touch()

    samples = discover_videos(tmp_path, CLASSES, group_pattern=GROUP_PATTERN)

    groups = {sample.label: sample.group for sample in samples}
    assert groups["walking"] == "clip_without_group"
    assert groups["falling"] == "g01"


def test_discover_videos_requires_every_class_directory(tmp_path):
    build_dataset(tmp_path, classes=("walking",), groups_per_class=2)

    with pytest.raises(DatasetLayoutError, match="missing directory for class 'falling'"):
        discover_videos(tmp_path, CLASSES, group_pattern=GROUP_PATTERN)


def test_discover_videos_requires_a_non_empty_class(tmp_path):
    build_dataset(tmp_path, groups_per_class=1, clips_per_group=1)
    for path in (tmp_path / "falling").iterdir():
        path.unlink()

    with pytest.raises(DatasetLayoutError, match="no videos"):
        discover_videos(tmp_path, CLASSES, group_pattern=GROUP_PATTERN)


def test_discover_videos_requires_an_existing_root(tmp_path):
    with pytest.raises(DatasetLayoutError, match="dataset root not found"):
        discover_videos(tmp_path / "absent", CLASSES)


def test_split_keeps_every_recording_in_a_single_split(tmp_path):
    build_dataset(tmp_path, groups_per_class=12, clips_per_group=5)
    samples = discover_videos(tmp_path, CLASSES, group_pattern=GROUP_PATTERN)

    splits = split_videos(samples, RATIOS, seed=42)

    assert_no_group_leakage(splits)
    owners = {}
    for split_name, split_samples in splits.items():
        for sample in split_samples:
            owners.setdefault(sample.group_key, split_name)
            assert owners[sample.group_key] == split_name


def test_split_preserves_all_samples_without_duplication(tmp_path):
    build_dataset(tmp_path, groups_per_class=12, clips_per_group=3)
    samples = discover_videos(tmp_path, CLASSES, group_pattern=GROUP_PATTERN)

    splits = split_videos(samples, RATIOS, seed=42)

    recovered = [sample for split_samples in splits.values() for sample in split_samples]
    assert len(recovered) == len(samples)
    assert {sample.path for sample in recovered} == {sample.path for sample in samples}


def test_split_is_deterministic_for_a_given_seed(tmp_path):
    build_dataset(tmp_path, groups_per_class=12, clips_per_group=4)
    samples = discover_videos(tmp_path, CLASSES, group_pattern=GROUP_PATTERN)

    first = split_videos(samples, RATIOS, seed=42)
    second = split_videos(samples, RATIOS, seed=42)

    assert first == second


def test_split_changes_with_the_seed(tmp_path):
    build_dataset(tmp_path, groups_per_class=20, clips_per_group=2)
    samples = discover_videos(tmp_path, CLASSES, group_pattern=GROUP_PATTERN)

    first = split_videos(samples, RATIOS, seed=1)
    second = split_videos(samples, RATIOS, seed=2)

    assert first["test"] != second["test"]


def test_split_approximates_the_requested_ratios_per_class(tmp_path):
    build_dataset(tmp_path, groups_per_class=40, clips_per_group=2)
    samples = discover_videos(tmp_path, CLASSES, group_pattern=GROUP_PATTERN)

    splits = split_videos(samples, RATIOS, seed=42)

    per_class_total = 80
    for split_name, ratio in RATIOS.items():
        for label in CLASSES:
            count = sum(1 for sample in splits[split_name] if sample.label == label)
            assert abs(count / per_class_total - ratio) < 0.05


def test_split_normalises_ratios_that_do_not_sum_to_one(tmp_path):
    build_dataset(tmp_path, groups_per_class=10, clips_per_group=2)
    samples = discover_videos(tmp_path, CLASSES, group_pattern=GROUP_PATTERN)

    splits = split_videos(samples, {"train": 8, "test": 2}, seed=5)

    assert len(splits["train"]) > len(splits["test"])
    assert len(splits["train"]) + len(splits["test"]) == len(samples)


def test_split_rejects_empty_input():
    with pytest.raises(SplitError, match="no samples"):
        split_videos([], RATIOS)


def test_split_rejects_non_positive_ratio(tmp_path):
    build_dataset(tmp_path, groups_per_class=4, clips_per_group=1)
    samples = discover_videos(tmp_path, CLASSES, group_pattern=GROUP_PATTERN)

    with pytest.raises(SplitError, match="must be > 0"):
        split_videos(samples, {"train": 1.0, "test": 0.0})


def test_split_fails_when_a_split_would_be_empty(tmp_path):
    build_dataset(tmp_path, groups_per_class=1, clips_per_group=1)
    samples = discover_videos(tmp_path, CLASSES, group_pattern=GROUP_PATTERN)

    with pytest.raises(SplitError, match="received no videos"):
        split_videos(samples, RATIOS, seed=42)


def test_split_warns_when_a_class_has_too_few_recordings(tmp_path, caplog):
    build_dataset(tmp_path, classes=("walking",), groups_per_class=2, clips_per_group=2)
    (tmp_path / "falling").mkdir()
    for group in range(1, 9):
        (tmp_path / "falling" / f"v_falling_g{group:02d}_c01.avi").touch()
    samples = discover_videos(tmp_path, CLASSES, group_pattern=GROUP_PATTERN)

    with caplog.at_level("WARNING"):
        split_videos(samples, RATIOS, seed=42)

    assert "walking" in caplog.text


def test_leakage_detection_reports_the_shared_recording():
    shared = VideoSample(path=Path("a.avi"), label="walking", group="g01")
    other = VideoSample(path=Path("b.avi"), label="walking", group="g01")

    with pytest.raises(SplitLeakageError, match="walking/g01"):
        assert_no_group_leakage({"train": [shared], "test": [other]})


def test_summarize_splits_counts_videos_per_class(tmp_path):
    build_dataset(tmp_path, groups_per_class=10, clips_per_group=2)
    samples = discover_videos(tmp_path, CLASSES, group_pattern=GROUP_PATTERN)

    summary = summarize_splits(split_videos(samples, RATIOS, seed=42))

    assert sum(counts["total"] for counts in summary.values()) == len(samples)
    for counts in summary.values():
        assert counts["total"] == sum(counts[label] for label in CLASSES if label in counts)


def test_manifest_round_trip_preserves_the_assignment(tmp_path):
    build_dataset(tmp_path, groups_per_class=10, clips_per_group=2)
    samples = discover_videos(tmp_path, CLASSES, group_pattern=GROUP_PATTERN)
    splits = split_videos(samples, RATIOS, seed=42)

    manifest = write_manifest(splits, tmp_path / "manifests" / "splits.csv", relative_to=tmp_path)
    restored = read_manifest(manifest, relative_to=tmp_path)

    assert manifest.is_file()
    assert restored == splits
    assert_no_group_leakage(restored)


def test_manifest_stores_relative_paths(tmp_path):
    build_dataset(tmp_path, groups_per_class=4, clips_per_group=1)
    samples = discover_videos(tmp_path, CLASSES, group_pattern=GROUP_PATTERN)
    splits = split_videos(samples, {"train": 0.75, "test": 0.25}, seed=3)

    manifest = write_manifest(splits, tmp_path / "splits.csv", relative_to=tmp_path)

    content = manifest.read_text(encoding="utf-8")
    assert str(tmp_path) not in content
    assert "walking/" in content


def test_read_manifest_rejects_unexpected_columns(tmp_path):
    manifest = tmp_path / "broken.csv"
    manifest.write_text("split,label\ntrain,walking\n", encoding="utf-8")

    with pytest.raises(SplitError, match="manifest columns"):
        read_manifest(manifest)


def test_read_manifest_reports_missing_file(tmp_path):
    with pytest.raises(SplitError, match="manifest not found"):
        read_manifest(tmp_path / "absent.csv")


def test_config_helpers_use_configured_paths_and_seed(tmp_path):
    build_dataset(tmp_path, groups_per_class=10, clips_per_group=2)
    config = dataset_config(tmp_path)

    samples = discover_from_config(config)
    splits = split_from_config(samples, config)

    assert len(samples) == 40
    assert splits == split_videos(samples, config.split.as_ratios(), seed=config.split.seed)
    assert_no_group_leakage(splits)
