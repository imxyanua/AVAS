import json

import yaml

from src.preprocessing.build_splits import main
from src.preprocessing.splits import assert_no_group_leakage, read_manifest

CLASSES = ("walking", "falling")


def build_dataset(root, *, groups_per_class=8, clips_per_group=3):
    for label in CLASSES:
        class_dir = root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for group in range(1, groups_per_class + 1):
            for clip in range(1, clips_per_group + 1):
                (class_dir / f"v_{label}_g{group:02d}_c{clip:02d}.avi").touch()


def write_config(tmp_path, dataset_root):
    config_path = tmp_path / "dataset.yaml"
    payload = {
        "name": "cli_test",
        "root": str(dataset_root),
        "processed_root": str(tmp_path / "processed"),
        "group_pattern": r"_(g\d+)_",
        "classes": list(CLASSES),
        "behavior_map": {"normal": ["walking"], "abnormal": ["falling"]},
        "split": {"train": 0.7, "validation": 0.15, "test": 0.15, "seed": 42},
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_cli_writes_a_leak_free_manifest(tmp_path):
    dataset_root = tmp_path / "raw"
    build_dataset(dataset_root)
    config_path = write_config(tmp_path, dataset_root)

    exit_code = main(["--config", str(config_path)])

    manifest = tmp_path / "processed" / "splits.csv"
    assert exit_code == 0
    assert manifest.is_file()

    splits = read_manifest(manifest, relative_to=dataset_root)
    assert_no_group_leakage(splits)
    assert sum(len(samples) for samples in splits.values()) == 48
    assert set(splits) == {"train", "validation", "test"}


def test_cli_honours_custom_output_and_writes_summary(tmp_path):
    dataset_root = tmp_path / "raw"
    build_dataset(dataset_root)
    config_path = write_config(tmp_path, dataset_root)
    manifest_path = tmp_path / "custom" / "manifest.csv"
    summary_path = tmp_path / "custom" / "summary.json"

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--output",
            str(manifest_path),
            "--summary",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    assert manifest_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["dataset"] == "cli_test"
    assert summary["seed"] == 42
    assert summary["counts"]["train"]["total"] > summary["counts"]["test"]["total"]
