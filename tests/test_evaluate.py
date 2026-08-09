import csv
import json

import pytest
import torch

from src.training.evaluate import load_checkpoint, main
from src.training.train import main as train_main


def train_args(feature_cache, run_name="eval"):
    return [
        "--dataset-config",
        str(feature_cache["dataset_config"]),
        "--model-config",
        str(feature_cache["model_config"]),
        "--training-config",
        str(feature_cache["training_config"]),
        "--feature-index",
        str(feature_cache["index"]),
        "--run-name",
        run_name,
        "--epochs",
        "6",
    ]


def evaluate_args(feature_cache, checkpoint, output_dir, *extra):
    return [
        "--checkpoint",
        str(checkpoint),
        "--dataset-config",
        str(feature_cache["dataset_config"]),
        "--training-config",
        str(feature_cache["training_config"]),
        "--feature-index",
        str(feature_cache["index"]),
        "--output-dir",
        str(output_dir),
        *extra,
    ]


@pytest.fixture
def trained_checkpoint(feature_cache):
    train_main(train_args(feature_cache))
    return feature_cache["checkpoint_dir"] / "eval.pt"


def test_evaluation_writes_a_report_with_action_and_anomaly_metrics(
    feature_cache, trained_checkpoint, tmp_path
):
    output_dir = tmp_path / "report"

    exit_code = main(evaluate_args(feature_cache, trained_checkpoint, output_dir))
    payload = json.loads((output_dir / "test_report.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["split"] == "test"
    assert payload["clips"] == feature_cache["split_sizes"]["test"]
    assert payload["action_metrics"]["classes"] == list(feature_cache["classes"])
    assert 0.0 <= payload["action_metrics"]["macro_f1"] <= 1.0
    assert set(payload["anomaly_metrics"]) >= {
        "abnormal_precision",
        "abnormal_recall",
        "abnormal_f1",
        "false_alarm_rate",
    }
    assert payload["anomaly_thresholds"] == {"suspicious": 0.5, "high_risk": 0.8}
    assert sum(payload["risk_levels"].values()) == payload["clips"]


def test_learned_model_reaches_high_accuracy_on_separable_features(
    feature_cache, trained_checkpoint, tmp_path
):
    output_dir = tmp_path / "report"

    main(evaluate_args(feature_cache, trained_checkpoint, output_dir))
    payload = json.loads((output_dir / "test_report.json").read_text(encoding="utf-8"))

    assert payload["action_metrics"]["accuracy"] > 0.8
    assert payload["anomaly_metrics"]["abnormal_recall"] > 0.5


def test_predictions_file_has_one_row_per_clip(feature_cache, trained_checkpoint, tmp_path):
    output_dir = tmp_path / "report"

    main(evaluate_args(feature_cache, trained_checkpoint, output_dir))
    with (output_dir / "predictions.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == feature_cache["split_sizes"]["test"]
    for row in rows:
        assert row["label"] in feature_cache["classes"]
        assert row["prediction"] in feature_cache["classes"]
        assert 0.0 <= float(row["confidence"]) <= 1.0
        assert 0.0 <= float(row["anomaly_score"]) <= 1.0
        assert row["risk_level"] in {"normal", "suspicious", "high_risk"}
        assert row["correct"] in {"0", "1"}


def test_confusion_matrix_csv_is_labelled(feature_cache, trained_checkpoint, tmp_path):
    output_dir = tmp_path / "report"

    main(evaluate_args(feature_cache, trained_checkpoint, output_dir))
    with (output_dir / "confusion_matrix.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["actual", *feature_cache["classes"]]
    assert [row[0] for row in rows[1:]] == list(feature_cache["classes"])
    total = sum(int(value) for row in rows[1:] for value in row[1:])
    assert total == feature_cache["split_sizes"]["test"]


def test_figure_flag_saves_an_image(feature_cache, trained_checkpoint, tmp_path):
    output_dir = tmp_path / "report"

    main(evaluate_args(feature_cache, trained_checkpoint, output_dir, "--figure"))

    figure = output_dir / "confusion_matrix.png"
    assert figure.is_file()
    assert figure.stat().st_size > 0


def test_other_splits_can_be_evaluated(feature_cache, trained_checkpoint, tmp_path):
    output_dir = tmp_path / "report"

    main(evaluate_args(feature_cache, trained_checkpoint, output_dir, "--split", "validation"))

    payload = json.loads((output_dir / "validation_report.json").read_text(encoding="utf-8"))
    assert payload["clips"] == feature_cache["split_sizes"]["validation"]


def test_class_mismatch_between_checkpoint_and_config_is_refused(
    feature_cache, trained_checkpoint, tmp_path
):
    payload = torch.load(trained_checkpoint, map_location="cpu", weights_only=True)
    payload["classes"] = ["walking", "running"]
    torch.save(payload, trained_checkpoint)

    with pytest.raises(SystemExit, match="was trained on classes"):
        main(evaluate_args(feature_cache, trained_checkpoint, tmp_path / "report"))


def test_checkpoint_without_architecture_requires_an_explicit_config(
    feature_cache, trained_checkpoint, tmp_path
):
    payload = torch.load(trained_checkpoint, map_location="cpu", weights_only=True)
    del payload["model_config"]
    torch.save(payload, trained_checkpoint)

    with pytest.raises(SystemExit, match="does not record its architecture"):
        main(evaluate_args(feature_cache, trained_checkpoint, tmp_path / "report"))

    assert (
        main(
            evaluate_args(
                feature_cache,
                trained_checkpoint,
                tmp_path / "report",
                "--model-config",
                str(feature_cache["model_config"]),
            )
        )
        == 0
    )


def test_missing_checkpoint_is_reported(feature_cache, tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        main(evaluate_args(feature_cache, tmp_path / "absent.pt", tmp_path / "report"))


def test_incomplete_checkpoint_is_reported(tmp_path):
    checkpoint = tmp_path / "partial.pt"
    torch.save({"model_state": {}}, checkpoint)

    with pytest.raises(ValueError, match="missing keys"):
        load_checkpoint(checkpoint, torch.device("cpu"))
