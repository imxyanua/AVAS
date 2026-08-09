import numpy as np
import pytest

from src.utils.metrics import (
    anomaly_detection_metrics,
    balanced_class_weights,
    compute_metrics,
)

CLASSES = ("walking", "falling", "fighting")


def test_perfect_predictions_score_one():
    labels = [0, 1, 2, 0, 1, 2]

    report = compute_metrics(labels, labels, CLASSES)

    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0
    assert np.array_equal(report.confusion, np.eye(3, dtype=int) * 2)


def test_metrics_match_a_hand_computed_example():
    y_true = [0, 0, 0, 1, 1, 2]
    y_pred = [0, 0, 1, 1, 2, 2]

    report = compute_metrics(y_true, y_pred, CLASSES)

    assert report.accuracy == pytest.approx(4 / 6)
    walking = report.per_class["walking"]
    assert walking.precision == pytest.approx(1.0)
    assert walking.recall == pytest.approx(2 / 3)
    assert walking.f1 == pytest.approx(0.8)
    assert walking.support == 3
    falling = report.per_class["falling"]
    assert falling.precision == pytest.approx(0.5)
    assert falling.recall == pytest.approx(0.5)
    assert report.per_class["fighting"].precision == pytest.approx(0.5)
    assert report.per_class["fighting"].recall == pytest.approx(1.0)


def test_confusion_matrix_orientation_is_actual_by_predicted():
    report = compute_metrics([0, 0], [1, 1], CLASSES)

    assert report.confusion[0, 1] == 2
    assert report.confusion[1, 0] == 0


def test_absent_classes_are_reported_with_zeros():
    report = compute_metrics([0, 0, 1], [0, 0, 1], CLASSES)

    fighting = report.per_class["fighting"]
    assert fighting.support == 0
    assert fighting.precision == 0.0
    assert fighting.recall == 0.0


def test_macro_average_ignores_class_frequency():
    y_true = [0] * 18 + [1, 2]
    y_pred = [0] * 18 + [0, 0]

    report = compute_metrics(y_true, y_pred, CLASSES)

    assert report.accuracy == pytest.approx(0.9)
    assert report.macro_recall == pytest.approx(1 / 3)


def test_format_table_lists_every_class_and_accuracy():
    report = compute_metrics([0, 1, 2], [0, 1, 2], CLASSES)

    table = report.format_table()

    for name in CLASSES:
        assert name in table
    assert "macro" in table
    assert "accuracy" in table


def test_report_to_dict_is_json_friendly():
    report = compute_metrics([0, 1, 2], [0, 1, 1], CLASSES)

    payload = report.to_dict()

    assert payload["classes"] == list(CLASSES)
    assert isinstance(payload["confusion_matrix"], list)
    assert set(payload["per_class"]["walking"]) == {"precision", "recall", "f1", "support"}


def test_compute_metrics_validates_input():
    with pytest.raises(ValueError, match="match in length"):
        compute_metrics([0, 1], [0], CLASSES)
    with pytest.raises(ValueError, match="must not be empty"):
        compute_metrics([], [], CLASSES)
    with pytest.raises(ValueError, match="at least two"):
        compute_metrics([0], [0], ("walking",))
    with pytest.raises(ValueError, match="outside the class list"):
        compute_metrics([0, 5], [0, 1], CLASSES)


def test_anomaly_metrics_separate_abnormal_detection_from_action_accuracy():
    y_true = [0, 1, 2, 1]
    y_pred = [0, 2, 1, 0]

    metrics = anomaly_detection_metrics(y_true, y_pred, abnormal_indices=(1, 2))

    assert metrics["true_positive"] == 2
    assert metrics["false_negative"] == 1
    assert metrics["abnormal_recall"] == pytest.approx(2 / 3)
    assert metrics["abnormal_precision"] == pytest.approx(1.0)
    assert metrics["false_alarm_rate"] == pytest.approx(0.0)


def test_missed_abnormal_events_drive_recall_to_zero():
    metrics = anomaly_detection_metrics([1, 2], [0, 0], abnormal_indices=(1, 2))

    assert metrics["abnormal_recall"] == 0.0
    assert metrics["abnormal_f1"] == 0.0
    assert metrics["false_negative"] == 2


def test_false_alarms_are_reported():
    metrics = anomaly_detection_metrics([0, 0, 0, 1], [1, 0, 0, 1], abnormal_indices=(1, 2))

    assert metrics["false_positive"] == 1
    assert metrics["false_alarm_rate"] == pytest.approx(1 / 3)


def test_anomaly_metrics_require_abnormal_indices():
    with pytest.raises(ValueError, match="abnormal_indices"):
        anomaly_detection_metrics([0, 1], [0, 1], abnormal_indices=())


def test_balanced_weights_favour_rare_classes():
    weights = balanced_class_weights([0] * 9 + [1], num_classes=2)

    assert weights[1] > weights[0]
    assert weights[0] == pytest.approx(10 / (2 * 9))
    assert weights[1] == pytest.approx(10 / (2 * 1))


def test_balanced_weights_are_uniform_for_a_balanced_split():
    weights = balanced_class_weights([0, 0, 1, 1], num_classes=2)

    assert weights[0] == pytest.approx(weights[1])
    assert weights[0] == pytest.approx(1.0)


def test_absent_classes_receive_zero_weight():
    weights = balanced_class_weights([0, 0, 1], num_classes=3)

    assert weights[2] == 0.0


def test_balanced_weights_validate_input():
    with pytest.raises(ValueError, match="num_classes"):
        balanced_class_weights([0, 1], num_classes=1)
    with pytest.raises(ValueError, match="must not be empty"):
        balanced_class_weights([], num_classes=2)
