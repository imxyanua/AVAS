import pytest
import torch

from src.models.classifier import HIGH_RISK, NORMAL, SUSPICIOUS, BehaviorClassifier
from src.utils.config import AnomalyConfig, ConfigError, DatasetConfig, ModelConfig

CLASSES = ("walking", "standing", "falling", "fighting")
ABNORMAL = ("falling", "fighting")


def build_classifier(**anomaly) -> BehaviorClassifier:
    return BehaviorClassifier(CLASSES, ABNORMAL, AnomalyConfig(**anomaly) if anomaly else None)


def test_confident_normal_action_is_reported_as_normal():
    classifier = build_classifier()

    decision = classifier.decide(torch.tensor([0.9, 0.06, 0.02, 0.02]))

    assert decision.action == "walking"
    assert decision.confidence == pytest.approx(0.9)
    assert decision.is_abnormal is False
    assert decision.status == "normal"
    assert decision.anomaly_score == pytest.approx(0.04)
    assert decision.risk_level == NORMAL


def test_confident_abnormal_action_is_high_risk():
    classifier = build_classifier()

    decision = classifier.decide(torch.tensor([0.02, 0.03, 0.90, 0.05]))

    assert decision.action == "falling"
    assert decision.is_abnormal is True
    assert decision.status == "abnormal"
    assert decision.anomaly_score == pytest.approx(0.95)
    assert decision.risk_level == HIGH_RISK


def test_probability_spread_across_abnormal_classes_still_raises_the_score():
    classifier = build_classifier()

    decision = classifier.decide(torch.tensor([0.4, 0.0, 0.3, 0.3]))

    assert decision.action == "walking"
    assert decision.is_abnormal is False
    assert decision.anomaly_score == pytest.approx(0.6)
    assert decision.risk_level == SUSPICIOUS


def test_risk_levels_follow_the_configured_thresholds():
    classifier = build_classifier(suspicious_threshold=0.3, high_risk_threshold=0.6)

    assert classifier.risk_level(0.29) == NORMAL
    assert classifier.risk_level(0.3) == SUSPICIOUS
    assert classifier.risk_level(0.59) == SUSPICIOUS
    assert classifier.risk_level(0.6) == HIGH_RISK


def test_decide_batch_returns_one_decision_per_row():
    classifier = build_classifier()
    probabilities = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.05, 0.05, 0.1, 0.8]])

    decisions = classifier.decide_batch(probabilities)

    assert [decision.action for decision in decisions] == ["walking", "fighting"]
    assert [decision.status for decision in decisions] == ["normal", "abnormal"]


def test_decide_from_logits_applies_softmax():
    classifier = build_classifier()
    logits = torch.tensor([[8.0, 0.0, 0.0, 0.0]])

    decisions = classifier.decide_from_logits(logits)

    assert decisions[0].action == "walking"
    assert decisions[0].confidence > 0.99


def test_logits_are_rejected_when_passed_as_probabilities():
    classifier = build_classifier()

    with pytest.raises(ValueError, match="softmax"):
        classifier.decide(torch.tensor([4.0, 1.0, 0.5, 0.5]))


def test_negative_values_are_rejected():
    classifier = build_classifier()

    with pytest.raises(ValueError, match="negative"):
        classifier.decide(torch.tensor([1.2, 0.1, -0.2, -0.1]))


def test_wrong_number_of_probabilities_is_rejected():
    classifier = build_classifier()

    with pytest.raises(ValueError, match="expected 4 probabilities"):
        classifier.decide(torch.tensor([0.5, 0.5]))


def test_batch_shape_is_validated():
    classifier = build_classifier()

    with pytest.raises(ValueError, match=r"\(B, num_classes\)"):
        classifier.decide_batch(torch.tensor([0.25, 0.25, 0.25, 0.25]))


def test_unknown_abnormal_class_is_rejected():
    with pytest.raises(ValueError, match="unknown classes"):
        BehaviorClassifier(CLASSES, ("arson",))


def test_at_least_two_classes_are_required():
    with pytest.raises(ValueError, match="at least two"):
        BehaviorClassifier(("walking",), ())


def test_from_config_uses_the_dataset_taxonomy_and_model_thresholds():
    dataset = DatasetConfig.from_mapping(
        {
            "name": "decision_test",
            "root": "data/raw/decision_test",
            "classes": list(CLASSES),
            "behavior_map": {"normal": ["walking", "standing"], "abnormal": list(ABNORMAL)},
            "split": {"train": 0.7, "validation": 0.15, "test": 0.15},
        }
    )
    model = ModelConfig.from_mapping(
        {"anomaly": {"suspicious_threshold": 0.4, "high_risk_threshold": 0.7}}
    )

    classifier = BehaviorClassifier.from_config(dataset, model)

    assert classifier.classes == CLASSES
    assert classifier.abnormal_indices == (2, 3)
    assert classifier.risk_level(0.45) == SUSPICIOUS
    assert classifier.risk_level(0.75) == HIGH_RISK


def test_thresholds_must_be_ordered():
    with pytest.raises(ConfigError, match="lower than"):
        AnomalyConfig(suspicious_threshold=0.9, high_risk_threshold=0.5)


def test_thresholds_must_be_within_range():
    with pytest.raises(ConfigError, match=r"\(0.0, 1.0\]"):
        AnomalyConfig(suspicious_threshold=0.0, high_risk_threshold=0.8)
