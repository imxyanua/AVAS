import pytest
import torch

from src.models.cnn_lstm import CnnLstm, TemporalClassifier, build_model
from src.utils.config import ModelConfig

FEATURE_DIM = 16
NUM_CLASSES = 3


def model_config(**overrides) -> ModelConfig:
    payload = {
        "backbone": {"name": "resnet18", "pretrained": False, "freeze": True},
        "temporal": {"hidden_size": 8, "num_layers": 1, "bidirectional": False, "dropout": 0.0},
        "classifier": {"dropout": 0.0},
    }
    payload.update(overrides)
    return ModelConfig.from_mapping(payload)


def test_temporal_classifier_maps_sequences_to_class_logits():
    model = TemporalClassifier(FEATURE_DIM, NUM_CLASSES, model_config())

    logits = model(torch.randn(4, 6, FEATURE_DIM))

    assert logits.shape == (4, NUM_CLASSES)


def test_bidirectional_head_doubles_the_classifier_input():
    config = model_config(
        temporal={"hidden_size": 8, "num_layers": 2, "bidirectional": True, "dropout": 0.1}
    )
    model = TemporalClassifier(FEATURE_DIM, NUM_CLASSES, config)

    logits = model(torch.randn(2, 5, FEATURE_DIM))

    assert model.classifier.in_features == 16
    assert logits.shape == (2, NUM_CLASSES)


def test_single_layer_lstm_disables_recurrent_dropout():
    config = model_config(
        temporal={"hidden_size": 8, "num_layers": 1, "bidirectional": False, "dropout": 0.5}
    )
    model = TemporalClassifier(FEATURE_DIM, NUM_CLASSES, config)

    assert model.lstm.dropout == 0.0


def test_multi_layer_lstm_keeps_recurrent_dropout():
    config = model_config(
        temporal={"hidden_size": 8, "num_layers": 2, "bidirectional": False, "dropout": 0.5}
    )
    model = TemporalClassifier(FEATURE_DIM, NUM_CLASSES, config)

    assert model.lstm.dropout == pytest.approx(0.5)


def test_prediction_depends_on_frame_order():
    torch.manual_seed(0)
    model = TemporalClassifier(FEATURE_DIM, NUM_CLASSES, model_config()).eval()
    sequence = torch.randn(1, 6, FEATURE_DIM)

    forward = model(sequence)
    reversed_order = model(sequence.flip(dims=[1]))

    assert not torch.allclose(forward, reversed_order)


def test_temporal_classifier_rejects_wrong_feature_dimension():
    model = TemporalClassifier(FEATURE_DIM, NUM_CLASSES, model_config())

    with pytest.raises(ValueError, match="feature dimension"):
        model(torch.randn(2, 4, FEATURE_DIM + 1))


def test_temporal_classifier_rejects_wrong_rank():
    model = TemporalClassifier(FEATURE_DIM, NUM_CLASSES, model_config())

    with pytest.raises(ValueError, match=r"\(B, T, D\)"):
        model(torch.randn(4, FEATURE_DIM))


@pytest.mark.parametrize(
    ("feature_dim", "num_classes"),
    [(0, 3), (16, 1)],
)
def test_temporal_classifier_validates_dimensions(feature_dim, num_classes):
    with pytest.raises(ValueError):
        TemporalClassifier(feature_dim, num_classes, model_config())


def test_cnn_lstm_consumes_raw_clips():
    model = CnnLstm(NUM_CLASSES, model_config()).eval()

    logits = model(torch.zeros(2, 3, 3, 64, 64))

    assert logits.shape == (2, NUM_CLASSES)
    assert model.feature_dim == 512


def test_frozen_backbone_leaves_only_the_head_trainable():
    model = CnnLstm(NUM_CLASSES, model_config())

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert trainable
    assert all(name.startswith("temporal.") for name in trainable)


def test_unfrozen_backbone_is_trainable_end_to_end():
    config = model_config(backbone={"name": "resnet18", "pretrained": False, "freeze": False})
    model = CnnLstm(NUM_CLASSES, config)

    trainable = {name.split(".")[0] for name, p in model.named_parameters() if p.requires_grad}

    assert "backbone" in trainable
    assert "temporal" in trainable


def test_build_model_returns_the_head_when_features_are_cached():
    model = build_model(model_config(), NUM_CLASSES, feature_dim=FEATURE_DIM)

    assert isinstance(model, TemporalClassifier)


def test_build_model_returns_the_full_model_without_cached_features():
    model = build_model(model_config(), NUM_CLASSES)

    assert isinstance(model, CnnLstm)


def test_build_model_rejects_unknown_architecture():
    config = model_config()
    object.__setattr__(config, "architecture", "transformer")

    with pytest.raises(ValueError, match="unsupported architecture"):
        build_model(config, NUM_CLASSES, feature_dim=FEATURE_DIM)


def test_gradients_reach_the_head():
    model = TemporalClassifier(FEATURE_DIM, NUM_CLASSES, model_config())
    logits = model(torch.randn(3, 5, FEATURE_DIM))

    logits.sum().backward()

    assert model.classifier.weight.grad is not None
    assert torch.isfinite(model.classifier.weight.grad).all()
