import pytest
import torch
from torch import nn

from src.features.feature_extractor import (
    build_backbone,
    encode_clips,
    extract_clip_features,
)
from src.utils.config import BackboneConfig
from src.utils.device import resolve_device


@pytest.fixture(scope="module")
def backbone_and_dim():
    return build_backbone(BackboneConfig(name="resnet18", pretrained=False, freeze=True))


def test_backbone_reports_its_feature_dimension(backbone_and_dim):
    backbone, feature_dim = backbone_and_dim

    assert feature_dim == 512
    assert isinstance(backbone.fc, nn.Identity)


def test_frozen_backbone_has_no_trainable_parameters(backbone_and_dim):
    backbone, _ = backbone_and_dim

    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    assert backbone.training is False


def test_unfrozen_backbone_keeps_gradients():
    backbone, _ = build_backbone(BackboneConfig(name="resnet18", pretrained=False, freeze=False))

    assert any(parameter.requires_grad for parameter in backbone.parameters())


def test_extract_clip_features_preserves_batch_and_time(backbone_and_dim):
    backbone, feature_dim = backbone_and_dim
    clips = torch.zeros(2, 3, 3, 64, 64)

    with torch.no_grad():
        features = extract_clip_features(backbone, clips)

    assert features.shape == (2, 3, feature_dim)


def test_extract_clip_features_rejects_wrong_rank(backbone_and_dim):
    backbone, _ = backbone_and_dim

    with pytest.raises(ValueError, match=r"\(B, T, C, H, W\)"):
        extract_clip_features(backbone, torch.zeros(3, 3, 64, 64))


def test_encode_clips_returns_detached_cpu_tensors(backbone_and_dim):
    backbone, feature_dim = backbone_and_dim

    features = encode_clips(backbone, torch.zeros(1, 2, 3, 64, 64), resolve_device("cpu"))

    assert features.shape == (1, 2, feature_dim)
    assert features.device.type == "cpu"
    assert features.requires_grad is False


def test_encode_clips_restores_the_training_flag():
    backbone, _ = build_backbone(BackboneConfig(name="resnet18", pretrained=False, freeze=False))
    backbone.train()

    encode_clips(backbone, torch.zeros(1, 1, 3, 64, 64), resolve_device("cpu"))

    assert backbone.training is True


def test_resolve_device_accepts_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_rejects_unknown_preference():
    with pytest.raises(ValueError, match="device must be"):
        resolve_device("tpu")


def test_resolve_device_auto_falls_back_to_cpu_without_cuda():
    device = resolve_device("auto")

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert device.type == expected
