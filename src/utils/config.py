"""Typed configuration objects loaded from YAML files.

Configuration is validated eagerly so that an inconsistent experiment setup
fails before any video is decoded or any model is trained.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_VIDEO_EXTENSIONS: tuple[str, ...] = (".avi", ".mp4", ".mkv", ".mov")
SAMPLING_STRATEGIES: frozenset[str] = frozenset({"uniform", "center", "random"})
SUPPORTED_BACKBONES: frozenset[str] = frozenset({"resnet18", "resnet34", "resnet50"})
SUPPORTED_ARCHITECTURES: frozenset[str] = frozenset({"cnn_lstm"})
SUPPORTED_DEVICES: frozenset[str] = frozenset({"auto", "cpu", "cuda"})
SUPPORTED_CLASS_WEIGHTING: frozenset[str] = frozenset({"none", "balanced"})
RATIO_TOLERANCE = 1e-6


class ConfigError(ValueError):
    """Raised when a configuration file is missing or internally inconsistent."""


@dataclass(frozen=True)
class SplitConfig:
    """Fractions of source recordings assigned to each split."""

    train: float
    validation: float
    test: float
    seed: int = 42

    def __post_init__(self) -> None:
        for name, value in self.as_ratios().items():
            if not 0.0 < value < 1.0:
                raise ConfigError(f"split.{name} must be between 0 and 1, got {value}")
        total = sum(self.as_ratios().values())
        if abs(total - 1.0) > RATIO_TOLERANCE:
            raise ConfigError(f"split ratios must sum to 1.0, got {total}")

    def as_ratios(self) -> dict[str, float]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


@dataclass(frozen=True)
class SamplingConfig:
    """How a fixed-length clip is drawn from a variable-length video."""

    clip_length: int = 16
    frame_stride: int = 2
    strategy: str = "uniform"
    image_size: int = 224

    def __post_init__(self) -> None:
        if self.clip_length < 1:
            raise ConfigError(f"sampling.clip_length must be >= 1, got {self.clip_length}")
        if self.frame_stride < 1:
            raise ConfigError(f"sampling.frame_stride must be >= 1, got {self.frame_stride}")
        if self.image_size < 1:
            raise ConfigError(f"sampling.image_size must be >= 1, got {self.image_size}")
        if self.strategy not in SAMPLING_STRATEGIES:
            allowed = ", ".join(sorted(SAMPLING_STRATEGIES))
            raise ConfigError(
                f"sampling.strategy must be one of [{allowed}], got {self.strategy!r}"
            )


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset location, label taxonomy, split ratios and clip sampling."""

    name: str
    root: Path
    processed_root: Path
    classes: tuple[str, ...]
    normal_classes: frozenset[str]
    abnormal_classes: frozenset[str]
    split: SplitConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    video_extensions: tuple[str, ...] = DEFAULT_VIDEO_EXTENSIONS
    group_pattern: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("name must not be empty")
        if not self.classes:
            raise ConfigError("classes must not be empty")

        duplicates = _duplicates(self.classes)
        if duplicates:
            raise ConfigError(f"classes contains duplicates: {sorted(duplicates)}")

        if not self.video_extensions:
            raise ConfigError("video_extensions must not be empty")
        for extension in self.video_extensions:
            if not extension.startswith("."):
                raise ConfigError(f"video extension must start with '.', got {extension!r}")

        overlap = self.normal_classes & self.abnormal_classes
        if overlap:
            raise ConfigError(
                f"behavior_map lists the same class as normal and abnormal: {sorted(overlap)}"
            )
        mapped = self.normal_classes | self.abnormal_classes
        declared = set(self.classes)
        unmapped = declared - mapped
        if unmapped:
            raise ConfigError(f"behavior_map is missing classes: {sorted(unmapped)}")
        unknown = mapped - declared
        if unknown:
            raise ConfigError(f"behavior_map references unknown classes: {sorted(unknown)}")

        if self.group_pattern is not None:
            try:
                compiled = re.compile(self.group_pattern)
            except re.error as error:
                raise ConfigError(f"group_pattern is not a valid regex: {error}") from error
            if compiled.groups < 1:
                raise ConfigError("group_pattern must contain at least one capturing group")

    @property
    def class_to_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.classes)}

    def is_abnormal(self, class_name: str) -> bool:
        if class_name not in self.class_to_index:
            raise KeyError(f"unknown class: {class_name!r}")
        return class_name in self.abnormal_classes

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DatasetConfig:
        required = ("name", "root", "classes", "behavior_map", "split")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ConfigError(f"missing required keys: {missing}")

        behavior_map = payload["behavior_map"]
        if not isinstance(behavior_map, Mapping):
            raise ConfigError("behavior_map must be a mapping with 'normal' and 'abnormal' keys")
        unexpected = set(behavior_map) - {"normal", "abnormal"}
        if unexpected:
            raise ConfigError(f"behavior_map has unexpected keys: {sorted(unexpected)}")

        root = Path(payload["root"])
        processed_root = Path(payload.get("processed_root", root.parent / "processed"))
        extensions = payload.get("video_extensions", DEFAULT_VIDEO_EXTENSIONS)

        return cls(
            name=str(payload["name"]),
            root=root,
            processed_root=processed_root,
            classes=_as_str_tuple(payload["classes"], "classes"),
            normal_classes=frozenset(_as_str_tuple(behavior_map.get("normal", ()), "normal")),
            abnormal_classes=frozenset(_as_str_tuple(behavior_map.get("abnormal", ()), "abnormal")),
            split=SplitConfig(**_as_mapping(payload["split"], "split")),
            sampling=SamplingConfig(**_as_mapping(payload.get("sampling", {}), "sampling")),
            video_extensions=tuple(
                extension.lower() for extension in _as_str_tuple(extensions, "video_extensions")
            ),
            group_pattern=payload.get("group_pattern"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetConfig:
        return cls.from_mapping(load_yaml(path))


@dataclass(frozen=True)
class BackboneConfig:
    """Convolutional feature extractor applied to every frame."""

    name: str = "resnet18"
    pretrained: bool = True
    freeze: bool = True

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_BACKBONES:
            allowed = ", ".join(sorted(SUPPORTED_BACKBONES))
            raise ConfigError(f"backbone.name must be one of [{allowed}], got {self.name!r}")


@dataclass(frozen=True)
class TemporalConfig:
    """Recurrent head that models how frame features evolve over time."""

    hidden_size: int = 256
    num_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.3

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ConfigError(f"temporal.hidden_size must be >= 1, got {self.hidden_size}")
        if self.num_layers < 1:
            raise ConfigError(f"temporal.num_layers must be >= 1, got {self.num_layers}")
        _check_dropout(self.dropout, "temporal.dropout")


@dataclass(frozen=True)
class AnomalyConfig:
    """Score thresholds that separate normal, suspicious and high-risk events.

    The defaults are provisional. Thresholds decide how many events an operator
    must review, so they have to be calibrated on validation data rather than
    accepted as given.
    """

    suspicious_threshold: float = 0.5
    high_risk_threshold: float = 0.8

    def __post_init__(self) -> None:
        for name, value in (
            ("suspicious_threshold", self.suspicious_threshold),
            ("high_risk_threshold", self.high_risk_threshold),
        ):
            if not 0.0 < value <= 1.0:
                raise ConfigError(f"anomaly.{name} must be in (0.0, 1.0], got {value}")
        if self.suspicious_threshold >= self.high_risk_threshold:
            raise ConfigError(
                "anomaly.suspicious_threshold must be lower than anomaly.high_risk_threshold, "
                f"got {self.suspicious_threshold} and {self.high_risk_threshold}"
            )


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the action recognition model."""

    architecture: str = "cnn_lstm"
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    classifier_dropout: float = 0.5

    def __post_init__(self) -> None:
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            allowed = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
            raise ConfigError(f"architecture must be one of [{allowed}], got {self.architecture!r}")
        _check_dropout(self.classifier_dropout, "classifier.dropout")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ModelConfig:
        classifier = _as_mapping(payload.get("classifier", {}), "classifier")
        unexpected = set(classifier) - {"dropout"}
        if unexpected:
            raise ConfigError(f"classifier has unexpected keys: {sorted(unexpected)}")

        return cls(
            architecture=str(payload.get("architecture", "cnn_lstm")),
            backbone=BackboneConfig(**_as_mapping(payload.get("backbone", {}), "backbone")),
            temporal=TemporalConfig(**_as_mapping(payload.get("temporal", {}), "temporal")),
            anomaly=AnomalyConfig(**_as_mapping(payload.get("anomaly", {}), "anomaly")),
            classifier_dropout=float(classifier.get("dropout", 0.5)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        return cls.from_mapping(load_yaml(path))

    def to_mapping(self) -> dict[str, Any]:
        """Serialise back to the YAML shape, for checkpoints and run provenance."""
        return {
            "architecture": self.architecture,
            "backbone": {
                "name": self.backbone.name,
                "pretrained": self.backbone.pretrained,
                "freeze": self.backbone.freeze,
            },
            "temporal": {
                "hidden_size": self.temporal.hidden_size,
                "num_layers": self.temporal.num_layers,
                "bidirectional": self.temporal.bidirectional,
                "dropout": self.temporal.dropout,
            },
            "anomaly": {
                "suspicious_threshold": self.anomaly.suspicious_threshold,
                "high_risk_threshold": self.anomaly.high_risk_threshold,
            },
            "classifier": {"dropout": self.classifier_dropout},
        }


@dataclass(frozen=True)
class DetectionConfig:
    """Person detector settings."""

    weights: str = "yolov8n.pt"
    person_class_id: int = 0
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.5
    image_size: int = 640
    max_detections: int = 20

    def __post_init__(self) -> None:
        if not self.weights:
            raise ConfigError("detection.weights must not be empty")
        if self.person_class_id < 0:
            raise ConfigError(f"detection.person_class_id must be >= 0, got {self.person_class_id}")
        if not 0.0 < self.confidence_threshold < 1.0:
            raise ConfigError(
                f"detection.confidence_threshold must be in (0.0, 1.0), "
                f"got {self.confidence_threshold}"
            )
        if not 0.0 < self.iou_threshold < 1.0:
            raise ConfigError(
                f"detection.iou_threshold must be in (0.0, 1.0), got {self.iou_threshold}"
            )
        if self.image_size < 32:
            raise ConfigError(f"detection.image_size must be >= 32, got {self.image_size}")
        if self.max_detections < 1:
            raise ConfigError(f"detection.max_detections must be >= 1, got {self.max_detections}")


@dataclass(frozen=True)
class TrackingConfig:
    """Association and lifecycle settings for person tracking."""

    iou_threshold: float = 0.3
    max_age: int = 15
    min_hits: int = 3
    min_track_length: int = 8

    def __post_init__(self) -> None:
        if not 0.0 < self.iou_threshold < 1.0:
            raise ConfigError(
                f"tracking.iou_threshold must be in (0.0, 1.0), got {self.iou_threshold}"
            )
        if self.max_age < 1:
            raise ConfigError(f"tracking.max_age must be >= 1, got {self.max_age}")
        if self.min_hits < 1:
            raise ConfigError(f"tracking.min_hits must be >= 1, got {self.min_hits}")
        if self.min_track_length < 1:
            raise ConfigError(
                f"tracking.min_track_length must be >= 1, got {self.min_track_length}"
            )


@dataclass(frozen=True)
class ClipSelectionConfig:
    """How clips are cut out of a person track for recognition."""

    box_padding: float = 0.1
    max_clips_per_track: int = 4

    def __post_init__(self) -> None:
        if not 0.0 <= self.box_padding < 1.0:
            raise ConfigError(f"clips.box_padding must be in [0.0, 1.0), got {self.box_padding}")
        if self.max_clips_per_track < 1:
            raise ConfigError(
                f"clips.max_clips_per_track must be >= 1, got {self.max_clips_per_track}"
            )


@dataclass(frozen=True)
class InferenceConfig:
    """Detection, tracking and clip selection settings for analysing a video."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    clips: ClipSelectionConfig = field(default_factory=ClipSelectionConfig)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> InferenceConfig:
        unexpected = set(payload) - {"detection", "tracking", "clips"}
        if unexpected:
            raise ConfigError(f"inference config has unexpected keys: {sorted(unexpected)}")
        return cls(
            detection=DetectionConfig(**_as_mapping(payload.get("detection", {}), "detection")),
            tracking=TrackingConfig(**_as_mapping(payload.get("tracking", {}), "tracking")),
            clips=ClipSelectionConfig(**_as_mapping(payload.get("clips", {}), "clips")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> InferenceConfig:
        return cls.from_mapping(load_yaml(path))


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Stops training once the monitored validation metric stops improving."""

    monitor: str = "val_macro_f1"
    patience: int = 5

    def __post_init__(self) -> None:
        if not self.monitor:
            raise ConfigError("early_stopping.monitor must not be empty")
        if self.patience < 1:
            raise ConfigError(f"early_stopping.patience must be >= 1, got {self.patience}")


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation settings and output locations for a training run."""

    seed: int = 42
    device: str = "auto"
    epochs: int = 30
    batch_size: int = 16
    num_workers: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float | None = 5.0
    class_weighting: str = "balanced"
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    feature_cache_root: Path = Path("data/processed/features")
    checkpoint_dir: Path = Path("models/checkpoints")
    log_dir: Path = Path("outputs/logs")

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ConfigError(f"seed must be non-negative, got {self.seed}")
        if self.device not in SUPPORTED_DEVICES:
            allowed = ", ".join(sorted(SUPPORTED_DEVICES))
            raise ConfigError(f"device must be one of [{allowed}], got {self.device!r}")
        if self.epochs < 1:
            raise ConfigError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_workers < 0:
            raise ConfigError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.learning_rate <= 0:
            raise ConfigError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ConfigError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ConfigError(
                f"gradient_clip_norm must be > 0 or null, got {self.gradient_clip_norm}"
            )
        if self.class_weighting not in SUPPORTED_CLASS_WEIGHTING:
            allowed = ", ".join(sorted(SUPPORTED_CLASS_WEIGHTING))
            raise ConfigError(
                f"class_weighting must be one of [{allowed}], got {self.class_weighting!r}"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TrainingConfig:
        settings = dict(payload)
        early_stopping = EarlyStoppingConfig(
            **_as_mapping(settings.pop("early_stopping", {}), "early_stopping")
        )
        paths = {
            key: Path(settings.pop(key))
            for key in ("feature_cache_root", "checkpoint_dir", "log_dir")
            if key in settings
        }
        return cls(early_stopping=early_stopping, **settings, **paths)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        return cls.from_mapping(load_yaml(path))


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dictionary."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        raise ConfigError(f"configuration file is empty: {config_path}")
    if not isinstance(payload, dict):
        raise ConfigError(f"configuration root must be a mapping: {config_path}")
    return payload


def _as_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field_name} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _as_str_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigError(f"{field_name} must be a list of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{field_name} must contain non-empty strings, got {item!r}")
        items.append(item)
    return tuple(items)


def _check_dropout(value: float, field_name: str) -> None:
    if not 0.0 <= value < 1.0:
        raise ConfigError(f"{field_name} must be in [0.0, 1.0), got {value}")


def _duplicates(values: Sequence[str]) -> set[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return repeated
