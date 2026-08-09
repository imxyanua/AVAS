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


def _duplicates(values: Sequence[str]) -> set[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return repeated
