"""Reading the feature cache index produced by ``build_feature_cache``.

The index is the link between cached arrays and the split they belong to. Reading
it back also gives a cheap chance to re-check that a cache built earlier still
respects the split boundaries, which catches a cache left over from a different
manifest.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.preprocessing.dataset import CachedFeatureDataset
from src.utils.config import DatasetConfig

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = frozenset(
    {"split", "label", "label_index", "group", "video", "clip", "feature_path"}
)


class FeatureCacheError(ValueError):
    """Raised when the feature cache index is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class CachedClip:
    """One cached clip: where its features live and what they represent."""

    split: str
    label: str
    label_index: int
    group: str
    video: Path
    clip: int
    feature_path: Path

    @property
    def group_key(self) -> str:
        return f"{self.label}/{self.group}"


def read_feature_index(path: str | Path) -> list[CachedClip]:
    """Load every row of a feature cache index."""
    index_path = Path(path)
    if not index_path.is_file():
        raise FeatureCacheError(f"feature index not found: {index_path}")

    with index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise FeatureCacheError(f"feature index is missing columns: {sorted(missing)}")

        clips = [
            CachedClip(
                split=row["split"],
                label=row["label"],
                label_index=int(row["label_index"]),
                group=row["group"],
                video=Path(row["video"]),
                clip=int(row["clip"]),
                feature_path=Path(row["feature_path"]),
            )
            for row in reader
        ]

    if not clips:
        raise FeatureCacheError(f"feature index is empty: {index_path}")
    return clips


def clips_for_split(clips: Sequence[CachedClip], split: str) -> list[CachedClip]:
    """Select the clips belonging to one split."""
    selected = [clip for clip in clips if clip.split == split]
    if not selected:
        available = sorted({clip.split for clip in clips})
        raise FeatureCacheError(f"no cached clips for split {split!r}; available: {available}")
    return selected


def assert_no_group_leakage(clips: Sequence[CachedClip]) -> None:
    """Verify that no source recording has cached clips in two splits."""
    owner: dict[str, str] = {}
    conflicts: dict[str, set[str]] = defaultdict(set)

    for clip in clips:
        previous = owner.setdefault(clip.group_key, clip.split)
        if previous != clip.split:
            conflicts[clip.group_key].update({previous, clip.split})

    if conflicts:
        details = ", ".join(
            f"{group} in {sorted(splits)}" for group, splits in sorted(conflicts.items())
        )
        raise FeatureCacheError(f"cached recordings shared between splits: {details}")


def validate_against_config(clips: Sequence[CachedClip], config: DatasetConfig) -> None:
    """Check that cached labels and indices match the configured taxonomy."""
    expected = config.class_to_index
    for clip in clips:
        if clip.label not in expected:
            raise FeatureCacheError(
                f"cached label {clip.label!r} is not in the configured classes: {sorted(expected)}"
            )
        if expected[clip.label] != clip.label_index:
            raise FeatureCacheError(
                f"cached label index for {clip.label!r} is {clip.label_index}, "
                f"config expects {expected[clip.label]}"
            )


def feature_dimension(clips: Sequence[CachedClip]) -> int:
    """Read the feature dimension from the first cached array."""
    if not clips:
        raise FeatureCacheError("clips must not be empty")

    first = clips[0].feature_path
    if not first.is_file():
        raise FeatureCacheError(f"cached feature file not found: {first}")

    features = np.load(first)
    if features.ndim != 2:
        raise FeatureCacheError(
            f"expected features with shape (T, D) in {first}, got {features.shape}"
        )
    return int(features.shape[1])


def build_dataset(
    clips: Sequence[CachedClip], *, expected_dim: int | None = None
) -> CachedFeatureDataset:
    """Wrap cached clips in a dataset, checking that every file is present."""
    missing = [clip.feature_path for clip in clips if not clip.feature_path.is_file()]
    if missing:
        raise FeatureCacheError(
            f"{len(missing)} cached feature files are missing, first: {missing[0]}"
        )

    return CachedFeatureDataset(
        [clip.feature_path for clip in clips],
        [clip.label_index for clip in clips],
        expected_dim=expected_dim,
    )
