"""Dataset partitioning at the level of source recordings.

Action recognition datasets usually contain several clips cut from the same
recording, and consecutive frames of one clip are almost identical. Splitting
such material per frame or per clip lets near-duplicates appear in both training
and test data, which inflates reported accuracy. Every function here therefore
treats a *group* of clips from one recording as the smallest indivisible unit.
"""

from __future__ import annotations

import csv
import logging
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from src.utils.config import DEFAULT_VIDEO_EXTENSIONS, DatasetConfig

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = ("split", "label", "group", "path")


class DatasetLayoutError(ValueError):
    """Raised when the dataset directory does not match the configured classes."""


class SplitError(ValueError):
    """Raised when a requested partition cannot be produced."""


class SplitLeakageError(AssertionError):
    """Raised when clips from one recording appear in more than one split."""


@dataclass(frozen=True)
class VideoSample:
    """A single video file with its class label and source-recording group."""

    path: Path
    label: str
    group: str

    @property
    def group_key(self) -> str:
        """Group identifier namespaced by label, so identical names never merge."""
        return f"{self.label}/{self.group}"


def discover_videos(
    root: str | Path,
    classes: Sequence[str],
    *,
    extensions: Sequence[str] = DEFAULT_VIDEO_EXTENSIONS,
    group_pattern: str | None = None,
) -> list[VideoSample]:
    """Collect videos from ``root/<class_name>/`` in a deterministic order."""
    dataset_root = Path(root)
    if not dataset_root.is_dir():
        raise DatasetLayoutError(f"dataset root not found: {dataset_root}")
    if not classes:
        raise DatasetLayoutError("classes must not be empty")

    allowed = {extension.lower() for extension in extensions}
    compiled = re.compile(group_pattern) if group_pattern else None

    samples: list[VideoSample] = []
    for label in classes:
        class_dir = dataset_root / label
        if not class_dir.is_dir():
            raise DatasetLayoutError(f"missing directory for class {label!r}: {class_dir}")

        files = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in allowed
        )
        if not files:
            raise DatasetLayoutError(f"no videos with extensions {sorted(allowed)} in {class_dir}")

        for path in files:
            samples.append(VideoSample(path=path, label=label, group=_group_of(path, compiled)))

    return samples


def discover_from_config(config: DatasetConfig) -> list[VideoSample]:
    """Discover videos using the paths and taxonomy of a dataset configuration."""
    return discover_videos(
        config.root,
        config.classes,
        extensions=config.video_extensions,
        group_pattern=config.group_pattern,
    )


def split_videos(
    samples: Iterable[VideoSample],
    ratios: Mapping[str, float],
    *,
    seed: int = 42,
) -> dict[str, list[VideoSample]]:
    """Assign whole groups to splits, keeping each class close to ``ratios``.

    Groups are shuffled with ``seed``, ordered from largest to smallest, and then
    given to whichever split is furthest below its target video count. Larger
    groups are placed first because they constrain the achievable proportions
    most, and the assignment is fully determined by ``seed``.
    """
    sample_list = list(samples)
    if not sample_list:
        raise SplitError("no samples to split")

    split_names = list(ratios)
    if not split_names:
        raise SplitError("ratios must not be empty")
    for name, ratio in ratios.items():
        if ratio <= 0:
            raise SplitError(f"ratio for {name!r} must be > 0, got {ratio}")

    total_ratio = sum(ratios.values())
    normalised = {name: ratio / total_ratio for name, ratio in ratios.items()}
    assigned: dict[str, list[VideoSample]] = {name: [] for name in split_names}

    for label, label_samples in sorted(_by_label(sample_list).items()):
        groups = _by_group(label_samples)
        if len(groups) < len(split_names):
            logger.warning(
                "class %r has %d source recordings for %d splits; "
                "some splits will not contain this class",
                label,
                len(groups),
                len(split_names),
            )

        group_keys = sorted(groups)
        random.Random(f"{seed}:{label}").shuffle(group_keys)
        group_keys.sort(key=lambda key: len(groups[key]), reverse=True)

        targets = {name: normalised[name] * len(label_samples) for name in split_names}
        counts = dict.fromkeys(split_names, 0)
        for group_key in group_keys:
            chosen = max(split_names, key=lambda name: targets[name] - counts[name])
            assigned[chosen].extend(groups[group_key])
            counts[chosen] += len(groups[group_key])

    empty = [name for name, items in assigned.items() if not items]
    if empty:
        recordings = {
            label: len({sample.group_key for sample in label_samples})
            for label, label_samples in sorted(_by_label(sample_list).items())
        }
        smallest = min(normalised.values())
        raise SplitError(
            f"splits {empty} received no videos. Splitting keeps whole source recordings "
            f"together, so each split needs at least one. Recordings per class: {recordings}; "
            f"requested ratios: { {name: round(ratio, 3) for name, ratio in normalised.items()} }. "
            f"Add more recordings, or raise the smallest ratio ({smallest:.3f})."
        )

    return {name: sorted(items, key=lambda sample: sample.path) for name, items in assigned.items()}


def split_from_config(
    samples: Iterable[VideoSample], config: DatasetConfig
) -> dict[str, list[VideoSample]]:
    """Split samples using the ratios and seed of a dataset configuration."""
    return split_videos(samples, config.split.as_ratios(), seed=config.split.seed)


def assert_no_group_leakage(splits: Mapping[str, Sequence[VideoSample]]) -> None:
    """Verify that no source recording contributes videos to two splits."""
    owner: dict[str, str] = {}
    conflicts: dict[str, set[str]] = defaultdict(set)

    for split_name, samples in splits.items():
        for sample in samples:
            previous = owner.setdefault(sample.group_key, split_name)
            if previous != split_name:
                conflicts[sample.group_key].update({previous, split_name})

    if conflicts:
        details = ", ".join(
            f"{group} in {sorted(names)}" for group, names in sorted(conflicts.items())
        )
        raise SplitLeakageError(f"source recordings shared between splits: {details}")


def summarize_splits(splits: Mapping[str, Sequence[VideoSample]]) -> dict[str, dict[str, int]]:
    """Count videos per class per split, plus per-split totals."""
    summary: dict[str, dict[str, int]] = {}
    for split_name, samples in splits.items():
        counts: dict[str, int] = defaultdict(int)
        for sample in samples:
            counts[sample.label] += 1
        counts["total"] = len(samples)
        summary[split_name] = dict(sorted(counts.items()))
    return summary


def write_manifest(
    splits: Mapping[str, Sequence[VideoSample]],
    path: str | Path,
    *,
    relative_to: str | Path | None = None,
) -> Path:
    """Persist a split assignment as CSV so experiments can be reproduced."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    base = Path(relative_to) if relative_to is not None else None

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_COLUMNS)
        for split_name, samples in splits.items():
            for sample in samples:
                video_path = sample.path
                if base is not None:
                    with suppress(ValueError):
                        video_path = video_path.relative_to(base)
                writer.writerow([split_name, sample.label, sample.group, video_path.as_posix()])

    return manifest_path


def read_manifest(
    path: str | Path, *, relative_to: str | Path | None = None
) -> dict[str, list[VideoSample]]:
    """Load a split assignment written by :func:`write_manifest`."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise SplitError(f"manifest not found: {manifest_path}")

    base = Path(relative_to) if relative_to is not None else None
    splits: dict[str, list[VideoSample]] = defaultdict(list)

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(MANIFEST_COLUMNS):
            raise SplitError(
                f"manifest columns must be {list(MANIFEST_COLUMNS)}, got {reader.fieldnames}"
            )
        for row in reader:
            video_path = Path(row["path"])
            if base is not None and not video_path.is_absolute():
                video_path = base / video_path
            splits[row["split"]].append(
                VideoSample(path=video_path, label=row["label"], group=row["group"])
            )

    return dict(splits)


def _group_of(path: Path, pattern: re.Pattern[str] | None) -> str:
    if pattern is None:
        return path.stem
    match = pattern.search(path.stem)
    if match is None:
        logger.warning(
            "group pattern %r did not match %s; treating the file as its own recording",
            pattern.pattern,
            path.name,
        )
        return path.stem
    return match.group(1)


def _by_label(samples: Sequence[VideoSample]) -> dict[str, list[VideoSample]]:
    grouped: dict[str, list[VideoSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)
    return grouped


def _by_group(samples: Sequence[VideoSample]) -> dict[str, list[VideoSample]]:
    grouped: dict[str, list[VideoSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.group_key].append(sample)
    return grouped
