"""PyTorch datasets for clips decoded from video and for cached backbone features."""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.frame_extractor import sample_indices
from src.preprocessing.splits import VideoSample
from src.preprocessing.transforms import build_clip_transform, frames_to_tensor
from src.preprocessing.video_reader import (
    VideoReadError,
    count_frames,
    read_frames,
    read_metadata,
)
from src.utils.config import DatasetConfig, SamplingConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClipBatchItem:
    """One training example: a clip tensor, its label and its provenance."""

    clip: torch.Tensor
    label: int
    label_name: str
    video_path: Path
    frame_indices: tuple[int, ...]


class ClipDataset(Dataset[ClipBatchItem]):
    """Decode a fixed-length clip from each video listed in a split.

    Frame counts reported by containers are unreliable, so a video whose declared
    length is too optimistic is recounted once and then sampled again.
    """

    def __init__(
        self,
        samples: Sequence[VideoSample],
        class_to_index: dict[str, int],
        sampling: SamplingConfig,
        *,
        train: bool = False,
        seed: int = 42,
    ) -> None:
        if not samples:
            raise ValueError("samples must not be empty")
        unknown = {sample.label for sample in samples} - set(class_to_index)
        if unknown:
            raise ValueError(
                f"samples contain labels missing from class_to_index: {sorted(unknown)}"
            )

        self.samples = list(samples)
        self.class_to_index = dict(class_to_index)
        self.sampling = sampling
        self.train = train
        self.seed = seed
        self.transform = build_clip_transform(sampling.image_size, train=train)
        self._frame_counts: dict[Path, int] = {}
        self._epoch = 0

    @classmethod
    def from_config(
        cls,
        samples: Sequence[VideoSample],
        config: DatasetConfig,
        *,
        train: bool = False,
    ) -> ClipDataset:
        return cls(
            samples,
            config.class_to_index,
            config.sampling,
            train=train,
            seed=config.split.seed,
        )

    def set_epoch(self, epoch: int) -> None:
        """Change the sampling seed so random clips differ between epochs."""
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ClipBatchItem:
        sample = self.samples[index]
        num_frames = self._num_frames(sample.path)
        indices = self._sample_indices(num_frames, index)

        try:
            frames = read_frames(sample.path, indices)
        except VideoReadError:
            actual = count_frames(sample.path)
            if actual < 1:
                raise
            logger.warning(
                "%s reports %d frames but only %d could be decoded; resampling",
                sample.path,
                num_frames,
                actual,
            )
            self._frame_counts[sample.path] = actual
            indices = self._sample_indices(actual, index)
            frames = read_frames(sample.path, indices)

        clip = self.transform(frames_to_tensor(frames))
        return ClipBatchItem(
            clip=clip,
            label=self.class_to_index[sample.label],
            label_name=sample.label,
            video_path=sample.path,
            frame_indices=tuple(indices),
        )

    def _num_frames(self, path: Path) -> int:
        cached = self._frame_counts.get(path)
        if cached is not None:
            return cached
        metadata = read_metadata(path)
        total = metadata.frame_count if metadata.frame_count > 0 else count_frames(path)
        if total < 1:
            raise VideoReadError(f"no decodable frames in {path}")
        self._frame_counts[path] = total
        return total

    def _sample_indices(self, num_frames: int, index: int) -> list[int]:
        return sample_indices(
            num_frames,
            self.sampling.clip_length,
            stride=self.sampling.frame_stride,
            strategy=self.sampling.strategy,
            rng=random.Random((self.seed, self._epoch, index).__hash__()),
        )


class CachedFeatureDataset(Dataset[tuple[torch.Tensor, int]]):
    """Serve pre-computed backbone features so training does not decode video.

    Each entry is a ``(clip_length, feature_dim)`` array written by the feature
    cache builder, which makes an epoch cheap enough to run on a CPU.
    """

    def __init__(
        self,
        feature_paths: Sequence[Path],
        labels: Sequence[int],
        *,
        expected_dim: int | None = None,
    ) -> None:
        if len(feature_paths) != len(labels):
            raise ValueError(
                f"feature_paths and labels must have the same length, "
                f"got {len(feature_paths)} and {len(labels)}"
            )
        if not feature_paths:
            raise ValueError("feature_paths must not be empty")

        self.feature_paths = [Path(path) for path in feature_paths]
        self.labels = [int(label) for label in labels]
        self.expected_dim = expected_dim

    def __len__(self) -> int:
        return len(self.feature_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.feature_paths[index]
        features = np.load(path)
        if features.ndim != 2:
            raise ValueError(f"expected features with shape (T, D) in {path}, got {features.shape}")
        if self.expected_dim is not None and features.shape[1] != self.expected_dim:
            raise ValueError(
                f"{path} has feature dimension {features.shape[1]}, expected {self.expected_dim}"
            )
        return torch.from_numpy(features.astype(np.float32)), self.labels[index]


def collate_clips(items: Sequence[ClipBatchItem]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack clips and labels into batched tensors."""
    clips = torch.stack([item.clip for item in items])
    labels = torch.tensor([item.label for item in items], dtype=torch.long)
    return clips, labels
