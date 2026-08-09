"""Frame index selection for fixed-length clips.

These helpers only compute indices, which keeps sampling policy independent of
video decoding and cheap to test. Videos shorter than the requested span are
padded by repeating the last available frame, so every clip has the same length.
"""

from __future__ import annotations

import random

UNIFORM = "uniform"
CENTER = "center"
RANDOM = "random"


def clip_span(clip_length: int, stride: int) -> int:
    """Number of source frames covered by a strided clip."""
    _validate(clip_length=clip_length, stride=stride)
    return (clip_length - 1) * stride + 1


def sample_indices(
    num_frames: int,
    clip_length: int,
    *,
    stride: int = 1,
    strategy: str = UNIFORM,
    rng: random.Random | None = None,
) -> list[int]:
    """Select ``clip_length`` frame indices from a video of ``num_frames`` frames.

    ``uniform`` spreads indices across the whole video and ignores ``stride``.
    ``center`` takes the strided window at the middle of the video.
    ``random`` takes a strided window starting at a random offset.
    """
    _validate(num_frames=num_frames, clip_length=clip_length, stride=stride)

    if strategy == UNIFORM:
        if clip_length == 1:
            return [(num_frames - 1) // 2]
        step = (num_frames - 1) / (clip_length - 1)
        return [min(num_frames - 1, round(position * step)) for position in range(clip_length)]

    span = clip_span(clip_length, stride)
    latest_start = max(0, num_frames - span)

    if strategy == CENTER:
        start = latest_start // 2
    elif strategy == RANDOM:
        generator = rng or random
        start = generator.randint(0, latest_start)
    else:
        raise ValueError(
            f"strategy must be one of ['{UNIFORM}', '{CENTER}', '{RANDOM}'], got {strategy!r}"
        )

    return [min(start + offset * stride, num_frames - 1) for offset in range(clip_length)]


def sliding_windows(
    num_frames: int,
    clip_length: int,
    *,
    stride: int = 1,
    window_stride: int | None = None,
) -> list[list[int]]:
    """Split a video into consecutive clips for inference over its full duration.

    ``window_stride`` is the distance in source frames between window starts and
    defaults to non-overlapping windows. The final window is aligned to the end
    of the video so that trailing frames are still analysed.
    """
    _validate(num_frames=num_frames, clip_length=clip_length, stride=stride)

    span = clip_span(clip_length, stride)
    step = span if window_stride is None else window_stride
    if step < 1:
        raise ValueError(f"window_stride must be >= 1, got {window_stride}")

    if num_frames <= span:
        return [[min(offset * stride, num_frames - 1) for offset in range(clip_length)]]

    starts: list[int] = list(range(0, num_frames - span + 1, step))
    last_start = num_frames - span
    if starts[-1] != last_start:
        starts.append(last_start)

    return [[start + offset * stride for offset in range(clip_length)] for start in starts]


def _validate(*, num_frames: int | None = None, clip_length: int, stride: int) -> None:
    if num_frames is not None and num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if clip_length < 1:
        raise ValueError(f"clip_length must be >= 1, got {clip_length}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
