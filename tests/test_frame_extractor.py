import random
from itertools import pairwise

import pytest

from src.preprocessing.frame_extractor import (
    clip_span,
    sample_indices,
    sliding_windows,
)


@pytest.mark.parametrize("strategy", ["uniform", "center", "random"])
def test_indices_have_requested_length_and_stay_in_range(strategy):
    indices = sample_indices(100, clip_length=16, stride=2, strategy=strategy, rng=random.Random(0))

    assert len(indices) == 16
    assert all(0 <= index < 100 for index in indices)
    assert indices == sorted(indices)


def test_uniform_sampling_covers_first_and_last_frame():
    indices = sample_indices(50, clip_length=10, strategy="uniform")

    assert indices[0] == 0
    assert indices[-1] == 49


def test_uniform_sampling_of_single_frame_takes_the_middle():
    assert sample_indices(9, clip_length=1, strategy="uniform") == [4]


def test_center_sampling_leaves_equal_margins():
    indices = sample_indices(21, clip_length=4, stride=2, strategy="center")

    assert indices == [7, 9, 11, 13]
    assert indices[0] == 20 - indices[-1]


def test_short_videos_are_padded_with_the_last_frame():
    indices = sample_indices(5, clip_length=8, stride=3, strategy="center")

    assert len(indices) == 8
    assert max(indices) == 4
    assert indices[-1] == 4
    assert indices.count(4) > 1


def test_random_sampling_is_reproducible_for_a_given_seed():
    first = sample_indices(200, clip_length=8, stride=2, strategy="random", rng=random.Random(11))
    second = sample_indices(200, clip_length=8, stride=2, strategy="random", rng=random.Random(11))
    other = sample_indices(200, clip_length=8, stride=2, strategy="random", rng=random.Random(12))

    assert first == second
    assert first != other


def test_random_sampling_keeps_the_configured_stride():
    indices = sample_indices(120, clip_length=6, stride=4, strategy="random", rng=random.Random(3))

    steps = {second - first for first, second in pairwise(indices)}
    assert steps == {4}


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="strategy"):
        sample_indices(30, clip_length=4, strategy="newest")


@pytest.mark.parametrize(
    ("num_frames", "clip_length", "stride"),
    [(0, 4, 1), (10, 0, 1), (10, 4, 0)],
)
def test_invalid_arguments_are_rejected(num_frames, clip_length, stride):
    with pytest.raises(ValueError):
        sample_indices(num_frames, clip_length=clip_length, stride=stride)


def test_clip_span_accounts_for_stride():
    assert clip_span(16, 1) == 16
    assert clip_span(16, 2) == 31


def test_sliding_windows_are_non_overlapping_by_default():
    windows = sliding_windows(64, clip_length=8, stride=1)

    assert [window[0] for window in windows] == [0, 8, 16, 24, 32, 40, 48, 56]
    assert all(len(window) == 8 for window in windows)


def test_sliding_windows_cover_the_end_of_the_video():
    windows = sliding_windows(70, clip_length=8, stride=1)

    assert windows[-1][-1] == 69


def test_sliding_windows_support_overlap():
    windows = sliding_windows(32, clip_length=8, stride=1, window_stride=4)

    assert [window[0] for window in windows] == [0, 4, 8, 12, 16, 20, 24]


def test_sliding_windows_pads_videos_shorter_than_one_clip():
    windows = sliding_windows(4, clip_length=8, stride=2)

    assert len(windows) == 1
    assert len(windows[0]) == 8
    assert max(windows[0]) == 3


def test_sliding_windows_reject_invalid_window_stride():
    with pytest.raises(ValueError, match="window_stride"):
        sliding_windows(50, clip_length=8, stride=1, window_stride=0)
