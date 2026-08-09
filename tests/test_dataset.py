import numpy as np
import pytest
import torch

from src.preprocessing.dataset import CachedFeatureDataset, ClipDataset, collate_clips
from src.preprocessing.splits import discover_videos
from src.preprocessing.transforms import build_clip_transform, frames_to_tensor
from src.utils.config import SamplingConfig

CLASSES = ("walking", "falling")
CLASS_TO_INDEX = {"walking": 0, "falling": 1}
GROUP_PATTERN = r"_(g\d+)_"


@pytest.fixture
def samples(video_dataset):
    return discover_videos(video_dataset, CLASSES, group_pattern=GROUP_PATTERN)


def sampling(**overrides) -> SamplingConfig:
    settings = {"clip_length": 4, "frame_stride": 2, "strategy": "uniform", "image_size": 32}
    settings.update(overrides)
    return SamplingConfig(**settings)


def test_clip_dataset_returns_normalised_clip_tensors(samples):
    dataset = ClipDataset(samples, CLASS_TO_INDEX, sampling())

    item = dataset[0]

    assert len(dataset) == len(samples)
    assert item.clip.shape == (4, 3, 32, 32)
    assert item.clip.dtype == torch.float32
    assert item.label == CLASS_TO_INDEX[item.label_name]
    assert len(item.frame_indices) == 4
    assert item.clip.abs().max() > 0


def test_clip_dataset_is_deterministic_for_uniform_sampling(samples):
    dataset = ClipDataset(samples, CLASS_TO_INDEX, sampling())

    first = dataset[2]
    second = dataset[2]

    assert first.frame_indices == second.frame_indices
    assert torch.equal(first.clip, second.clip)


def test_random_sampling_changes_between_epochs(samples):
    dataset = ClipDataset(samples, CLASS_TO_INDEX, sampling(strategy="random", clip_length=3))

    dataset.set_epoch(0)
    first = [dataset[index].frame_indices for index in range(len(dataset))]
    dataset.set_epoch(1)
    second = [dataset[index].frame_indices for index in range(len(dataset))]

    assert first != second


def test_set_epoch_rejects_negative_values(samples):
    dataset = ClipDataset(samples, CLASS_TO_INDEX, sampling())

    with pytest.raises(ValueError, match="epoch"):
        dataset.set_epoch(-1)


def test_clip_dataset_rejects_labels_outside_the_taxonomy(samples):
    with pytest.raises(ValueError, match="missing from class_to_index"):
        ClipDataset(samples, {"walking": 0}, sampling())


def test_clip_dataset_rejects_empty_samples():
    with pytest.raises(ValueError, match="must not be empty"):
        ClipDataset([], CLASS_TO_INDEX, sampling())


def test_short_video_is_padded_to_the_clip_length(tmp_path, make_video):
    make_video(tmp_path / "walking" / "v_walking_g01_c01.avi", 3)
    make_video(tmp_path / "falling" / "v_falling_g01_c01.avi", 3)
    samples = discover_videos(tmp_path, CLASSES, group_pattern=GROUP_PATTERN)
    dataset = ClipDataset(samples, CLASS_TO_INDEX, sampling(clip_length=8, strategy="center"))

    item = dataset[0]

    assert item.clip.shape[0] == 8
    assert max(item.frame_indices) <= 2


def test_train_transform_keeps_the_clip_shape(samples):
    train_dataset = ClipDataset(samples, CLASS_TO_INDEX, sampling(), train=True)

    first = train_dataset[0].clip
    second = train_dataset[0].clip

    assert first.shape == second.shape == (4, 3, 32, 32)


def test_train_transform_augments_differently_on_each_call():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(4, 48, 48, 3), dtype=np.uint8)
    transform = build_clip_transform(32, train=True)

    first = transform(frames_to_tensor(frames))
    second = transform(frames_to_tensor(frames))

    assert first.shape == second.shape
    assert not torch.equal(first, second)


def test_collate_clips_stacks_batches(samples):
    dataset = ClipDataset(samples, CLASS_TO_INDEX, sampling())

    clips, labels = collate_clips([dataset[0], dataset[1]])

    assert clips.shape == (2, 4, 3, 32, 32)
    assert labels.shape == (2,)
    assert labels.dtype == torch.long


def test_clip_transform_applies_one_flip_to_the_whole_clip():
    frames = np.zeros((5, 8, 8, 3), dtype=np.uint8)
    frames[:, :, 0] = 255
    transform = build_clip_transform(8, train=True)

    clip = transform(frames_to_tensor(frames))

    left_edges = clip[:, 0, :, 0]
    right_edges = clip[:, 0, :, -1]
    assert torch.allclose(left_edges, left_edges[0].expand_as(left_edges))
    assert torch.allclose(right_edges, right_edges[0].expand_as(right_edges))


def test_frames_to_tensor_rejects_unexpected_shapes():
    with pytest.raises(ValueError, match="expected frames"):
        frames_to_tensor(np.zeros((4, 8, 8), dtype=np.uint8))


def test_build_clip_transform_rejects_invalid_size():
    with pytest.raises(ValueError, match="image_size"):
        build_clip_transform(0, train=False)


def test_cached_feature_dataset_loads_arrays(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"clip{index}.npy"
        np.save(path, np.full((4, 8), index, dtype=np.float32))
        paths.append(path)

    dataset = CachedFeatureDataset(paths, [0, 1, 0], expected_dim=8)
    features, label = dataset[1]

    assert len(dataset) == 3
    assert features.shape == (4, 8)
    assert features.dtype == torch.float32
    assert label == 1


def test_cached_feature_dataset_validates_feature_dimension(tmp_path):
    path = tmp_path / "clip.npy"
    np.save(path, np.zeros((4, 8), dtype=np.float32))
    dataset = CachedFeatureDataset([path], [0], expected_dim=16)

    with pytest.raises(ValueError, match="feature dimension"):
        dataset[0]


def test_cached_feature_dataset_rejects_non_sequence_arrays(tmp_path):
    path = tmp_path / "clip.npy"
    np.save(path, np.zeros((4,), dtype=np.float32))
    dataset = CachedFeatureDataset([path], [0])

    with pytest.raises(ValueError, match=r"shape \(T, D\)"):
        dataset[0]


def test_cached_feature_dataset_requires_matching_lengths(tmp_path):
    with pytest.raises(ValueError, match="same length"):
        CachedFeatureDataset([tmp_path / "a.npy"], [0, 1])


def test_cached_feature_dataset_rejects_empty_input():
    with pytest.raises(ValueError, match="must not be empty"):
        CachedFeatureDataset([], [])
