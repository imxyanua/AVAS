import cv2
import numpy as np
import pytest

from src.preprocessing.video_reader import (
    VideoReadError,
    count_frames,
    iter_frames,
    read_frames,
    read_metadata,
)

FRAME_COUNT = 12
WIDTH = 64
HEIGHT = 48
FPS = 10.0


def frame_colour(index: int) -> tuple[int, int, int]:
    """Distinct RGB colour per frame index, robust to lossy compression."""
    return (20 * index, 255 - 20 * index, 10 * index)


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("videos") / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV build cannot write MJPG AVI files")

    for index in range(FRAME_COUNT):
        red, green, blue = frame_colour(index)
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :] = (blue, green, red)
        writer.write(frame)
    writer.release()

    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("OpenCV did not produce a readable video file")
    return path


def test_read_metadata_reports_resolution_and_rate(sample_video):
    metadata = read_metadata(sample_video)

    assert metadata.width == WIDTH
    assert metadata.height == HEIGHT
    assert metadata.frame_count == FRAME_COUNT
    assert metadata.fps == pytest.approx(FPS, rel=0.05)
    assert metadata.duration_seconds == pytest.approx(FRAME_COUNT / FPS, rel=0.05)


def test_count_frames_matches_the_written_frames(sample_video):
    assert count_frames(sample_video) == FRAME_COUNT


def test_read_frames_returns_rgb_in_requested_order(sample_video):
    frames = read_frames(sample_video, [5, 0])

    assert frames.shape == (2, HEIGHT, WIDTH, 3)
    assert frames.dtype == np.uint8
    first_pixel = frames[0, 0, 0].astype(int)
    second_pixel = frames[1, 0, 0].astype(int)
    assert np.allclose(first_pixel, frame_colour(5), atol=12)
    assert np.allclose(second_pixel, frame_colour(0), atol=12)


def test_read_frames_allows_repeated_indices(sample_video):
    frames = read_frames(sample_video, [3, 3, 3])

    assert frames.shape[0] == 3
    assert np.array_equal(frames[0], frames[2])


def test_read_frames_rejects_indices_beyond_the_stream(sample_video):
    with pytest.raises(VideoReadError, match="missing indices"):
        read_frames(sample_video, [FRAME_COUNT + 5])


def test_read_frames_rejects_empty_and_negative_indices(sample_video):
    with pytest.raises(ValueError, match="must not be empty"):
        read_frames(sample_video, [])
    with pytest.raises(ValueError, match="non-negative"):
        read_frames(sample_video, [-1])


def test_iter_frames_walks_the_whole_stream(sample_video):
    indices = [index for index, _ in iter_frames(sample_video)]

    assert indices == list(range(FRAME_COUNT))


def test_iter_frames_honours_the_step(sample_video):
    indices = [index for index, _ in iter_frames(sample_video, step=4)]

    assert indices == [0, 4, 8]


def test_iter_frames_rejects_invalid_step(sample_video):
    with pytest.raises(ValueError, match="step"):
        next(iter_frames(sample_video, step=0))


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(VideoReadError, match="not found"):
        read_metadata(tmp_path / "absent.avi")


def test_unreadable_file_is_reported(tmp_path):
    broken = tmp_path / "broken.avi"
    broken.write_bytes(b"not a video")

    with pytest.raises(VideoReadError, match="could not open video"):
        read_metadata(broken)
