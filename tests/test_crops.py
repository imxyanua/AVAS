import numpy as np
import pytest

from src.detection.detector import Detection
from src.inference.crops import crop_person, expand_box, letterbox


def frame(width: int = 200, height: int = 150) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(width, dtype=np.uint8)
    return image


def test_expand_box_grows_on_every_side():
    detection = Detection(x1=50, y1=40, x2=90, y2=120, confidence=0.9)

    grown = expand_box(detection, frame_width=200, frame_height=150, padding=0.25)

    assert grown.x1 == pytest.approx(40)
    assert grown.x2 == pytest.approx(100)
    assert grown.y1 == pytest.approx(20)
    assert grown.y2 == pytest.approx(140)


def test_expanded_box_stays_inside_the_frame():
    detection = Detection(x1=5, y1=5, x2=45, y2=105, confidence=0.9)

    grown = expand_box(detection, frame_width=200, frame_height=150, padding=0.5)

    assert grown.x1 >= 0
    assert grown.y1 >= 0
    assert grown.x2 <= 200
    assert grown.y2 <= 150


def test_expand_box_with_zero_padding_only_clips():
    detection = Detection(x1=10, y1=10, x2=50, y2=60, confidence=0.9)

    grown = expand_box(detection, frame_width=200, frame_height=150, padding=0.0)

    assert grown.box == detection.box


def test_expand_box_rejects_negative_padding():
    detection = Detection(x1=10, y1=10, x2=50, y2=60, confidence=0.9)

    with pytest.raises(ValueError, match="padding"):
        expand_box(detection, 200, 150, -0.1)


def test_crop_is_square_and_sized_for_the_model():
    detection = Detection(x1=40, y1=20, x2=80, y2=120, confidence=0.9)

    crop = crop_person(frame(), detection, padding=0.1, size=64)

    assert crop.shape == (64, 64, 3)
    assert crop.dtype == np.uint8


def test_tall_person_keeps_its_proportions():
    tall = np.zeros((150, 200, 3), dtype=np.uint8)
    tall[20:120, 40:60] = 255
    detection = Detection(x1=40, y1=20, x2=60, y2=120, confidence=0.9)

    crop = crop_person(tall, detection, padding=0.0, size=100)

    filled_columns = np.where(crop.any(axis=(0, 2)))[0]
    filled_rows = np.where(crop.any(axis=(1, 2)))[0]
    aspect = (filled_rows[-1] - filled_rows[0] + 1) / (filled_columns[-1] - filled_columns[0] + 1)
    assert aspect == pytest.approx(5.0, rel=0.15)


def test_letterbox_centres_the_patch_and_pads_the_rest():
    patch = np.full((40, 20, 3), 255, dtype=np.uint8)

    canvas = letterbox(patch, 80)

    assert canvas.shape == (80, 80, 3)
    assert canvas[:, 0].sum() == 0
    assert canvas[:, -1].sum() == 0
    assert canvas[40, 40].sum() > 0


def test_letterbox_keeps_a_square_patch_full_frame():
    patch = np.full((30, 30, 3), 200, dtype=np.uint8)

    canvas = letterbox(patch, 60)

    assert canvas.shape == (60, 60, 3)
    assert (canvas > 0).all()


def test_letterbox_upscales_small_patches():
    patch = np.full((4, 2, 3), 128, dtype=np.uint8)

    canvas = letterbox(patch, 32)

    assert canvas.shape == (32, 32, 3)
    assert canvas.any()


def test_crop_validates_frame_and_size():
    detection = Detection(x1=10, y1=10, x2=50, y2=60, confidence=0.9)

    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        crop_person(np.zeros((100, 100), dtype=np.uint8), detection)
    with pytest.raises(ValueError, match="size"):
        crop_person(frame(), detection, size=0)


def test_letterbox_validates_the_patch_shape():
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        letterbox(np.zeros((10, 10), dtype=np.uint8), 32)


def test_crop_of_a_box_at_the_frame_edge_is_still_usable():
    detection = Detection(x1=190, y1=140, x2=260, y2=220, confidence=0.9)

    crop = crop_person(frame(), detection, padding=0.2, size=48)

    assert crop.shape == (48, 48, 3)
