import numpy as np
import pytest
import torch

from src.detection.detector import (
    Detection,
    Detector,
    YoloDetector,
    parse_yolo_result,
)
from src.utils.config import ConfigError, DetectionConfig


class FakeBoxes:
    def __init__(self, boxes, confidences, classes, *, as_tensor=True):
        wrap = torch.tensor if as_tensor else np.array
        self.xyxy = wrap(boxes, dtype=torch.float32 if as_tensor else np.float32)
        self.conf = wrap(confidences, dtype=torch.float32 if as_tensor else np.float32)
        self.cls = wrap(classes, dtype=torch.float32 if as_tensor else np.float32)

    def __len__(self):
        return len(self.xyxy)


class FakeResult:
    def __init__(self, boxes=None):
        self.boxes = boxes


class FakeModel:
    def __init__(self, results):
        self.results = results
        self.calls: list[dict] = []

    def predict(self, frames, **kwargs):
        self.calls.append({"frames": len(frames), **kwargs})
        return self.results


def person_result():
    return FakeResult(
        FakeBoxes(
            boxes=[[10, 20, 50, 120], [200, 40, 260, 180], [0, 0, 30, 30]],
            confidences=[0.6, 0.95, 0.9],
            classes=[0, 0, 2],
        )
    )


def test_detection_geometry():
    detection = Detection(x1=10, y1=20, x2=30, y2=60, confidence=0.8)

    assert detection.box == (10, 20, 30, 60)
    assert detection.width == 20
    assert detection.height == 40
    assert detection.area == 800
    assert detection.center == (20, 40)
    assert detection.to_int_box() == (10, 20, 30, 60)


def test_identical_boxes_have_full_overlap():
    box = Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.5)

    assert box.iou(box) == pytest.approx(1.0)


def test_iou_of_partially_overlapping_boxes():
    first = Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.5)
    second = Detection(x1=5, y1=0, x2=15, y2=10, confidence=0.5)

    assert first.iou(second) == pytest.approx(50 / 150)
    assert first.iou(second) == pytest.approx(second.iou(first))


def test_disjoint_and_touching_boxes_have_no_overlap():
    first = Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.5)
    apart = Detection(x1=20, y1=20, x2=30, y2=30, confidence=0.5)
    touching = Detection(x1=10, y1=0, x2=20, y2=10, confidence=0.5)

    assert first.iou(apart) == 0.0
    assert first.iou(touching) == 0.0


def test_boxes_are_clipped_to_the_frame():
    detection = Detection(x1=-15, y1=-8, x2=200, y2=150, confidence=0.7)

    clipped = detection.clipped_to(width=100, height=80)

    assert clipped.x1 == 0
    assert clipped.y1 == 0
    assert clipped.x2 == 100
    assert clipped.y2 == 80
    assert clipped.confidence == pytest.approx(0.7)


def test_clipping_keeps_a_box_that_starts_at_the_edge_usable():
    detection = Detection(x1=99, y1=79, x2=140, y2=120, confidence=0.7)

    clipped = detection.clipped_to(width=100, height=80)

    assert clipped.width >= 1
    assert clipped.height >= 1


def test_clipping_validates_the_frame_size():
    detection = Detection(x1=0, y1=0, x2=5, y2=5, confidence=0.5)

    with pytest.raises(ValueError, match="frame size"):
        detection.clipped_to(width=0, height=10)


@pytest.mark.parametrize(
    ("x1", "y1", "x2", "y2"),
    [(10, 0, 10, 5), (0, 10, 5, 10), (10, 10, 5, 5)],
)
def test_degenerate_boxes_are_rejected(x1, y1, x2, y2):
    with pytest.raises(ValueError, match="positive width and height"):
        Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.5)


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_confidence_outside_the_unit_range_is_rejected(confidence):
    with pytest.raises(ValueError, match="confidence"):
        Detection(x1=0, y1=0, x2=5, y2=5, confidence=confidence)


def test_parsing_keeps_only_people_and_orders_by_confidence():
    detections = parse_yolo_result(person_result(), person_class_id=0)

    assert len(detections) == 2
    assert [round(detection.confidence, 2) for detection in detections] == [0.95, 0.6]
    assert detections[0].box == (200, 40, 260, 180)


def test_parsing_respects_the_configured_person_class():
    detections = parse_yolo_result(person_result(), person_class_id=2)

    assert len(detections) == 1
    assert detections[0].box == (0, 0, 30, 30)


def test_parsing_applies_the_confidence_threshold():
    detections = parse_yolo_result(person_result(), person_class_id=0, confidence_threshold=0.8)

    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.95)


def test_parsing_caps_the_number_of_detections():
    detections = parse_yolo_result(person_result(), person_class_id=0, max_detections=1)

    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.95)


def test_parsing_accepts_numpy_arrays():
    result = FakeResult(
        FakeBoxes(boxes=[[1, 2, 9, 20]], confidences=[0.5], classes=[0], as_tensor=False)
    )

    detections = parse_yolo_result(result, person_class_id=0)

    assert len(detections) == 1
    assert detections[0].box == (1, 2, 9, 20)


def test_parsing_skips_degenerate_boxes_instead_of_failing():
    result = FakeResult(
        FakeBoxes(boxes=[[5, 5, 5, 20], [1, 1, 11, 21]], confidences=[0.9, 0.7], classes=[0, 0])
    )

    detections = parse_yolo_result(result, person_class_id=0)

    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.7)


def test_parsing_handles_frames_without_detections():
    assert parse_yolo_result(FakeResult(None)) == []
    assert parse_yolo_result(FakeResult(FakeBoxes([], [], []))) == []


def test_parsing_reports_inconsistent_result_arrays():
    boxes = FakeBoxes(boxes=[[0, 0, 10, 10], [0, 0, 20, 20]], confidences=[0.9], classes=[0, 0])

    with pytest.raises(ValueError, match="inconsistent result arrays"):
        parse_yolo_result(FakeResult(boxes))


def test_detector_passes_configured_thresholds_to_the_model():
    config = DetectionConfig(
        weights="yolov8n.pt",
        confidence_threshold=0.4,
        iou_threshold=0.6,
        image_size=320,
        max_detections=5,
    )
    model = FakeModel([person_result()])
    detector = YoloDetector(config, model=model)

    detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))

    call = model.calls[0]
    assert call["conf"] == pytest.approx(0.4)
    assert call["iou"] == pytest.approx(0.6)
    assert call["imgsz"] == 320
    assert call["max_det"] == 5
    assert call["classes"] == [0]
    assert call["verbose"] is False
    assert len(detections) == 2


def test_detector_returns_one_list_per_frame():
    model = FakeModel([person_result(), FakeResult(None)])
    detector = YoloDetector(model=model)

    results = detector.detect_batch([np.zeros((32, 32, 3), dtype=np.uint8)] * 2)

    assert len(results) == 2
    assert len(results[0]) == 2
    assert results[1] == []


def test_detector_validates_frame_shapes():
    detector = YoloDetector(model=FakeModel([person_result()]))

    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        detector.detect(np.zeros((32, 32), dtype=np.uint8))
    with pytest.raises(ValueError, match="must not be empty"):
        detector.detect_batch([])


def test_detector_satisfies_the_detector_protocol():
    detector = YoloDetector(model=FakeModel([person_result()]))

    assert isinstance(detector, Detector)


def test_repository_inference_config_has_valid_detection_settings():
    from src.utils.config import InferenceConfig

    config = InferenceConfig.from_yaml("configs/inference.yaml")

    assert config.detection.person_class_id == 0
    assert 0 < config.detection.confidence_threshold < 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weights", ""),
        ("person_class_id", -1),
        ("confidence_threshold", 0.0),
        ("confidence_threshold", 1.0),
        ("iou_threshold", 1.0),
        ("image_size", 16),
        ("max_detections", 0),
    ],
)
def test_detection_config_validates_its_fields(field, value):
    with pytest.raises(ConfigError, match=field.replace("_", "_")):
        DetectionConfig(**{field: value})
