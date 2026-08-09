import json

import cv2
import numpy as np
import pytest
import torch

from src.detection.detector import Detection
from src.features.feature_extractor import build_backbone
from src.inference.predict import (
    ClipRecognizer,
    VideoAnalyzer,
    _thin,
    main,
)
from src.models.classifier import BehaviorClassifier
from src.models.cnn_lstm import TemporalClassifier
from src.utils.config import (
    ClipSelectionConfig,
    InferenceConfig,
    ModelConfig,
    SamplingConfig,
    TrackingConfig,
)

CLASSES = ("walking", "falling")
ABNORMAL = ("falling",)
WIDTH = 96
HEIGHT = 96
FPS = 10.0
FRAMES = 20


class BrightRegionDetector:
    """Stands in for YOLO by locating the bright rectangle in a synthetic frame."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        mask = frame.max(axis=2) > 150
        if not mask.any():
            return []
        rows, columns = np.where(mask)
        return [
            Detection(
                x1=float(columns.min()),
                y1=float(rows.min()),
                x2=float(columns.max() + 1),
                y2=float(rows.max() + 1),
                confidence=0.9,
            )
        ]


class BlindDetector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return []


@pytest.fixture
def moving_person_video(tmp_path):
    path = tmp_path / "person.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV build cannot write MJPG AVI files")

    for index in range(FRAMES):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        left = 4 + index * 3
        frame[20:70, left : left + 14] = 255
        writer.write(frame)
    writer.release()

    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("OpenCV did not produce a readable video file")
    return path


def model_config() -> ModelConfig:
    return ModelConfig.from_mapping(
        {
            "backbone": {"name": "resnet18", "pretrained": False, "freeze": True},
            "temporal": {"hidden_size": 8, "num_layers": 1, "bidirectional": False, "dropout": 0.0},
            "classifier": {"dropout": 0.0},
        }
    )


def sampling() -> SamplingConfig:
    return SamplingConfig(clip_length=4, frame_stride=1, strategy="uniform", image_size=32)


def inference_config(**clip_overrides) -> InferenceConfig:
    clips = {"box_padding": 0.1, "max_clips_per_track": 3}
    clips.update(clip_overrides)
    return InferenceConfig(
        tracking=TrackingConfig(iou_threshold=0.2, max_age=3, min_hits=2, min_track_length=6),
        clips=ClipSelectionConfig(**clips),
    )


@pytest.fixture(scope="module")
def recognizer():
    config = model_config()
    backbone, feature_dim = build_backbone(config.backbone)
    head = TemporalClassifier(feature_dim, len(CLASSES), config)
    return ClipRecognizer(backbone, head, device=torch.device("cpu"))


def build_analyzer(recognizer, config=None):
    return VideoAnalyzer(
        BrightRegionDetector(),
        recognizer,
        BehaviorClassifier(CLASSES, ABNORMAL),
        sampling(),
        config or inference_config(),
        batch_size=2,
    )


def test_analysis_reports_one_person_with_timestamps(moving_person_video, recognizer):
    analysis = build_analyzer(recognizer).analyze(moving_person_video)

    assert len(analysis.people) == 1
    person = analysis.people[0]
    assert person.track_id == 1
    assert person.frames_tracked >= 6
    assert person.start_time == pytest.approx(person.first_frame / FPS)
    assert person.end_time == pytest.approx(person.last_frame / FPS)
    assert person.end_time > person.start_time
    assert person.decision.action in CLASSES
    assert 0.0 <= person.decision.anomaly_score <= 1.0


def test_video_properties_are_reported(moving_person_video, recognizer):
    analysis = build_analyzer(recognizer).analyze(moving_person_video)

    assert analysis.fps == pytest.approx(FPS, rel=0.05)
    assert analysis.frame_count == FRAMES
    assert analysis.duration_seconds == pytest.approx(FRAMES / FPS, rel=0.05)
    assert analysis.classes == CLASSES


def test_clip_budget_is_respected(moving_person_video, recognizer):
    analysis = build_analyzer(recognizer, inference_config(max_clips_per_track=2)).analyze(
        moving_person_video
    )

    person = analysis.people[0]
    assert 1 <= len(person.clips) <= 2
    for clip in person.clips:
        assert len(clip.frame_indices) == 4
        assert clip.end_time >= clip.start_time
        assert sum(clip.probabilities) == pytest.approx(1.0, abs=1e-5)


def test_clips_cover_frames_the_person_was_tracked_in(moving_person_video, recognizer):
    analysis = build_analyzer(recognizer).analyze(moving_person_video)

    person = analysis.people[0]
    for clip in person.clips:
        assert clip.frame_indices[0] >= person.first_frame
        assert clip.frame_indices[-1] <= person.last_frame


def test_peak_window_is_the_most_anomalous_clip(moving_person_video, recognizer):
    analysis = build_analyzer(recognizer).analyze(moving_person_video)

    person = analysis.people[0]
    assert person.peak_clip.decision.anomaly_score == max(
        clip.decision.anomaly_score for clip in person.clips
    )


def test_aggregated_decision_uses_all_clips(moving_person_video, recognizer):
    analysis = build_analyzer(recognizer).analyze(moving_person_video)

    person = analysis.people[0]
    expected = np.mean([clip.probabilities for clip in person.clips], axis=0)
    assert person.decision.anomaly_score == pytest.approx(
        float(expected[CLASSES.index("falling")]), abs=1e-4
    )


def test_report_dictionary_is_json_serialisable(moving_person_video, recognizer):
    analysis = build_analyzer(recognizer).analyze(moving_person_video)

    payload = json.loads(json.dumps(analysis.to_dict()))

    assert payload["people_detected"] == 1
    assert payload["classes"] == list(CLASSES)
    person = payload["people"][0]
    assert set(person) >= {
        "person_id",
        "action",
        "confidence",
        "status",
        "anomaly_score",
        "risk_level",
        "peak_window",
        "clips",
    }
    assert set(person["clips"][0]["probabilities"]) == set(CLASSES)


def test_video_without_people_produces_an_empty_report(moving_person_video, recognizer):
    analyzer = VideoAnalyzer(
        BlindDetector(),
        recognizer,
        BehaviorClassifier(CLASSES, ABNORMAL),
        sampling(),
        inference_config(),
    )

    analysis = analyzer.analyze(moving_person_video)

    assert analysis.people == ()
    assert analysis.abnormal_people == ()
    assert analysis.max_anomaly_score == 0.0
    assert analysis.to_dict()["people_detected"] == 0


def test_frame_stride_reduces_the_frames_examined(moving_person_video, recognizer):
    analyzer = build_analyzer(recognizer)

    dense = analyzer.analyze(moving_person_video, frame_stride=1)
    sparse = analyzer.analyze(moving_person_video, frame_stride=2)

    assert dense.people[0].frames_tracked > sparse.people[0].frames_tracked


def test_frame_stride_must_be_positive(moving_person_video, recognizer):
    with pytest.raises(ValueError, match="frame_stride"):
        build_analyzer(recognizer).analyze(moving_person_video, frame_stride=0)


def test_batch_size_must_be_positive(recognizer):
    with pytest.raises(ValueError, match="batch_size"):
        VideoAnalyzer(
            BrightRegionDetector(),
            recognizer,
            BehaviorClassifier(CLASSES, ABNORMAL),
            sampling(),
            batch_size=0,
        )


def test_recognizer_validates_clip_rank(recognizer):
    with pytest.raises(ValueError, match=r"\(B, T, C, H, W\)"):
        recognizer.probabilities(torch.zeros(4, 3, 32, 32, dtype=torch.uint8))


def test_recognizer_returns_normalised_probabilities(recognizer):
    clips = torch.zeros(2, 4, 3, 32, 32, dtype=torch.uint8)

    probabilities = recognizer.probabilities(clips)

    assert probabilities.shape == (2, len(CLASSES))
    assert probabilities.sum(dim=1).allclose(torch.ones(2), atol=1e-5)


def test_recognizer_rebuilds_from_a_checkpoint(tmp_path):
    config = model_config()
    _, feature_dim = build_backbone(config.backbone)
    head = TemporalClassifier(feature_dim, len(CLASSES), config)
    checkpoint = tmp_path / "run.pt"
    torch.save(
        {
            "model_state": head.state_dict(),
            "classes": list(CLASSES),
            "feature_dim": feature_dim,
            "model_config": config.to_mapping(),
        },
        checkpoint,
    )

    restored, classes, restored_config = ClipRecognizer.from_checkpoint(checkpoint)

    assert classes == CLASSES
    assert restored_config.temporal.hidden_size == config.temporal.hidden_size
    assert restored.probabilities(torch.zeros(1, 4, 3, 32, 32, dtype=torch.uint8)).shape == (
        1,
        len(CLASSES),
    )


def test_checkpoint_with_mismatched_feature_dimension_is_refused(tmp_path):
    config = model_config()
    head = TemporalClassifier(8, len(CLASSES), config)
    checkpoint = tmp_path / "mismatch.pt"
    torch.save(
        {
            "model_state": head.state_dict(),
            "classes": list(CLASSES),
            "feature_dim": 8,
            "model_config": config.to_mapping(),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="backbone produces"):
        ClipRecognizer.from_checkpoint(checkpoint)


def test_checkpoint_without_architecture_is_refused(tmp_path):
    config = model_config()
    _, feature_dim = build_backbone(config.backbone)
    head = TemporalClassifier(feature_dim, len(CLASSES), config)
    checkpoint = tmp_path / "bare.pt"
    torch.save(
        {
            "model_state": head.state_dict(),
            "classes": list(CLASSES),
            "feature_dim": feature_dim,
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="does not record its architecture"):
        ClipRecognizer.from_checkpoint(checkpoint)


def test_missing_checkpoint_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        ClipRecognizer.from_checkpoint(tmp_path / "absent.pt")


def test_incomplete_checkpoint_is_reported(tmp_path):
    checkpoint = tmp_path / "partial.pt"
    torch.save({"model_state": {}}, checkpoint)

    with pytest.raises(ValueError, match="missing keys"):
        ClipRecognizer.from_checkpoint(checkpoint)


def test_window_thinning_spreads_the_selection():
    windows = [[index] for index in range(10)]

    chosen = _thin(windows, 3)

    assert chosen == [[0], [4], [9]]


def test_window_thinning_keeps_short_lists_untouched():
    windows = [[0], [1]]

    assert _thin(windows, 5) == windows


def test_window_thinning_requires_a_positive_limit():
    with pytest.raises(ValueError, match="limit"):
        _thin([[0]], 0)


def test_cli_refuses_a_class_mismatch(tmp_path, moving_person_video):
    import yaml

    config = model_config()
    _, feature_dim = build_backbone(config.backbone)
    head = TemporalClassifier(feature_dim, len(CLASSES), config)
    checkpoint = tmp_path / "run.pt"
    torch.save(
        {
            "model_state": head.state_dict(),
            "classes": ["walking", "running"],
            "feature_dim": feature_dim,
            "model_config": config.to_mapping(),
        },
        checkpoint,
    )

    dataset_config = tmp_path / "dataset.yaml"
    dataset_config.write_text(
        yaml.safe_dump(
            {
                "name": "cli_mismatch",
                "root": str(tmp_path / "raw"),
                "classes": list(CLASSES),
                "behavior_map": {"normal": ["walking"], "abnormal": ["falling"]},
                "split": {"train": 0.7, "validation": 0.15, "test": 0.15},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="was trained on classes"):
        main(
            [
                "--video",
                str(moving_person_video),
                "--checkpoint",
                str(checkpoint),
                "--dataset-config",
                str(dataset_config),
                "--device",
                "cpu",
            ]
        )
