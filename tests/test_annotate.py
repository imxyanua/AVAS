import cv2
import numpy as np
import pytest

from src.detection.detector import Detection
from src.inference.annotate import (
    build_frame_overlays,
    colour_for_person,
    draw_overlays,
    label_for_person,
    people_table_rows,
    write_annotated_video,
)
from src.inference.predict import PersonAnalysis, VideoAnalysis
from src.models.classifier import BehaviorDecision
from src.tracking.tracker import Track


def _decision(action: str = "falling", *, abnormal: bool = True) -> BehaviorDecision:
    return BehaviorDecision(
        action=action,
        confidence=0.91,
        is_abnormal=abnormal,
        anomaly_score=0.88 if abnormal else 0.12,
        risk_level="high" if abnormal else "normal",
    )


def _person(track_id: int = 1) -> PersonAnalysis:
    from src.inference.predict import ClipPrediction

    clip = ClipPrediction(
        frame_indices=(0, 1, 2, 3),
        start_time=0.0,
        end_time=0.3,
        decision=_decision(),
        probabilities=(0.09, 0.91),
    )
    return PersonAnalysis(
        track_id=track_id,
        first_frame=0,
        last_frame=3,
        start_time=0.0,
        end_time=0.3,
        frames_tracked=4,
        decision=_decision(),
        clips=(clip,),
    )


def _track(track_id: int = 1) -> Track:
    detection = Detection(x1=10, y1=20, x2=40, y2=80, confidence=0.9)
    track = Track(track_id=track_id, detection=detection, first_frame=0, last_frame=0)
    track.history.clear()
    for index in range(4):
        track.update(index, Detection(x1=10 + index, y1=20, x2=40 + index, y2=80, confidence=0.9))
    return track


def test_build_frame_overlays_uses_recognition_labels():
    overlays = build_frame_overlays([_track()], [_person()])
    assert set(overlays) == {0, 1, 2, 3}
    first = overlays[0][0]
    assert first.track_id == 1
    assert "falling" in first.text
    assert first.colour == colour_for_person(_person())


def test_draw_overlays_returns_modified_copy():
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    overlays = build_frame_overlays([_track()], [_person()])[0]
    drawn = draw_overlays(frame, overlays)
    assert drawn is not frame
    assert drawn.shape == frame.shape
    assert drawn.sum() > 0


def test_people_table_rows_flattens_peak_window():
    rows = people_table_rows(
        VideoAnalysis(
            video=__import__("pathlib").Path("demo.avi"),
            fps=10.0,
            frame_count=4,
            duration_seconds=0.4,
            classes=("walking", "falling"),
            people=(_person(),),
            tracks=(_track(),),
        )
    )
    assert rows[0]["person_id"] == 1
    assert rows[0]["action"] == "falling"
    assert rows[0]["peak_start"] == 0.0


def test_write_annotated_video(tmp_path):
    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 96))
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV build cannot write MJPG AVI files")
    for _ in range(4):
        writer.write(np.zeros((96, 96, 3), dtype=np.uint8))
    writer.release()

    analysis = VideoAnalysis(
        video=video,
        fps=10.0,
        frame_count=4,
        duration_seconds=0.4,
        classes=("walking", "falling"),
        people=(_person(),),
        tracks=(_track(),),
    )
    output = write_annotated_video(analysis, tmp_path / "annotated.avi")
    assert output.is_file()
    assert output.stat().st_size > 0


def test_label_for_person_includes_identity_and_scores():
    text = label_for_person(_person())
    assert "ID 1" in text
    assert "falling" in text
    assert "abnormal" in text
