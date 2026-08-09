import pytest

from src.detection.detector import Detection
from src.tracking.tracker import IouTracker, track_video, usable_tracks
from src.utils.config import ConfigError, TrackingConfig


def box(x: float, y: float, *, size: float = 40.0, confidence: float = 0.9) -> Detection:
    return Detection(x1=x, y1=y, x2=x + size, y2=y + size * 2, confidence=confidence)


def walking_person(frames: int, *, start: float = 0.0, step: float = 5.0) -> list[list[Detection]]:
    return [[box(start + index * step, 50)] for index in range(frames)]


def tracker(**overrides) -> IouTracker:
    settings = {"iou_threshold": 0.3, "max_age": 3, "min_hits": 2, "min_track_length": 4}
    settings.update(overrides)
    return IouTracker(TrackingConfig(**settings))


def test_one_person_keeps_a_single_identity():
    tracks = track_video(tracker(), walking_person(10))

    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].length == 10
    assert tracks[0].first_frame == 0
    assert tracks[0].last_frame == 9


def test_two_people_far_apart_get_separate_identities():
    frames = [[box(0 + index * 4, 50), box(300 - index * 4, 60)] for index in range(8)]

    tracks = track_video(tracker(), frames)

    assert len(tracks) == 2
    assert {track.track_id for track in tracks} == {1, 2}
    assert all(track.length == 8 for track in tracks)


def test_track_is_reported_only_after_the_minimum_number_of_hits():
    instance = tracker(min_hits=3)

    first = instance.update(0, [box(0, 50)])
    second = instance.update(1, [box(2, 50)])
    third = instance.update(2, [box(4, 50)])

    assert first == []
    assert second == []
    assert len(third) == 1
    assert third[0].track_id == 1


def test_single_frame_false_positive_never_becomes_a_track():
    instance = tracker(min_hits=3, max_age=1)

    instance.update(0, [box(500, 400)])
    instance.update(1, [])
    instance.update(2, [])
    instance.update(3, [])

    assert instance.finalize() == []


def test_identity_survives_a_short_occlusion():
    instance = tracker(max_age=3, min_hits=2)
    frames = [[box(0, 50)], [box(5, 50)], [], [], [box(20, 50)], [box(25, 50)]]

    for index, detections in enumerate(frames):
        instance.update(index, detections)
    tracks = instance.finalize()

    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].length == 4


def test_a_long_absence_starts_a_new_identity():
    instance = tracker(max_age=2, min_hits=1)
    frames = [[box(0, 50)], [box(5, 50)], [], [], [], [box(10, 50)], [box(15, 50)]]

    for index, detections in enumerate(frames):
        instance.update(index, detections)
    tracks = instance.finalize()

    assert len(tracks) == 2
    assert {track.track_id for track in tracks} == {1, 2}


def test_fast_movement_beyond_the_overlap_threshold_breaks_the_track():
    instance = tracker(min_hits=1, iou_threshold=0.5)

    instance.update(0, [box(0, 50)])
    instance.update(1, [box(200, 50)])

    assert len({track.track_id for track in instance.finalize()}) == 2


def test_each_detection_is_matched_to_at_most_one_track():
    instance = tracker(min_hits=1)
    instance.update(0, [box(0, 50), box(20, 50)])

    updated = instance.update(1, [box(10, 50)])

    assert len(updated) == 1
    ids = {track.track_id for track in instance.active}
    assert ids == {1, 2}


def test_history_records_every_frame_of_a_track():
    tracks = track_video(tracker(min_hits=1), walking_person(6))

    history = tracks[0].history
    assert [entry.frame_index for entry in history] == [0, 1, 2, 3, 4, 5]
    assert all(entry.track_id == 1 for entry in history)
    assert history[0].box == (0.0, 50.0, 40.0, 130.0)
    assert history[0].confidence == pytest.approx(0.9)


def test_finalize_orders_tracks_by_length():
    instance = tracker(min_hits=1, max_age=1)
    for index in range(6):
        detections = [box(index * 3, 50)]
        if index < 2:
            detections.append(box(400, 300))
        instance.update(index, detections)

    tracks = instance.finalize()

    assert [track.length for track in tracks] == sorted(
        (track.length for track in tracks), reverse=True
    )


def test_reset_clears_state_and_identities():
    instance = tracker(min_hits=1)
    track_video(instance, walking_person(5))

    instance.reset()

    assert instance.active == []
    assert instance.finished == []
    assert [track.track_id for track in track_video(instance, walking_person(5))] == [1]


def test_short_tracks_are_discarded():
    instance = tracker(min_hits=1, max_age=1)
    for index in range(8):
        detections = [box(index * 3, 50)]
        if index < 3:
            detections.append(box(400, 300))
        instance.update(index, detections)

    kept = usable_tracks(instance.finalize(), min_track_length=6)

    assert len(kept) == 1
    assert kept[0].length >= 6


def test_usable_tracks_validates_the_minimum_length():
    with pytest.raises(ValueError, match="min_track_length"):
        usable_tracks([], 0)


def test_frames_without_detections_are_allowed():
    instance = tracker(min_hits=1)

    assert instance.update(0, []) == []
    assert instance.finalize() == []


def test_negative_frame_index_is_rejected():
    with pytest.raises(ValueError, match="frame_index"):
        tracker().update(-1, [])


def test_tracking_is_deterministic_for_equal_overlaps():
    frames = [[box(0, 50), box(0, 50, confidence=0.5)] for _ in range(4)]

    first = [track.track_id for track in track_video(tracker(min_hits=1), frames)]
    second = [track.track_id for track in track_video(tracker(min_hits=1), frames)]

    assert first == second


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("iou_threshold", 0.0),
        ("iou_threshold", 1.0),
        ("max_age", 0),
        ("min_hits", 0),
        ("min_track_length", 0),
    ],
)
def test_tracking_config_validates_its_fields(field, value):
    with pytest.raises(ConfigError, match=field):
        TrackingConfig(**{field: value})
