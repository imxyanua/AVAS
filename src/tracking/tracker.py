"""Associating person detections across frames.

Action recognition needs the motion history of one person, not a pile of boxes.
Tracking supplies that history by giving the same identity to detections of the
same person in consecutive frames.

The association step here is overlap based and deliberately simple: detections
are matched to existing tracks by intersection over union, strongest match first.
Two properties make it useful in practice. A track survives ``max_age`` frames
without a match, which carries a person through a brief occlusion, and a new
track is only trusted after ``min_hits`` detections, which suppresses tracks
created by single-frame false positives.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from src.detection.detector import Detection
from src.utils.config import TrackingConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackedDetection:
    """A detection with the identity and frame it belongs to."""

    track_id: int
    frame_index: int
    detection: Detection

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.detection.box

    @property
    def confidence(self) -> float:
        return self.detection.confidence


@dataclass
class Track:
    """State of one tracked person."""

    track_id: int
    detection: Detection
    first_frame: int
    last_frame: int
    hits: int = 1
    frames_since_update: int = 0
    confirmed: bool = False
    history: list[TrackedDetection] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.history)

    def update(self, frame_index: int, detection: Detection) -> TrackedDetection:
        self.detection = detection
        self.last_frame = frame_index
        self.hits += 1
        self.frames_since_update = 0
        entry = TrackedDetection(self.track_id, frame_index, detection)
        self.history.append(entry)
        return entry

    def mark_missed(self) -> None:
        self.frames_since_update += 1


class IouTracker:
    """Greedy overlap-based multi-person tracker."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self.active: list[Track] = []
        self.finished: list[Track] = []
        self._next_id = 1

    @classmethod
    def from_config(cls, config: TrackingConfig) -> IouTracker:
        return cls(config)

    def reset(self) -> None:
        """Forget all tracks, for example before analysing another video."""
        self.active.clear()
        self.finished.clear()
        self._next_id = 1

    def update(self, frame_index: int, detections: Sequence[Detection]) -> list[TrackedDetection]:
        """Advance the tracker by one frame and return the confirmed tracks seen in it."""
        if frame_index < 0:
            raise ValueError(f"frame_index must be non-negative, got {frame_index}")

        matches, unmatched_detections, unmatched_tracks = self._associate(detections)
        updated: list[TrackedDetection] = []

        for track, detection in matches:
            entry = track.update(frame_index, detection)
            if not track.confirmed and track.hits >= self.config.min_hits:
                track.confirmed = True
            if track.confirmed:
                updated.append(entry)

        for track in unmatched_tracks:
            track.mark_missed()

        for detection in unmatched_detections:
            track = Track(
                track_id=self._next_id,
                detection=detection,
                first_frame=frame_index,
                last_frame=frame_index,
                confirmed=self.config.min_hits <= 1,
            )
            track.history.append(TrackedDetection(track.track_id, frame_index, detection))
            self._next_id += 1
            self.active.append(track)
            if track.confirmed:
                updated.append(track.history[-1])

        self._retire_stale_tracks()
        updated.sort(key=lambda entry: entry.track_id)
        return updated

    def finalize(self) -> list[Track]:
        """Close all tracks and return the confirmed ones, longest first."""
        self.finished.extend(self.active)
        self.active.clear()
        confirmed = [track for track in self.finished if track.confirmed]
        confirmed.sort(key=lambda track: (-track.length, track.track_id))
        return confirmed

    def _associate(
        self, detections: Sequence[Detection]
    ) -> tuple[list[tuple[Track, Detection]], list[Detection], list[Track]]:
        """Match detections to tracks by decreasing overlap."""
        candidates = [
            (track.detection.iou(detection), track_index, detection_index)
            for track_index, track in enumerate(self.active)
            for detection_index, detection in enumerate(detections)
        ]
        candidates = [
            candidate for candidate in candidates if candidate[0] >= self.config.iou_threshold
        ]
        # Sorting by track and detection index after the score keeps the outcome
        # deterministic when several pairs share the same overlap.
        candidates.sort(key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        matches: list[tuple[Track, Detection]] = []

        for _, track_index, detection_index in candidates:
            if track_index in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            matches.append((self.active[track_index], detections[detection_index]))

        unmatched_detections = [
            detection for index, detection in enumerate(detections) if index not in used_detections
        ]
        unmatched_tracks = [
            track for index, track in enumerate(self.active) if index not in used_tracks
        ]
        return matches, unmatched_detections, unmatched_tracks

    def _retire_stale_tracks(self) -> None:
        surviving: list[Track] = []
        for track in self.active:
            if track.frames_since_update > self.config.max_age:
                self.finished.append(track)
            else:
                surviving.append(track)
        self.active = surviving


def track_video(
    tracker: IouTracker, detections_per_frame: Iterable[Sequence[Detection]]
) -> list[Track]:
    """Run a tracker over a whole sequence of per-frame detections."""
    for frame_index, detections in enumerate(detections_per_frame):
        tracker.update(frame_index, detections)
    return tracker.finalize()


def usable_tracks(tracks: Sequence[Track], min_track_length: int) -> list[Track]:
    """Drop tracks too short to fill a clip, reporting how many were discarded."""
    if min_track_length < 1:
        raise ValueError(f"min_track_length must be >= 1, got {min_track_length}")

    keep = [track for track in tracks if track.length >= min_track_length]
    discarded = len(tracks) - len(keep)
    if discarded:
        logger.info(
            "discarded %d of %d tracks shorter than %d frames",
            discarded,
            len(tracks),
            min_track_length,
        )
    return keep
