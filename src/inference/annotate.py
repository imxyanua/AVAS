"""Draw track boxes and recognition labels onto video frames.

The analysis report already carries per-person conclusions. This module turns
the track histories kept on ``VideoAnalysis`` into overlays an operator can
watch next to the original video.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.inference.predict import PersonAnalysis, VideoAnalysis
from src.preprocessing.video_reader import iter_frames
from src.tracking.tracker import Track

# BGR colours used by OpenCV drawing helpers.
_STATUS_COLOUR = {
    "abnormal": (40, 40, 220),
    "normal": (60, 180, 75),
}
_DEFAULT_COLOUR = (220, 180, 40)


@dataclass(frozen=True)
class OverlayLabel:
    """Everything drawn for one person in one frame."""

    track_id: int
    box: tuple[float, float, float, float]
    text: str
    colour: tuple[int, int, int]


def label_for_person(person: PersonAnalysis) -> str:
    """Short caption placed above a track box."""
    return (
        f"ID {person.track_id} {person.decision.action} "
        f"{100 * person.decision.confidence:.0f}% "
        f"{person.decision.status}/{person.decision.risk_level}"
    )


def colour_for_person(person: PersonAnalysis) -> tuple[int, int, int]:
    """Pick a box colour from the predicted status."""
    return _STATUS_COLOUR.get(person.decision.status, _DEFAULT_COLOUR)


def build_frame_overlays(
    tracks: Sequence[Track],
    people: Sequence[PersonAnalysis],
) -> dict[int, list[OverlayLabel]]:
    """Index overlays by frame so a single video pass can look them up."""
    by_id = {person.track_id: person for person in people}
    overlays: dict[int, list[OverlayLabel]] = {}

    for track in tracks:
        person = by_id.get(track.track_id)
        text = label_for_person(person) if person is not None else f"ID {track.track_id}"
        colour = colour_for_person(person) if person is not None else _DEFAULT_COLOUR
        for entry in track.history:
            overlays.setdefault(entry.frame_index, []).append(
                OverlayLabel(
                    track_id=track.track_id,
                    box=entry.box,
                    text=text,
                    colour=colour,
                )
            )
    return overlays


def draw_overlays(frame: np.ndarray, overlays: Sequence[OverlayLabel]) -> np.ndarray:
    """Return a copy of ``frame`` with boxes and captions drawn on it."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 frame, got shape {frame.shape}")

    canvas = frame.copy()
    for overlay in overlays:
        x1, y1, x2, y2 = overlay.box
        left, top = round(x1), round(y1)
        right, bottom = round(x2), round(y2)
        cv2.rectangle(canvas, (left, top), (right, bottom), overlay.colour, 2)

        label_y = max(16, top - 8)
        cv2.putText(
            canvas,
            overlay.text,
            (left, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            overlay.colour,
            1,
            cv2.LINE_AA,
        )
    return canvas


def write_annotated_video(
    analysis: VideoAnalysis,
    output_path: str | Path,
    *,
    frame_stride: int = 1,
) -> Path:
    """Decode ``analysis.video`` and write a copy with track overlays."""
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    overlays = build_frame_overlays(analysis.tracks, analysis.people)
    writer: cv2.VideoWriter | None = None
    written = 0

    try:
        for index, frame in iter_frames(analysis.video, step=frame_stride):
            annotated = draw_overlays(frame, overlays.get(index, ()))
            if writer is None:
                height, width = annotated.shape[:2]
                fps = analysis.fps if analysis.fps > 0 else 25.0
                writer = _open_writer(path, fps, width, height)
            writer.write(annotated)
            written += 1
    finally:
        if writer is not None:
            writer.release()

    if written == 0:
        raise RuntimeError(f"no frames were written for {analysis.video}")
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"annotated video was not created at {path}")
    return path


def people_table_rows(analysis: VideoAnalysis) -> list[dict[str, object]]:
    """Flatten person records into rows suitable for a dataframe display."""
    rows: list[dict[str, object]] = []
    for person in analysis.people:
        peak = person.peak_clip
        rows.append(
            {
                "person_id": person.track_id,
                "action": person.decision.action,
                "confidence": round(person.decision.confidence, 4),
                "status": person.decision.status,
                "anomaly_score": round(person.decision.anomaly_score, 4),
                "risk_level": person.decision.risk_level,
                "start_time": round(person.start_time, 3),
                "end_time": round(person.end_time, 3),
                "peak_start": round(peak.start_time, 3),
                "peak_end": round(peak.end_time, 3),
            }
        )
    return rows


def _open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    suffix = path.suffix.lower()
    if suffix in {".avi"}:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"could not open video writer for {path}")
    return writer


def overlay_summary(overlays: Mapping[int, Sequence[OverlayLabel]]) -> dict[str, int]:
    """Compact counts used by tests and diagnostics."""
    return {
        "frames": len(overlays),
        "boxes": sum(len(items) for items in overlays.values()),
    }
