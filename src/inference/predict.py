"""Analyse a video: find people, follow them, and classify what each one does.

The pipeline decodes the video twice on purpose. The first pass runs detection and
tracking and keeps only boxes, and the second pass decodes the frames the selected
clips need and immediately reduces each one to a small person crop. Holding whole
decoded frames instead would cost megabytes per frame, which does not scale to a
long video, while crops stay bounded by the clip budget.

Example:
    python -m src.inference.predict --video sample.mp4 \
        --checkpoint models/checkpoints/baseline.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.detection.detector import Detection, Detector, YoloDetector
from src.features.feature_extractor import build_backbone, extract_clip_features
from src.inference.crops import crop_person
from src.models.classifier import BehaviorClassifier, BehaviorDecision
from src.models.cnn_lstm import TemporalClassifier
from src.preprocessing.frame_extractor import sliding_windows
from src.preprocessing.transforms import build_normalize_transform
from src.preprocessing.video_reader import VideoMetadata, iter_frames, read_metadata
from src.tracking.tracker import IouTracker, Track, usable_tracks
from src.utils.config import (
    DatasetConfig,
    InferenceConfig,
    ModelConfig,
    SamplingConfig,
)
from src.utils.device import resolve_device

logger = logging.getLogger("avas.predict")

FALLBACK_FPS = 25.0


@dataclass(frozen=True)
class ClipPrediction:
    """Model output for one clip of one person."""

    frame_indices: tuple[int, ...]
    start_time: float
    end_time: float
    decision: BehaviorDecision
    probabilities: tuple[float, ...]

    def to_dict(self, classes: Sequence[str]) -> dict[str, object]:
        return {
            "start_frame": self.frame_indices[0],
            "end_frame": self.frame_indices[-1],
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "action": self.decision.action,
            "confidence": round(self.decision.confidence, 4),
            "anomaly_score": round(self.decision.anomaly_score, 4),
            "risk_level": self.decision.risk_level,
            "probabilities": {
                name: round(value, 4)
                for name, value in zip(classes, self.probabilities, strict=True)
            },
        }


@dataclass(frozen=True)
class PersonAnalysis:
    """Everything the system concluded about one tracked person."""

    track_id: int
    first_frame: int
    last_frame: int
    start_time: float
    end_time: float
    frames_tracked: int
    decision: BehaviorDecision
    clips: tuple[ClipPrediction, ...]

    @property
    def peak_clip(self) -> ClipPrediction:
        """The clip that carries the strongest anomaly evidence."""
        return max(self.clips, key=lambda clip: clip.decision.anomaly_score)

    def to_dict(self, classes: Sequence[str]) -> dict[str, object]:
        peak = self.peak_clip
        return {
            "person_id": self.track_id,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "frames_tracked": self.frames_tracked,
            "action": self.decision.action,
            "confidence": round(self.decision.confidence, 4),
            "status": self.decision.status,
            "anomaly_score": round(self.decision.anomaly_score, 4),
            "risk_level": self.decision.risk_level,
            "peak_window": {
                "start_time": round(peak.start_time, 3),
                "end_time": round(peak.end_time, 3),
                "anomaly_score": round(peak.decision.anomaly_score, 4),
            },
            "clips": [clip.to_dict(classes) for clip in self.clips],
        }


@dataclass(frozen=True)
class VideoAnalysis:
    """Per-person conclusions for one video, plus the video's own properties."""

    video: Path
    fps: float
    frame_count: int
    duration_seconds: float
    classes: tuple[str, ...]
    people: tuple[PersonAnalysis, ...]

    @property
    def abnormal_people(self) -> tuple[PersonAnalysis, ...]:
        return tuple(person for person in self.people if person.decision.is_abnormal)

    @property
    def max_anomaly_score(self) -> float:
        if not self.people:
            return 0.0
        return max(person.decision.anomaly_score for person in self.people)

    def to_dict(self) -> dict[str, object]:
        return {
            "video": self.video.as_posix(),
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "duration_seconds": round(self.duration_seconds, 3),
            "classes": list(self.classes),
            "people_detected": len(self.people),
            "abnormal_people": len(self.abnormal_people),
            "max_anomaly_score": round(self.max_anomaly_score, 4),
            "people": [person.to_dict(self.classes) for person in self.people],
        }


class ClipRecognizer:
    """Encodes clip frames with the backbone and classifies them with the head."""

    def __init__(
        self,
        backbone: nn.Module,
        head: TemporalClassifier,
        *,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.backbone = backbone.to(self.device).eval()
        self.head = head.to(self.device).eval()
        self.transform = build_normalize_transform()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device | None = None,
        model_config: ModelConfig | None = None,
    ) -> tuple[ClipRecognizer, tuple[str, ...], ModelConfig]:
        """Rebuild the recogniser from a training checkpoint."""
        target = device or torch.device("cpu")
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")

        payload = torch.load(path, map_location=target, weights_only=True)
        missing = [key for key in ("model_state", "classes", "feature_dim") if key not in payload]
        if missing:
            raise ValueError(f"checkpoint {path} is missing keys: {missing}")

        if model_config is None:
            if "model_config" not in payload:
                raise ValueError(
                    f"checkpoint {path} does not record its architecture; "
                    "pass a model config explicitly"
                )
            model_config = ModelConfig.from_mapping(payload["model_config"])

        classes = tuple(str(name) for name in payload["classes"])
        head = TemporalClassifier(int(payload["feature_dim"]), len(classes), model_config)
        head.load_state_dict(payload["model_state"])
        backbone, feature_dim = build_backbone(model_config.backbone)
        if feature_dim != int(payload["feature_dim"]):
            raise ValueError(
                f"backbone produces {feature_dim} features but the checkpoint was trained "
                f"on {payload['feature_dim']}"
            )

        return cls(backbone, head, device=target), classes, model_config

    @torch.no_grad()
    def probabilities(self, clips: torch.Tensor) -> torch.Tensor:
        """Classify a batch of ``(B, T, C, H, W)`` uint8 or float clips."""
        if clips.ndim != 5:
            raise ValueError(f"expected clips with shape (B, T, C, H, W), got {tuple(clips.shape)}")

        batch = self.transform(clips).to(self.device)
        features = extract_clip_features(self.backbone, batch)
        logits = self.head(features)
        return torch.softmax(logits.float(), dim=1).cpu()


class VideoAnalyzer:
    """Runs detection, tracking and action recognition over a whole video."""

    def __init__(
        self,
        detector: Detector,
        recognizer: ClipRecognizer,
        behaviour: BehaviorClassifier,
        sampling: SamplingConfig,
        config: InferenceConfig | None = None,
        *,
        batch_size: int = 4,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.detector = detector
        self.recognizer = recognizer
        self.behaviour = behaviour
        self.sampling = sampling
        self.config = config or InferenceConfig()
        self.batch_size = batch_size

    def analyze(self, video_path: str | Path, *, frame_stride: int = 1) -> VideoAnalysis:
        """Analyse one video and return per-person conclusions.

        ``frame_stride`` skips frames during detection, which speeds up a long
        video at the cost of temporal resolution. Timestamps stay correct because
        tracks record the original frame indices.
        """
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")

        path = Path(video_path)
        metadata = read_metadata(path)
        fps = self._frame_rate(metadata)

        tracks = self._detect_and_track(path, frame_stride)
        logger.info("tracking produced %d usable tracks", len(tracks))
        if not tracks:
            return VideoAnalysis(
                video=path,
                fps=fps,
                frame_count=metadata.frame_count,
                duration_seconds=metadata.frame_count / fps if fps > 0 else 0.0,
                classes=tuple(self.behaviour.classes),
                people=(),
            )

        windows = self._select_windows(tracks)
        clips = self._collect_clips(path, windows)
        people = self._classify(tracks, windows, clips, fps)

        return VideoAnalysis(
            video=path,
            fps=fps,
            frame_count=metadata.frame_count,
            duration_seconds=metadata.frame_count / fps if fps > 0 else 0.0,
            classes=tuple(self.behaviour.classes),
            people=people,
        )

    def _frame_rate(self, metadata: VideoMetadata) -> float:
        if metadata.fps > 0:
            return metadata.fps
        logger.warning(
            "%s does not report a frame rate; assuming %.1f fps for timestamps",
            metadata.path,
            FALLBACK_FPS,
        )
        return FALLBACK_FPS

    def _detect_and_track(self, path: Path, frame_stride: int) -> list[Track]:
        tracker = IouTracker(self.config.tracking)
        for index, frame in iter_frames(path, step=frame_stride):
            tracker.update(index, self.detector.detect(frame))
        return usable_tracks(tracker.finalize(), self.config.tracking.min_track_length)

    def _select_windows(
        self, tracks: Sequence[Track]
    ) -> dict[tuple[int, int], list[tuple[int, Detection]]]:
        """Choose which stretches of each track to classify.

        Windows are taken over the frames where the person was actually detected,
        so a gap left by a brief occlusion does not create a window full of
        missing frames.
        """
        selected: dict[tuple[int, int], list[tuple[int, Detection]]] = {}

        for track in tracks:
            history = track.history
            positions = sliding_windows(
                len(history),
                self.sampling.clip_length,
                stride=self.sampling.frame_stride,
            )
            for clip_index, window in enumerate(
                _thin(positions, self.config.clips.max_clips_per_track)
            ):
                selected[(track.track_id, clip_index)] = [
                    (history[position].frame_index, history[position].detection)
                    for position in window
                ]

        return selected

    def _collect_clips(
        self,
        path: Path,
        windows: dict[tuple[int, int], list[tuple[int, Detection]]],
    ) -> dict[tuple[int, int], torch.Tensor]:
        """Decode once more, reducing each needed frame to person crops."""
        requests: dict[int, list[tuple[tuple[int, int], int, Detection]]] = defaultdict(list)
        for key, entries in windows.items():
            for slot, (frame_index, detection) in enumerate(entries):
                requests[frame_index].append((key, slot, detection))

        buffers: dict[tuple[int, int], list[np.ndarray | None]] = {
            key: [None] * len(entries) for key, entries in windows.items()
        }
        outstanding = set(requests)

        for frame_index, frame in iter_frames(path):
            pending = requests.get(frame_index)
            if pending is None:
                continue
            for key, slot, detection in pending:
                buffers[key][slot] = crop_person(
                    frame,
                    detection,
                    padding=self.config.clips.box_padding,
                    size=self.sampling.image_size,
                )
            outstanding.discard(frame_index)
            if not outstanding:
                break

        clips: dict[tuple[int, int], torch.Tensor] = {}
        for key, crops in buffers.items():
            if any(crop is None for crop in crops):
                logger.warning("skipping clip %s because some frames could not be decoded", key)
                continue
            stacked = np.stack([crop for crop in crops if crop is not None])
            clips[key] = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()

        return clips

    def _classify(
        self,
        tracks: Sequence[Track],
        windows: dict[tuple[int, int], list[tuple[int, Detection]]],
        clips: dict[tuple[int, int], torch.Tensor],
        fps: float,
    ) -> tuple[PersonAnalysis, ...]:
        keys = sorted(clips)
        if not keys:
            return ()

        probabilities: list[torch.Tensor] = []
        for start in range(0, len(keys), self.batch_size):
            batch_keys = keys[start : start + self.batch_size]
            batch = torch.stack([clips[key] for key in batch_keys])
            probabilities.append(self.recognizer.probabilities(batch))
        all_probabilities = torch.cat(probabilities)

        per_track: dict[int, list[ClipPrediction]] = defaultdict(list)
        for key, row in zip(keys, all_probabilities, strict=True):
            track_id, _ = key
            indices = tuple(frame_index for frame_index, _ in windows[key])
            per_track[track_id].append(
                ClipPrediction(
                    frame_indices=indices,
                    start_time=indices[0] / fps,
                    end_time=indices[-1] / fps,
                    decision=self.behaviour.decide(row),
                    probabilities=tuple(float(value) for value in row),
                )
            )

        people: list[PersonAnalysis] = []
        for track in tracks:
            predictions = per_track.get(track.track_id)
            if not predictions:
                continue
            mean = torch.stack(
                [torch.tensor(prediction.probabilities) for prediction in predictions]
            ).mean(dim=0)
            people.append(
                PersonAnalysis(
                    track_id=track.track_id,
                    first_frame=track.first_frame,
                    last_frame=track.last_frame,
                    start_time=track.first_frame / fps,
                    end_time=track.last_frame / fps,
                    frames_tracked=track.length,
                    decision=self.behaviour.decide(mean / float(mean.sum())),
                    clips=tuple(predictions),
                )
            )

        people.sort(key=lambda person: (-person.decision.anomaly_score, person.track_id))
        return tuple(people)


def _thin(windows: list[list[int]], limit: int) -> list[list[int]]:
    """Keep at most ``limit`` windows, spread evenly across the track."""
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if len(windows) <= limit:
        return windows

    step = (len(windows) - 1) / (limit - 1) if limit > 1 else 0
    chosen = [windows[round(index * step)] for index in range(limit)]
    return chosen


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--inference-config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="override the architecture stored in the checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="report path (default: outputs/predictions/<video name>.json)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="analyse every Nth frame, trading temporal resolution for speed",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--explain",
        action="store_true",
        help="also write an operator-facing incident report from the model findings",
    )
    parser.add_argument(
        "--template-only",
        action="store_true",
        help="with --explain, skip GenAI and use the deterministic template report",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    dataset_config = DatasetConfig.from_yaml(args.dataset_config)
    inference_config = InferenceConfig.from_yaml(args.inference_config)
    device = resolve_device(args.device)

    model_config = ModelConfig.from_yaml(args.model_config) if args.model_config else None
    recognizer, classes, model_config = ClipRecognizer.from_checkpoint(
        args.checkpoint, device=device, model_config=model_config
    )
    if classes != dataset_config.classes:
        raise SystemExit(
            f"checkpoint was trained on classes {list(classes)}, but the dataset config "
            f"declares {list(dataset_config.classes)}"
        )

    analyzer = VideoAnalyzer(
        YoloDetector(inference_config.detection, device=device),
        recognizer,
        BehaviorClassifier.from_config(dataset_config, model_config),
        dataset_config.sampling,
        inference_config,
        batch_size=args.batch_size,
    )
    analysis = analyzer.analyze(args.video, frame_stride=args.frame_stride)

    output = args.output or Path("outputs/predictions") / f"{args.video.stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = analysis.to_dict()
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info(
        "%s: %d people, %d abnormal, peak anomaly score %.3f",
        args.video.name,
        len(analysis.people),
        len(analysis.abnormal_people),
        analysis.max_anomaly_score,
    )
    for person in analysis.people:
        peak = person.peak_clip
        logger.info(
            "person %d: %s (%.1f%%) %s, anomaly %.3f, strongest window %.2fs to %.2fs",
            person.track_id,
            person.decision.action,
            100 * person.decision.confidence,
            person.decision.risk_level,
            person.decision.anomaly_score,
            peak.start_time,
            peak.end_time,
        )
    logger.info("report written to %s", output)

    if args.explain:
        from src.genai.analyzer import explain

        incident = explain(payload, prefer_template=args.template_only)
        incident_path = Path("outputs/reports") / f"{args.video.stem}_incident.json"
        incident_path.parent.mkdir(parents=True, exist_ok=True)
        incident_path.write_text(
            json.dumps(incident.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("incident report written to %s (source=%s)", incident_path, incident.source)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
