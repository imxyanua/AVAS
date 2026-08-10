"""Shared helpers for the desktop operator interface."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_CHECKPOINT = ROOT / "models" / "checkpoints" / "baseline.pt"
SMOKE_CHECKPOINT = ROOT / "models" / "checkpoints" / "smoke.pt"


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env without adding a dependency."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def default_checkpoint() -> Path:
    """Prefer a trained baseline checkpoint, otherwise a local smoke weight."""
    if DEFAULT_CHECKPOINT.is_file():
        return DEFAULT_CHECKPOINT
    if SMOKE_CHECKPOINT.is_file():
        return SMOKE_CHECKPOINT
    return DEFAULT_CHECKPOINT


def validate_video_path(path: Path) -> Path:
    """Accept only the video extensions the analysis pipeline can open."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"video not found: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        raise ValueError(
            f"unsupported video type {suffix}; expected one of {', '.join(sorted(UPLOAD_SUFFIXES))}"
        )
    return resolved


def format_incident_text(
    *,
    source: str,
    analysis: str,
    risk_assessment: str,
    recommended_action: str,
) -> str:
    """Plain-text incident report for the desktop report pane."""
    return "\n".join(
        [
            f"Source: {source}",
            "",
            "Analysis",
            analysis.strip(),
            "",
            "Risk assessment",
            risk_assessment.strip(),
            "",
            "Recommended action",
            recommended_action.strip(),
        ]
    )
