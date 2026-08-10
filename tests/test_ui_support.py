import os
from pathlib import Path

from app.ui_support import (
    format_incident_text,
    load_dotenv,
    validate_video_path,
)


def test_load_dotenv_sets_missing_keys_only(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("DEMO_KEY=from-file\nEXISTING=ignored\n", encoding="utf-8")
    monkeypatch.setenv("EXISTING", "keep-me")
    monkeypatch.delenv("DEMO_KEY", raising=False)

    load_dotenv(env_file)

    assert os.environ["DEMO_KEY"] == "from-file"
    assert os.environ["EXISTING"] == "keep-me"


def test_validate_video_path_rejects_bad_suffix(tmp_path: Path):
    path = tmp_path / "clip.txt"
    path.write_text("x", encoding="utf-8")
    try:
        validate_video_path(path)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_validate_video_path_accepts_mp4(tmp_path: Path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not-a-real-video")
    assert validate_video_path(path) == path.resolve()


def test_format_incident_text_includes_sections():
    text = format_incident_text(
        source="template",
        analysis="Person 1 fell.",
        risk_assessment="High risk.",
        recommended_action="Review the clip.",
    )
    assert "Source: template" in text
    assert "Person 1 fell." in text
    assert "High risk." in text
    assert "Review the clip." in text
