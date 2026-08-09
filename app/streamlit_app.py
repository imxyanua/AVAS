"""Streamlit interface for the AVAS video analysis pipeline.

Upload a video, run detection, tracking and recognition, then review the
per-person events next to an annotated preview and an operator-facing report.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from src.genai.analyzer import explain
from src.inference.annotate import people_table_rows, write_annotated_video
from src.inference.predict import VideoAnalysis, build_video_analyzer

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_CHECKPOINT = ROOT / "models" / "checkpoints" / "baseline.pt"
SMOKE_CHECKPOINT = ROOT / "models" / "checkpoints" / "smoke.pt"


def _load_dotenv(path: Path) -> None:
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


def _default_checkpoint() -> str:
    if DEFAULT_CHECKPOINT.is_file():
        return str(DEFAULT_CHECKPOINT)
    if SMOKE_CHECKPOINT.is_file():
        return str(SMOKE_CHECKPOINT)
    return str(DEFAULT_CHECKPOINT)


@st.cache_resource(show_spinner=False)
def _cached_analyzer(
    checkpoint: str,
    dataset_config: str,
    inference_config: str,
    device: str,
    batch_size: int,
):
    return build_video_analyzer(
        checkpoint,
        dataset_config=dataset_config,
        inference_config=inference_config,
        device=device,
        batch_size=batch_size,
    )


def _save_upload(upload, directory: Path) -> Path:
    suffix = Path(upload.name).suffix.lower() or ".mp4"
    if suffix not in UPLOAD_SUFFIXES:
        raise ValueError(
            f"unsupported upload type {suffix}; expected one of "
            f"{', '.join(sorted(UPLOAD_SUFFIXES))}"
        )
    path = directory / f"upload{suffix}"
    path.write_bytes(upload.getbuffer())
    return path


def _render_people(analysis: VideoAnalysis) -> None:
    rows = people_table_rows(analysis)
    if not rows:
        st.info("No usable person tracks were found in this video.")
        return

    st.subheader("Tracked people")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    peak = max(analysis.people, key=lambda person: person.decision.anomaly_score)
    cols = st.columns(4)
    cols[0].metric("People", len(analysis.people))
    cols[1].metric("Abnormal", len(analysis.abnormal_people))
    cols[2].metric("Peak anomaly", f"{analysis.max_anomaly_score:.3f}")
    cols[3].metric(
        f"Person {peak.track_id}",
        peak.decision.action,
        f"{100 * peak.decision.confidence:.1f}% · {peak.decision.status}",
    )


def _render_report(payload: dict, prefer_template: bool) -> None:
    st.subheader("Incident report")
    report = explain(payload, prefer_template=prefer_template)
    st.caption(f"Source: {report.source}")
    st.markdown("**Analysis**")
    st.write(report.analysis)
    st.markdown("**Risk assessment**")
    st.write(report.risk_assessment)
    st.markdown("**Recommended action**")
    st.write(report.recommended_action)
    with st.expander("Frozen model findings"):
        st.json(report.model_findings)
    with st.expander("Full report JSON"):
        st.code(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), language="json")


def main() -> None:
    _load_dotenv(ROOT / ".env")
    st.set_page_config(page_title="AVAS", layout="wide")
    st.title("AI Video Behavior Analysis")
    st.caption(
        "Deep learning owns the predictions. Generative AI only explains them "
        "and never revises labels, confidences, or anomaly scores."
    )

    with st.sidebar:
        st.header("Settings")
        checkpoint = st.text_input("Checkpoint", value=_default_checkpoint())
        dataset_config = st.text_input("Dataset config", value=str(ROOT / "configs/dataset.yaml"))
        inference_config = st.text_input(
            "Inference config", value=str(ROOT / "configs/inference.yaml")
        )
        device = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)
        frame_stride = st.number_input("Frame stride", min_value=1, max_value=30, value=2, step=1)
        batch_size = st.number_input("Batch size", min_value=1, max_value=16, value=4, step=1)
        write_preview = st.checkbox("Write annotated preview video", value=True)
        prefer_template = st.checkbox(
            "Template-only incident report",
            value=not bool(os.environ.get("GOOGLE_API_KEY", "").strip()),
            help="Use the offline template instead of calling Google GenAI.",
        )
        st.markdown(
            "Upload MP4, AVI, MOV, or MKV. Longer videos are faster with a higher "
            "frame stride; timestamps stay correct."
        )

    upload = st.file_uploader("Upload video", type=sorted(ext[1:] for ext in UPLOAD_SUFFIXES))
    run = st.button("Analyse video", type="primary", disabled=upload is None)

    if not run:
        st.info("Upload a video and press Analyse video to run the pipeline.")
        return

    if not Path(checkpoint).is_file():
        st.error(f"Checkpoint not found: {checkpoint}")
        return

    work = Path(tempfile.mkdtemp(prefix="avas-streamlit-"))
    try:
        video_path = _save_upload(upload, work)
    except ValueError as error:
        st.error(str(error))
        return

    with st.spinner("Running detection, tracking and recognition..."):
        try:
            analyzer = _cached_analyzer(
                str(Path(checkpoint).resolve()),
                str(Path(dataset_config).resolve()),
                str(Path(inference_config).resolve()),
                device,
                int(batch_size),
            )
            analysis = analyzer.analyze(video_path, frame_stride=int(frame_stride))
        except Exception as error:
            st.error(f"Analysis failed: {error}")
            return

    payload = analysis.to_dict()
    report_dir = ROOT / "outputs" / "predictions"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{video_path.stem}.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    left, right = st.columns(2)
    with left:
        st.subheader("Original video")
        st.video(str(video_path))
    with right:
        st.subheader("Annotated preview")
        if write_preview and analysis.tracks:
            annotated_path = work / f"{video_path.stem}_annotated.mp4"
            try:
                write_annotated_video(analysis, annotated_path, frame_stride=int(frame_stride))
                st.video(str(annotated_path))
            except Exception as error:
                st.warning(f"Could not write annotated preview: {error}")
                st.info("Person table and incident report are still available below.")
        elif not analysis.tracks:
            st.info("No tracks available to annotate.")
        else:
            st.info("Annotated preview disabled in the sidebar.")

    _render_people(analysis)
    _render_report(payload, prefer_template=prefer_template)
    st.caption(f"Structured analysis written to `{report_path.as_posix()}`.")


if __name__ == "__main__":
    main()
