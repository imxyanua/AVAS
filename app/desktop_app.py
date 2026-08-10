"""PySide6 desktop interface for the AVAS video analysis pipeline.

Upload a video, run detection, tracking and recognition on a worker thread,
then review the per-person events next to an annotated preview and an
operator-facing incident report.

Example:
    python -m app.desktop_app
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.ui_support import (
    ROOT,
    default_checkpoint,
    format_incident_text,
    load_dotenv,
    validate_video_path,
)
from src.genai.analyzer import explain
from src.inference.annotate import people_table_rows, write_annotated_video
from src.inference.predict import build_video_analyzer

TABLE_COLUMNS = (
    "person_id",
    "action",
    "confidence",
    "status",
    "anomaly_score",
    "risk_level",
    "start_time",
    "end_time",
    "peak_start",
    "peak_end",
)


@dataclass(frozen=True)
class AnalysisJob:
    video_path: Path
    checkpoint: Path
    dataset_config: Path
    inference_config: Path
    device: str
    frame_stride: int
    batch_size: int
    write_preview: bool
    prefer_template: bool
    work_dir: Path


@dataclass(frozen=True)
class AnalysisOutcome:
    video_path: Path
    payload: dict
    report: dict
    report_path: Path
    annotated_path: Path | None
    people_rows: list[dict]
    summary: str


class AnalysisWorker(QObject):
    """Runs the heavy analysis pipeline off the UI thread."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, job: AnalysisJob) -> None:
        super().__init__()
        self._job = job

    @Slot()
    def run(self) -> None:
        job = self._job
        try:
            analyzer = build_video_analyzer(
                job.checkpoint,
                dataset_config=job.dataset_config,
                inference_config=job.inference_config,
                device=job.device,
                batch_size=job.batch_size,
            )
            analysis = analyzer.analyze(job.video_path, frame_stride=job.frame_stride)
            payload = analysis.to_dict()

            report_dir = ROOT / "outputs" / "predictions"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"{job.video_path.stem}.json"
            report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            incident = explain(payload, prefer_template=job.prefer_template)
            incident_path = ROOT / "outputs" / "reports" / f"{job.video_path.stem}_incident.json"
            incident_path.parent.mkdir(parents=True, exist_ok=True)
            incident_path.write_text(
                json.dumps(incident.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            annotated_path: Path | None = None
            summary_note = ""
            if job.write_preview and analysis.tracks:
                candidate = job.work_dir / f"{job.video_path.stem}_annotated.mp4"
                try:
                    write_annotated_video(analysis, candidate, frame_stride=job.frame_stride)
                    annotated_path = candidate
                except Exception as preview_error:
                    summary_note = f"Annotated preview skipped: {preview_error}"

            peak = None
            if analysis.people:
                peak = max(analysis.people, key=lambda person: person.decision.anomaly_score)
            summary = (
                f"People: {len(analysis.people)} | "
                f"Abnormal: {len(analysis.abnormal_people)} | "
                f"Peak anomaly: {analysis.max_anomaly_score:.3f}"
            )
            if peak is not None:
                summary += (
                    f" | Person {peak.track_id}: {peak.decision.action} "
                    f"({100 * peak.decision.confidence:.1f}% · {peak.decision.status})"
                )
            summary += f" | Analysis: {report_path.as_posix()}"
            if summary_note:
                summary += f" | {summary_note}"

            self.finished.emit(
                AnalysisOutcome(
                    video_path=job.video_path,
                    payload=payload,
                    report=incident.to_dict(),
                    report_path=report_path,
                    annotated_path=annotated_path,
                    people_rows=people_table_rows(analysis),
                    summary=summary,
                )
            )
        except Exception as error:
            detail = f"{error}\n\n{traceback.format_exc()}"
            self.failed.emit(detail)


class VideoPane(QGroupBox):
    """Simple file-backed video player."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        self._video = QVideoWidget(self)
        self._player.setVideoOutput(self._video)
        self._path_label = QLabel("No video loaded")
        self._path_label.setWordWrap(True)

        play = QPushButton("Play")
        pause = QPushButton("Pause")
        stop = QPushButton("Stop")
        play.clicked.connect(self._player.play)
        pause.clicked.connect(self._player.pause)
        stop.clicked.connect(self._player.stop)

        controls = QHBoxLayout()
        controls.addWidget(play)
        controls.addWidget(pause)
        controls.addWidget(stop)
        controls.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self._video, stretch=1)
        layout.addLayout(controls)
        layout.addWidget(self._path_label)

    def load(self, path: Path | None) -> None:
        if path is None or not path.is_file():
            self._player.stop()
            self._player.setSource(QUrl())
            self._path_label.setText("No video loaded")
            return
        self._player.setSource(QUrl.fromLocalFile(str(path.resolve())))
        self._path_label.setText(str(path))
        self._player.play()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AVAS — AI Video Behavior Analysis")
        self.resize(1280, 800)

        self._worker_thread: QThread | None = None
        self._worker: AnalysisWorker | None = None
        self._work_dir: Path | None = None

        self._checkpoint = QLineEdit(str(default_checkpoint()))
        self._dataset_config = QLineEdit(str(ROOT / "configs" / "dataset.yaml"))
        self._inference_config = QLineEdit(str(ROOT / "configs" / "inference.yaml"))
        self._video_path = QLineEdit()
        self._video_path.setPlaceholderText("Select a video to analyse")
        self._video_path.setReadOnly(True)

        self._device = QComboBox()
        self._device.addItems(["auto", "cpu", "cuda"])
        self._frame_stride = QSpinBox()
        self._frame_stride.setRange(1, 30)
        self._frame_stride.setValue(2)
        self._batch_size = QSpinBox()
        self._batch_size.setRange(1, 16)
        self._batch_size.setValue(4)
        self._write_preview = QCheckBox("Write annotated preview video")
        self._write_preview.setChecked(True)
        self._template_only = QCheckBox("Template-only incident report")
        self._template_only.setChecked(not bool(os.environ.get("GOOGLE_API_KEY", "").strip()))

        browse_checkpoint = QPushButton("Browse…")
        browse_checkpoint.clicked.connect(self._browse_checkpoint)
        browse_video = QPushButton("Browse…")
        browse_video.clicked.connect(self._browse_video)
        self._analyse_button = QPushButton("Analyse video")
        self._analyse_button.clicked.connect(self._start_analysis)

        settings_form = QFormLayout()
        checkpoint_row = QHBoxLayout()
        checkpoint_row.addWidget(self._checkpoint, stretch=1)
        checkpoint_row.addWidget(browse_checkpoint)
        video_row = QHBoxLayout()
        video_row.addWidget(self._video_path, stretch=1)
        video_row.addWidget(browse_video)
        settings_form.addRow("Video", video_row)
        settings_form.addRow("Checkpoint", checkpoint_row)
        settings_form.addRow("Dataset config", self._dataset_config)
        settings_form.addRow("Inference config", self._inference_config)
        settings_form.addRow("Device", self._device)
        settings_form.addRow("Frame stride", self._frame_stride)
        settings_form.addRow("Batch size", self._batch_size)
        settings_form.addRow(self._write_preview)
        settings_form.addRow(self._template_only)
        settings_form.addRow(self._analyse_button)

        settings_box = QGroupBox("Settings")
        settings_box.setLayout(settings_form)

        caption = QLabel(
            "Deep learning owns the predictions. Generative AI only explains them "
            "and never revises labels, confidences, or anomaly scores."
        )
        caption.setWordWrap(True)

        self._status = QLabel("Select a video and press Analyse video.")
        self._status.setWordWrap(True)

        self._people = QTableWidget(0, len(TABLE_COLUMNS))
        self._people.setHorizontalHeaderLabels(list(TABLE_COLUMNS))
        self._people.horizontalHeader().setStretchLastSection(True)
        self._people.setAlternatingRowColors(True)

        self._report = QTextEdit()
        self._report.setReadOnly(True)

        self._original = VideoPane("Original video")
        self._annotated = VideoPane("Annotated preview")

        media = QSplitter(Qt.Orientation.Horizontal)
        media.addWidget(self._original)
        media.addWidget(self._annotated)
        media.setSizes([640, 640])

        right = QVBoxLayout()
        right.addWidget(media, stretch=3)
        right.addWidget(QLabel("Tracked people"))
        right.addWidget(self._people, stretch=2)
        right.addWidget(QLabel("Incident report"))
        right.addWidget(self._report, stretch=2)
        right.addWidget(self._status)
        right_wrap = QWidget()
        right_wrap.setLayout(right)

        left = QVBoxLayout()
        left.addWidget(caption)
        left.addWidget(settings_box)
        left.addStretch(1)
        left_wrap = QWidget()
        left_wrap.setLayout(left)
        left_wrap.setMaximumWidth(420)

        root = QHBoxLayout()
        root.addWidget(left_wrap)
        root.addWidget(right_wrap, stretch=1)
        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

    def closeEvent(self, event) -> None:  # Qt override
        self._shutdown_worker()
        super().closeEvent(event)

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select checkpoint",
            str(ROOT / "models" / "checkpoints"),
            "PyTorch checkpoint (*.pt *.pth);;All files (*.*)",
        )
        if path:
            self._checkpoint.setText(path)

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video",
            str(ROOT),
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)",
        )
        if path:
            self._video_path.setText(path)
            self._original.load(Path(path))
            self._annotated.load(None)

    def _start_analysis(self) -> None:
        try:
            video = validate_video_path(Path(self._video_path.text().strip()))
            checkpoint = Path(self._checkpoint.text().strip()).expanduser()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
            dataset_config = Path(self._dataset_config.text().strip()).expanduser()
            inference_config = Path(self._inference_config.text().strip()).expanduser()
            if not dataset_config.is_file():
                raise FileNotFoundError(f"dataset config not found: {dataset_config}")
            if not inference_config.is_file():
                raise FileNotFoundError(f"inference config not found: {inference_config}")
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Cannot start analysis", str(error))
            return

        self._shutdown_worker()
        self._work_dir = Path(tempfile.mkdtemp(prefix="avas-desktop-"))
        job = AnalysisJob(
            video_path=video,
            checkpoint=checkpoint.resolve(),
            dataset_config=dataset_config.resolve(),
            inference_config=inference_config.resolve(),
            device=self._device.currentText(),
            frame_stride=int(self._frame_stride.value()),
            batch_size=int(self._batch_size.value()),
            write_preview=self._write_preview.isChecked(),
            prefer_template=self._template_only.isChecked(),
            work_dir=self._work_dir,
        )

        self._analyse_button.setEnabled(False)
        self._status.setText("Running detection, tracking and recognition…")
        self._people.setRowCount(0)
        self._report.clear()

        thread = QThread(self)
        worker = AnalysisWorker(job)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._worker_thread = thread
        self._worker = worker
        thread.start()

    @Slot(object)
    def _on_finished(self, outcome: object) -> None:
        assert isinstance(outcome, AnalysisOutcome)
        self._analyse_button.setEnabled(True)
        self._original.load(outcome.video_path)
        self._annotated.load(outcome.annotated_path)
        self._fill_people(outcome.people_rows)
        report = outcome.report
        self._report.setPlainText(
            format_incident_text(
                source=str(report.get("source", "")),
                analysis=str(report.get("analysis", "")),
                risk_assessment=str(report.get("risk_assessment", "")),
                recommended_action=str(report.get("recommended_action", "")),
            )
            + "\n\nFrozen model findings\n"
            + json.dumps(report.get("model_findings", {}), indent=2, ensure_ascii=False)
        )
        self._status.setText(outcome.summary)
        self._worker_thread = None
        self._worker = None

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._analyse_button.setEnabled(True)
        self._status.setText("Analysis failed.")
        QMessageBox.critical(self, "Analysis failed", message)
        self._worker_thread = None
        self._worker = None

    def _fill_people(self, rows: list[dict]) -> None:
        self._people.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, key in enumerate(TABLE_COLUMNS):
                value = row.get(key, "")
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                self._people.setItem(row_index, column_index, item)
        self._people.resizeColumnsToContents()

    def _shutdown_worker(self) -> None:
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait(1000)
        self._worker_thread = None
        self._worker = None


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    app = QApplication(argv if argv is not None else sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
