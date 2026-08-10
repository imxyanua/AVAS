# Changelog

All notable project milestones for AVAS. Dates are UTC merge times on `main` unless noted.

## [Unreleased]

### Desktop UI (`feat/pyside6-desktop`)

- Replaced Streamlit with a PySide6 desktop app (`python -m app.desktop_app`).
- Analysis runs on a worker thread; UI shows original/annotated video, people table, and incident report.

### Dataset baseline (local WIP, separate track)

Branch: `feat/dataset-baseline` (may not be merged)

- Added `src.preprocessing.prepare_avas_baseline` to build `data/raw/avas_baseline` from UCF101 class proxies via selective remote zip extract (`huggingface_hub` + `remotezip`).
- Full download of `UCF-101.zip` (~7 GB) proved unreliable on this machine; selective extract is preferred.
- Local incomplete downloads / partial extracts were cleared; dataset prep must be re-run manually.

Next on the data track: prepare subset → `build_splits` → `build_feature_cache` → train → evaluate → calibrate anomaly thresholds.

## [0.1.0] — 2026-08-09 — end-to-end analysis pipeline on `main`

Pipeline code path is complete: prepare → features → train/eval → detect/track → video analysis → GenAI report → operator UI (originally Streamlit; see Unreleased for PySide6).

### Merged pull requests

| PR | Title | Commit |
| --- | --- | --- |
| [#1](https://github.com/imxyanua/AVAS/pull/1) | Expand README with AVAS system documentation | `dbb0246` |
| [#2](https://github.com/imxyanua/AVAS/pull/2) | Add data preparation foundation for the video pipeline | `3ae865e` |
| [#4](https://github.com/imxyanua/AVAS/pull/4) | Add GitHub Actions workflow for lint and tests | `ae508d5` |
| [#5](https://github.com/imxyanua/AVAS/pull/5) | Add clip datasets and frozen backbone feature caching | `72fc33e` |
| [#6](https://github.com/imxyanua/AVAS/pull/6) | Add baseline action recognition with training and evaluation | `ee9f37b` |
| [#7](https://github.com/imxyanua/AVAS/pull/7) | Add person detection and tracking | `79d7b9e` |
| [#8](https://github.com/imxyanua/AVAS/pull/8) | Add whole-video analysis producing per-person events | `a4e31c9` |
| [#11](https://github.com/imxyanua/AVAS/pull/11) | Add GenAI incident reports from model findings | `c507be7` |
| [#12](https://github.com/imxyanua/AVAS/pull/12) | Add Streamlit app for upload, analysis, overlays, and reports | `231a4cb` |

Notes on the stack after #8: #9 and #10 closed when their base branches were deleted during squash-merge; work landed as #11 and #12 against `main`.

### What shipped by area

**Data foundation**

- Typed YAML configs (`configs/dataset.yaml`, `model.yaml`, `training.yaml`, later `inference.yaml`).
- Video reader, deterministic frame sampling, video-level splits with group leakage checks.
- CLI: `python -m src.preprocessing.build_splits`.

**Features and training**

- Clip datasets and transforms.
- Frozen ResNet backbone feature cache (`src.features.build_feature_cache`).
- CNN + LSTM temporal head, behaviour classifier, anomaly score / risk levels.
- Metrics, early stopping on `val_macro_f1`, train/evaluate CLIs.

**Detection, tracking, inference**

- YOLO person detector behind a `Detector` protocol.
- IoU tracker with `max_age`, `min_hits`, `min_track_length`.
- Person crops (letterbox), whole-video analysis, per-person events and timestamps.
- CLI: `python -m src.inference.predict`.
- OpenCV dependency fixed to a single `opencv-python` variant (avoid dual `cv2` with Ultralytics).

**GenAI and UI**

- `src.genai.analyzer`: narrative report from frozen `model_findings`; template fallback without API key.
- Streamlit app: `streamlit run app/streamlit_app.py`.
- Annotated preview overlays from track histories kept on `VideoAnalysis` (not serialized in JSON).

### Design decisions that still bind new work

1. Split by source-recording group (`group_pattern`); never leak the same recording across splits.
2. Frozen backbone + on-disk feature cache for CPU-friendly training; refuse cache if backbone is unfrozen.
3. GenAI must not change labels, confidences, or anomaly scores.
4. `status` comes from predicted class; `risk_level` from anomaly thresholds — they can disagree until calibrated.
5. Recognition at inference uses person crops; models trained on full frames will see a domain shift unless the cache is built from crops too.

### Environment context

- Primary workstation: Windows, no NVIDIA GPU (Intel Iris Xe). Strategy: frozen ResNet, cache features, train LSTM on CPU.
- Secrets: local `.env` from `.env.example` (`GOOGLE_API_KEY`); never commit `.env`.
- Local-only files (never commit): `GOAL.md`, `RULE.md`.

## [0.0.1] — 2026-08-09 — repository bootstrap

- Initial commit and public README framing AVAS as action recognition + anomaly detection with operator review.
