# AGENT.md

Instructions for any AI working in this repository. Read `README.md` and `CHANGELOG.md` first. Local thesis notes live in `GOAL.md` (do not commit).

## Goal

AVAS = offline video analysis for human actions and anomalies. Deep learning owns predictions; GenAI only explains them. PySide6 is the operator desktop UI.

```text
upload → preprocess → YOLO person detect → track → CNN+LSTM → Normal/Abnormal → GenAI report → Desktop UI
```

## Hard rules

- Branch per task. Never commit on `main`.
- Stage explicit paths. Never `git add -A`.
- Never commit: `GOAL.md`, `RULE.md`, `.env`, weights (`*.pt`), `data/**`, `models/**` payloads, `outputs/**` payloads.
- Commit/PR bodies in English. Chat with the user in Vietnamese (anh = user, em = assistant).
- No emoji, no em-dash, no AI footers, no `Co-Authored-By: Claude/Codex/Cursor`.
- Do not squash-merge or push to `main` unless the user explicitly asks.
- Verify in code before claiming behaviour.

## Layout

| Path | Role |
| --- | --- |
| `configs/` | `dataset.yaml`, `model.yaml`, `training.yaml`, `inference.yaml` |
| `src/preprocessing/` | video IO, sampling, splits, clip dataset, `prepare_avas_baseline` |
| `src/features/` | frozen backbone + feature cache |
| `src/models/` | CNN-LSTM, behaviour / anomaly head |
| `src/training/` | train + evaluate |
| `src/detection/`, `src/tracking/` | YOLO + IoU tracker |
| `src/inference/` | crops, `predict`, annotate overlays |
| `src/genai/` | incident report from frozen findings |
| `app/desktop_app.py` | PySide6 desktop UI |
| `app/ui_support.py` | UI helpers shared with tests |
| `tests/` | pytest |

## Commands

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements-dev.txt

ruff check .
ruff format .
pytest

python -m src.preprocessing.prepare_avas_baseline --max-groups-per-class 20
python -m src.preprocessing.build_splits --summary outputs/reports/split_summary.json
python -m src.features.build_feature_cache --clips-per-video 1
python -m src.training.train --run-name baseline
python -m src.training.evaluate --checkpoint models/checkpoints/baseline.pt --split test --figure
python -m src.inference.predict --video sample.mp4 --checkpoint models/checkpoints/baseline.pt --explain
python -m app.desktop_app
```

File-scoped: `pytest path/to/test_file.py`, `ruff check path/to/file.py`.

## Current status (read CHANGELOG)

- **Done on `main`:** analysis pipeline through GenAI (#1–#8, #11–#12).
- **UI:** PySide6 desktop app replaces Streamlit (`feat/pyside6-desktop`).
- **Not done:** real UCF101-style baseline data, trained checkpoint, anomaly threshold calibration.
- **WIP:** `prepare_avas_baseline` script may exist locally; dataset prep must be re-run manually.

## Dataset map (UCF101 proxies)

| AVAS | UCF101 folder |
| --- | --- |
| walking | WalkingWithDog |
| running | Biking |
| standing | TaiChi |
| sitting | Typing |
| falling | Diving |
| fighting | Punch |

Keep original UCF filenames so `group_pattern: "_(g\\d+)_"` works. Prefer remote selective extract (`remotezip`), not the full 7 GB zip.

## Constraints

- Target machine often has **no NVIDIA GPU**. Keep backbone frozen; train on cached features.
- OpenCV: only `opencv-python` (Ultralytics also needs it). Do not also pin `opencv-python-headless`.
- GenAI: copy `model_findings` unchanged; template fallback without `GOOGLE_API_KEY`.
- `status` vs `risk_level` may disagree until thresholds in `configs/model.yaml` are calibrated.

## PR workflow

1. `git checkout -b feat/<task>` from updated `main`
2. Implement + `ruff` + `pytest`
3. Commit with a short why-focused message
4. `git push -u origin HEAD` and `gh pr create`
5. Squash-merge only when the user asks

Stacked PRs: after a base branch is squash-merged and deleted, retarget or open a replacement PR onto `main` (see #9→#11, #10→#12 in CHANGELOG).
