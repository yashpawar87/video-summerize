# Local PPE Video Summarization

A local Python application that analyzes and summarizes short personal protective equipment (PPE) incident videos using a local Vision-Language Model (VLM) running on Ollama, OpenCV for frame extraction, and SQLite for persistence.

## Goal

Provide structured, fact-grounded summaries of short CCTV-style PPE incident clips. The VLM describes visible facts and evidence without replacing upstream PPE detectors or fabricating unsupported safety conclusions.

## Hard Constraints & Architecture

- **Hardware Target:** Apple Silicon MacBook M4 (16 GB unified memory).
- **Core Stack:** Python 3.12, OpenCV (`opencv-python-headless`), SQLite (`sqlite3`), Pydantic v2.
- **Local Model Runner:** Ollama (`http://localhost:11434`), initial model: `qwen3-vl:4b`.
- **Zero Cloud Dependencies:** No OpenAI API key and no cloud inference APIs. All frames, prompts, and summaries stay entirely on the local machine.
- **Resource Constraints:** Processes one event at a time with at most 8 sampled frames per clip (frames resized to max 1280 px width).
- **Strict Evidence & Uncertainty Rules:**
  - Evaluates `helmet` and `high_visibility_vest` independently.
  - Reports only what is visible in sampled frames and provided metadata.
  - Flags `uncertain` when head or torso is blurred, occluded, or poorly lit.
  - Flags `human_review_required: true` when observations conflict or visibility is ambiguous.
  - Base64 image payloads are never written to logs.

---

## Repository Structure

```
.
├── CLAUDE.md                   # Project specifications and constraints
├── README.md                   # Documentation and usage guide
├── requirements.txt            # Python dependencies
├── schema.sql                  # SQLite schema definition
├── ppe_event_summarizer.py     # End-to-end CLI & pipeline
├── evaluate.py                 # Ground-truth evaluation script
├── data/                       # Local SQLite database (ignored by git)
│   └── ppe_events.db
├── tests/                      # Automated unit test suite
│   ├── test_video_reader.py
│   ├── test_frame_extraction.py
│   ├── test_schema_validation.py
│   ├── test_database.py
│   ├── test_ollama_client.py
│   └── test_evaluation.py
└── ppe_test_dataset/           # Smoke test dataset
    ├── helmet_and_vest_missing.mp4
    ├── helmet_missing.mp4
    └── manifest.json
```

---

## Setup Instructions

### 1. Model Setup (Ollama)

Ensure Ollama is installed and running locally:

```bash
ollama serve
ollama pull qwen3-vl:4b
ollama list
```

### 2. Python Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Process a Single Video

```bash
python ppe_event_summarizer.py --video ppe_test_dataset/helmet_missing.mp4 \
  --alert-type no_helmet \
  --camera-id cam-01 \
  --alert-offset 4.0
```

### Process the Entire Test Dataset

```bash
python ppe_event_summarizer.py --dataset-dir ppe_test_dataset/
```

### Idempotency and Forced Reprocessing

Events are recorded in SQLite (`data/ppe_events.db`). Completed events are skipped on subsequent runs to prevent redundant inference. To force reprocessing:

```bash
python ppe_event_summarizer.py --video ppe_test_dataset/helmet_missing.mp4 --force
```

### CLI Options

| Option | Default | Description |
|---|---|---|
| `--video` | - | Path to a single video file |
| `--dataset-dir` | - | Directory containing `.mp4` video files |
| `--event-id` | File stem | Custom unique event identifier |
| `--alert-type` | Auto-derived | Upstream detector alert type |
| `--camera-id` | `cam-01` | Camera identifier |
| `--alert-offset` | `None` | Clip-relative alert offset in seconds |
| `--detector-meta` | `None` | JSON string of upstream detector metadata |
| `--db-path` | `data/ppe_events.db` | SQLite database file path |
| `--ollama-host` | `http://localhost:11434` | Ollama API endpoint |
| `--model` | `qwen3-vl:4b` | Ollama VLM model identifier |
| `--force` | `False` | Force reprocessing of completed events |
| `--max-frames` | `8` | Maximum sampled frames (hard capped at 8) |

---

## Summary Schema

Each event produces a validated JSON object conforming to:

```json
{
  "summary": "Short factual description of visible event.",
  "visible_activity": "What the worker visibly does.",
  "alert_assessment": "supported",
  "ppe_observations": [
    {
      "item": "helmet",
      "visible_status": "missing",
      "evidence": "Worker's head is clearly visible without a helmet.",
      "evidence_timestamps_sec": [2.0, 4.0]
    },
    {
      "item": "high_visibility_vest",
      "visible_status": "present",
      "evidence": "High-visibility fluorescent vest worn on torso.",
      "evidence_timestamps_sec": [2.0, 4.0]
    }
  ],
  "recommended_follow_up": "Review the clip if the footage or observations are unclear.",
  "human_review_required": false
}
```

Allowed values:
- `alert_assessment`: `supported`, `contradicted`, `inconclusive`
- `visible_status`: `present`, `missing`, `uncertain`, `not_observed`
- Items: Always includes both `helmet` and `high_visibility_vest`.

---

## Evaluation Against Ground Truth

The ground-truth manifest is located at `ppe_test_dataset/manifest.json`.

```bash
python evaluate.py --manifest ppe_test_dataset/manifest.json --db-path data/ppe_events.db
```

- Compares model predictions against `reviewed` labels (not `intended`).
- When `reviewed` is `null`, reports `[⏳ PENDING REVIEW]`.
- Treats `uncertain` as a distinct outcome rather than a match or error.
- Summarizes counts across items and clips.

---

## Running Automated Tests

```bash
pytest -v tests/
```

Unit tests mock Ollama network calls, verifying frame extraction, timestamp calculations, Pydantic validation, SQLite transactions, and evaluation logic completely offline.
