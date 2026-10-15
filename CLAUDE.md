# CLAUDE.md — Local PPE Video Summarization

## Goal

Build and test a local Python application that summarizes short PPE (personal protective equipment) incident videos using a vision-language model (VLM).

The VLM describes what is visible in a clip. It does **not** replace the upstream PPE detector and must not make unsupported safety conclusions.

## Hard constraints

- Hardware: Apple Silicon MacBook M4, 16 GB unified memory.
- Language: Python. Video handling: OpenCV. Persistence: SQLite.
- Local model runner: Ollama. Initial model: `qwen3-vl:4b`.
- No OpenAI API key and no cloud inference APIs. Never add a cloud API dependency.
- Frames, prompts, and summaries stay on the local machine. After the model is downloaded, inference must work offline.
- Process one event at a time, with at most 8 sampled frames per clip.
- Never expose Ollama's local API to a public network.

## Current test videos

Two AI-generated, 10-second, CCTV-style clips. They are the initial smoke-test pair.

```
ppe_test_dataset/
├── helmet_and_vest_missing.mp4
├── helmet_missing.mp4
└── manifest.json
```

| Clip | Intended condition | Intended helmet | Intended vest |
|---|---|---|---|
| `helmet_and_vest_missing.mp4` | Worker without helmet and without vest | missing | missing |
| `helmet_missing.mp4` | Worker without helmet, vest worn | missing | present |

**The intended labels are hypotheses, not ground truth.** AI-generated video can have PPE appear or disappear between frames. Before using a clip as a test case, sample frames and review the clip manually, then record the *reviewed* labels in `manifest.json`.

The pair is deliberately useful: the two clips differ only in vest status, so together they test whether the model assesses the helmet and the vest **independently** rather than treating "PPE violation" as one blob.

Do not assume the files exist at these paths. Check the filesystem, and ask for the correct location only if they cannot be found. Never fabricate a test result for a missing video.

### Ground-truth manifest

Keep expected labels in `ppe_test_dataset/manifest.json`, used **only by the evaluation code**:

```json
{
  "helmet_and_vest_missing.mp4": {
    "intended": {"helmet": "missing", "high_visibility_vest": "missing"},
    "reviewed": {"helmet": null, "high_visibility_vest": null},
    "review_notes": ""
  },
  "helmet_missing.mp4": {
    "intended": {"helmet": "missing", "high_visibility_vest": "present"},
    "reviewed": {"helmet": null, "high_visibility_vest": null},
    "review_notes": ""
  }
}
```

- `reviewed` values are filled in by a human after watching the clip (`present`, `missing`, or `uncertain`). `null` means not yet reviewed.
- **Never put intended or expected labels into the model prompt.** That biases the output and invalidates the test.
- Score model output against `reviewed`, not `intended`. If `reviewed` is `null`, report the comparison as "pending manual review".

## End-to-end workflow

1. Validate that the video exists and opens in OpenCV.
2. Read duration, FPS, frame count, width, and height.
3. Extract up to 8 representative frames with clip-relative timestamps. If an alert offset is given, include frames near it; otherwise sample evenly across the clip.
4. Resize large frames (max width 1280 px) and encode as JPEG.
5. Send images, timestamps, and minimal event metadata to Ollama at `http://localhost:11434` using `qwen3-vl:4b`.
6. Ask the VLM to evaluate helmet and vest separately and return structured JSON.
7. Validate the JSON in Python. Malformed output must never be saved as a successful summary.
8. Persist the event, status, model and prompt version, and summary in SQLite.
9. Print the summary with evidence timestamps so a reviewer can jump back to the clip.

## Local setup

```bash
ollama pull qwen3-vl:4b
ollama list

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check that Ollama is running locally before attempting inference. Do not request cloud credentials.

## Repository structure

```
.
├── CLAUDE.md
├── README.md
├── requirements.txt
├── schema.sql
├── ppe_event_summarizer.py
├── evaluate.py
├── data/
├── tests/
└── ppe_test_dataset/
    ├── helmet_and_vest_missing.mp4
    ├── helmet_missing.mp4
    └── manifest.json
```

Keep videos, real CCTV footage, SQLite files, model files, secrets, and `.venv` out of version control. Update `.gitignore` accordingly.

## Event metadata

Accept where available: `event_id`, `video_path`, `camera_id`, `alert_type`, `event_timestamp`, `alert_offset_seconds`, `detector_metadata` (e.g. confidence, track ID).

Do not confuse the wall-clock `event_timestamp` with the clip-relative `alert_offset_seconds`. For the two test clips, derive `event_id` from the file name (e.g. `helmet_missing`) and use `alert_type` values such as `no_helmet` and `no_helmet_no_vest`.

## Required summary schema

One JSON object per event:

```json
{
  "summary": "Short factual description of the visible event.",
  "visible_activity": "What the worker visibly does.",
  "alert_assessment": "supported",
  "ppe_observations": [
    {
      "item": "helmet",
      "visible_status": "missing",
      "evidence": "The worker's head is clearly visible without a helmet.",
      "evidence_timestamps_sec": [2.0, 4.0]
    },
    {
      "item": "high_visibility_vest",
      "visible_status": "present",
      "evidence": "A high-visibility vest is visible on the worker's torso.",
      "evidence_timestamps_sec": [2.0, 4.0]
    }
  ],
  "recommended_follow_up": "Review the clip if the footage or observations are unclear.",
  "human_review_required": false
}
```

This is an example of the structure, **not** the expected answer for any clip. The model must base its output on the frames.

Allowed values:

- `alert_assessment`: `supported`, `contradicted`, `inconclusive`
- `visible_status`: `present`, `missing`, `uncertain`, `not_observed`

Every PPE observation must include clip-relative `evidence_timestamps_sec` that fall within the clip's duration. Always include both `helmet` and `high_visibility_vest` entries.

## Safety and evidence rules

- Report only what is supported by visible frames and provided metadata.
- Never infer that PPE is missing because it is absent from one sampled frame.
- Use `uncertain` if the body area is small, blurred, occluded, poorly lit, or outside the camera view.
- `not_observed` means the item was not seen in the sampled frames; it does not prove non-compliance.
- Set `human_review_required` to `true` if visibility is poor, observations conflict between frames, or the conclusion could affect a consequential safety decision.
- Keep the detector's original alert separate from the VLM's assessment. The VLM explains the clip; it never silently overwrites detector results.
- Do not infer identity, intent, or other unsupported facts.
- Synthetic clips test software behavior. They do not demonstrate real-world accuracy.

## SQLite persistence

At least two related tables:

- `events`: event ID, video path, camera ID, alert type, event timestamp, alert offset, detector metadata JSON, processing status, duration, sampled frame count, error message, created/updated timestamps.
- `event_summaries`: event ID foreign key, model, prompt version, summary JSON, creation timestamp.

Use parameterized SQL, enable foreign keys, keep transactions short, and store file paths rather than video bytes. Statuses: `pending`, `processing`, `completed`, `failed`. A completed event must not be reprocessed unless an explicit `--force` option is used.

## Testing plan

### Smoke test on the two clips

Run both clips through the full pipeline, then verify:

- Each clip opens and frame timestamps are correct.
- Helmet and vest are separate observations in both outputs.
- `helmet_and_vest_missing.mp4`: the model reports both items as missing, or reports `uncertain` where the footage is unclear. Flag it if the model reports a vest as present.
- `helmet_missing.mp4`: the model reports the helmet as missing and the vest as present (or `uncertain`). **Flag it if the model reports the vest as missing**, since this clip is where the independent-assessment behavior is tested.
- Evidence timestamps are within the clip and point to frames that actually show the item.
- Unclear evidence is reported as `uncertain`, not guessed.
- JSON validates and is saved in SQLite.
- Rerunning a completed event does not repeat inference unless forced.

### Evaluation script

`evaluate.py` should compare model output to `reviewed` labels in `manifest.json` and print, per clip and per item (helmet, vest): expected, predicted, and match / mismatch / pending review. Treat `uncertain` as a distinct outcome rather than a match or an error. With two clips this is a sanity check only; report counts, not percentages.

### Automated tests

- Missing, corrupt, and unreadable video paths.
- Frame extraction and alert-offset sampling.
- Valid and malformed model JSON (including a missing `helmet` or vest entry, bad enum values, and out-of-range timestamps).
- SQLite schema, foreign keys, status transitions, and idempotent processing.
- Mock the Ollama client in unit tests so they run without the model. Keep live Ollama tests optional and clearly documented.

Two clips cannot establish model accuracy. Do not claim broad accuracy or production readiness. Expand the dataset before drawing conclusions.

## Expanding the dataset

Future AI-generated clips should look like continuous fixed-camera CCTV:

- 5–20 seconds, MP4/H.264, 720p or 1080p where available.
- Fixed elevated or corner-mounted wide-angle camera; ordinary industrial lighting.
- No cuts, dramatic camera movement, slow motion, music, or dialogue.
- Head and torso sufficiently visible for the intended assessment.

Add clips for: missing vest only, correct PPE (all items present), and partial obstruction or poor visibility. Review each clip manually and add it to `manifest.json` before using it for evaluation.

## Coding conventions

- Type hints on public functions; use `pathlib.Path` for paths.
- Separate frame extraction, Ollama requests, JSON validation, and database persistence.
- Add timeouts and clear errors to local HTTP requests.
- Make the Ollama host and model configurable, defaulting to `http://localhost:11434` and `qwen3-vl:4b`.
- Keep concurrency low; default to one event at a time.
- Log event IDs, elapsed time, sampled timestamps, and failures. Never log base64 image payloads.
- Update tests and README when CLI options, schemas, or behavior change.

## Acceptance criteria

A change is complete when:

- Both test clips process locally without cloud credentials.
- Helmet and vest are assessed independently in every output.
- Model JSON is validated before persistence.
- Evidence timestamps refer to the original clip.
- Uncertainty is represented honestly.
- Results and status are stored in SQLite.
- Expected labels never appear in the model prompt.
- `evaluate.py` reports results against manually reviewed labels.
- Unit tests pass and the optional Ollama integration test is documented.
