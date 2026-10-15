#!/usr/bin/env python3
"""
ppe_event_summarizer.py — Local PPE Video Summarization

Summarizes short PPE (personal protective equipment) incident videos using
a local Vision-Language Model (VLM) via Ollama, following safety and evidence rules.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
from pydantic import BaseModel, Field, field_validator, model_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ppe_event_summarizer")

PROMPT_VERSION = "v1.0"
DEFAULT_MODEL = "qwen3-vl:4b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_DB_PATH = Path("data/ppe_events.db")
MAX_FRAMES_LIMIT = 8
MAX_FRAME_WIDTH = 1280


# =====================================================================
# Data Models and Enums
# =====================================================================

class AlertAssessment(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"


class VisibleStatus(str, Enum):
    PRESENT = "present"
    MISSING = "missing"
    UNCERTAIN = "uncertain"
    NOT_OBSERVED = "not_observed"


class PPEObservation(BaseModel):
    item: str
    visible_status: VisibleStatus
    evidence: str
    evidence_timestamps_sec: List[float] = Field(default_factory=list)

    @field_validator("item")
    @classmethod
    def normalize_item(cls, v: str) -> str:
        v_clean = v.strip().lower().replace("-", "_").replace(" ", "_")
        if v_clean in ("vest", "high_vis_vest", "hi_vis_vest", "high_visibility_vest"):
            return "high_visibility_vest"
        if v_clean in ("hard_hat", "hardhat", "helmet", "safety_helmet"):
            return "helmet"
        return v_clean

    @field_validator("evidence_timestamps_sec")
    @classmethod
    def validate_timestamps(cls, v: List[float]) -> List[float]:
        # Sort and round timestamps
        return [round(float(t), 2) for t in sorted(v)]


class PPESummary(BaseModel):
    summary: str
    visible_activity: str
    alert_assessment: AlertAssessment
    ppe_observations: List[PPEObservation]
    recommended_follow_up: str
    human_review_required: bool

    @model_validator(mode="after")
    def validate_required_items(self) -> PPESummary:
        items = {obs.item for obs in self.ppe_observations}
        if "helmet" not in items:
            raise ValueError("PPE observations must contain an entry for 'helmet'.")
        if "high_visibility_vest" not in items:
            raise ValueError("PPE observations must contain an entry for 'high_visibility_vest'.")
        return self


@dataclass
class VideoMetadata:
    video_path: Path
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass
class SampledFrame:
    timestamp_sec: float
    frame_index: int
    image_bytes: bytes
    image_base64: str


# =====================================================================
# Video Handling & Frame Extraction
# =====================================================================

def get_video_metadata(video_path: Path) -> VideoMetadata:
    """Validate video existence and extract basic metadata using OpenCV."""
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    if not video_path.is_file():
        raise ValueError(f"Path is not a regular file: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video file: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or frame_count <= 0:
            raise ValueError(
                f"Invalid video stream properties: frame_count={frame_count}, fps={fps}"
            )

        duration_seconds = frame_count / fps
        return VideoMetadata(
            video_path=video_path,
            duration_seconds=round(duration_seconds, 2),
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )
    finally:
        cap.release()


def calculate_sample_timestamps(
    duration_sec: float,
    max_frames: int = MAX_FRAMES_LIMIT,
    alert_offset_sec: Optional[float] = None,
) -> List[float]:
    """
    Calculate up to max_frames timestamps.
    If alert_offset_sec is specified, clusters sample frames near the alert offset
    while including surrounding context. Otherwise, samples evenly across the duration.
    """
    if duration_sec <= 0:
        return [0.0]

    max_frames = min(max_frames, MAX_FRAMES_LIMIT)
    if max_frames <= 0:
        max_frames = 1

    if alert_offset_sec is not None and 0.0 <= alert_offset_sec <= duration_sec:
        # Offsets around the alert event
        offsets = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        candidate_times = [alert_offset_sec + off for off in offsets]
        # Always ensure clip boundaries are represented if space allows
        candidate_times.append(0.5)
        candidate_times.append(max(0.0, duration_sec - 0.5))

        # Clamp and deduplicate
        clamped = {
            round(max(0.0, min(duration_sec, t)), 2)
            for t in candidate_times
        }
        sorted_times = sorted(clamped)
        if len(sorted_times) > max_frames:
            # Subsample evenly from candidates
            step = len(sorted_times) / max_frames
            selected = [sorted_times[int(i * step)] for i in range(max_frames)]
            return sorted(set(selected))
        return sorted_times

    # Evenly spaced sampling across clip
    step = duration_sec / (max_frames + 1)
    timestamps = [round((i + 1) * step, 2) for i in range(max_frames)]
    return timestamps


def extract_frames(
    video_path: Path,
    timestamps: List[float],
    max_width: int = MAX_FRAME_WIDTH,
) -> List[SampledFrame]:
    """
    Extract frames at given timestamps from the video.
    Resizes large frames (max width 1280 px) preserving aspect ratio and encodes as JPEG.
    Never returns more than MAX_FRAMES_LIMIT frames.
    """
    video_path = Path(video_path).resolve()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file for frame extraction: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled_frames: List[SampledFrame] = []

    # Enforce limit of at most 8 frames
    limited_timestamps = timestamps[:MAX_FRAMES_LIMIT]

    try:
        for ts in limited_timestamps:
            target_frame = int(round(ts * fps))
            if target_frame >= total_frames:
                target_frame = max(0, total_frames - 1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame at timestamp %.2fs (frame %d)", ts, target_frame)
                continue

            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                new_w = max_width
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not success:
                logger.warning("Failed to encode frame at timestamp %.2fs as JPEG", ts)
                continue

            img_bytes = buffer.tobytes()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            sampled_frames.append(
                SampledFrame(
                    timestamp_sec=ts,
                    frame_index=target_frame,
                    image_bytes=img_bytes,
                    image_base64=img_b64,
                )
            )
    finally:
        cap.release()

    return sampled_frames


# =====================================================================
# JSON Schema Validation
# =====================================================================

def validate_summary_json(
    raw_content: Union[str, Dict[str, Any]],
    video_duration: Optional[float] = None,
) -> PPESummary:
    """
    Validate raw model output or dictionary against the required PPE summary schema.
    Raises ValueError if validation fails.
    """
    if isinstance(raw_content, str):
        content = raw_content.strip()
        # Handle markdown fences if returned by LLM
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON returned by model: {exc}\nRaw: {raw_content[:200]}") from exc
    elif isinstance(raw_content, dict):
        data = raw_content
    else:
        raise ValueError(f"Unsupported content type for validation: {type(raw_content)}")

    try:
        summary_obj = PPESummary.model_validate(data)
    except Exception as exc:
        raise ValueError(f"Summary failed schema validation: {exc}") from exc

    # Validate timestamps within clip duration if provided
    if video_duration is not None and video_duration > 0:
        tolerance = 1.0  # slight tolerance for frame rounding
        for obs in summary_obj.ppe_observations:
            for ts in obs.evidence_timestamps_sec:
                if ts < 0.0 or ts > (video_duration + tolerance):
                    raise ValueError(
                        f"Evidence timestamp {ts:.2f}s falls outside clip duration ({video_duration:.2f}s)"
                    )

    return summary_obj


# =====================================================================
# Database Persistence (SQLite)
# =====================================================================

class PPEDatabase:
    """SQLite persistence layer for PPE incident events and summaries."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def initialize_schema(self) -> None:
        schema_file = Path(__file__).resolve().parent / "schema.sql"
        schema_sql = (
            schema_file.read_text(encoding="utf-8")
            if schema_file.exists()
            else """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                video_path TEXT NOT NULL,
                camera_id TEXT,
                alert_type TEXT,
                event_timestamp TEXT,
                alert_offset_seconds REAL,
                detector_metadata TEXT,
                status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
                duration_seconds REAL,
                sampled_frame_count INTEGER,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS event_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
            CREATE INDEX IF NOT EXISTS idx_event_summaries_event_id ON event_summaries(event_id);
            """
        )
        with self._get_connection() as conn:
            conn.executescript(schema_sql)

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT e.*, s.summary_json, s.model, s.prompt_version, s.created_at AS summary_created_at
                FROM events e
                LEFT JOIN event_summaries s ON e.event_id = s.event_id
                WHERE e.event_id = ?
                ORDER BY s.id DESC
                LIMIT 1;
                """,
                (event_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def begin_event(
        self,
        event_id: str,
        video_path: Path,
        camera_id: Optional[str] = None,
        alert_type: Optional[str] = None,
        event_timestamp: Optional[str] = None,
        alert_offset_seconds: Optional[float] = None,
        detector_metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Record start of event processing.
        Returns (should_process, existing_event_record).
        If event is already completed and force is False, returns (False, existing_event).
        """
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(detector_metadata) if detector_metadata else None
        video_str = str(video_path)

        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM events WHERE event_id = ?", (event_id,))
            existing = cur.fetchone()

            if existing:
                existing_dict = dict(existing)
                if existing_dict["status"] == "completed" and not force:
                    logger.info("Event %s is already completed. Skipping (use --force to reprocess).", event_id)
                    return False, existing_dict

                conn.execute(
                    """
                    UPDATE events
                    SET video_path = ?, camera_id = ?, alert_type = ?,
                        event_timestamp = ?, alert_offset_seconds = ?,
                        detector_metadata = ?, status = 'processing',
                        error_message = NULL, updated_at = ?
                    WHERE event_id = ?;
                    """,
                    (
                        video_str,
                        camera_id,
                        alert_type,
                        event_timestamp,
                        alert_offset_seconds,
                        meta_json,
                        now,
                        event_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO events (
                        event_id, video_path, camera_id, alert_type,
                        event_timestamp, alert_offset_seconds, detector_metadata,
                        status, duration_seconds, sampled_frame_count,
                        error_message, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'processing', NULL, NULL, NULL, ?, ?);
                    """,
                    (
                        event_id,
                        video_str,
                        camera_id,
                        alert_type,
                        event_timestamp,
                        alert_offset_seconds,
                        meta_json,
                        now,
                        now,
                    ),
                )

        return True, None

    def complete_event(
        self,
        event_id: str,
        duration_seconds: float,
        sampled_frame_count: int,
        model: str,
        prompt_version: str,
        summary: PPESummary,
    ) -> None:
        """Atomically persist event completion and validated summary."""
        now = datetime.now(timezone.utc).isoformat()
        summary_str = summary.model_dump_json()

        with self._get_connection() as conn:
            # Delete any previous summaries for this event if re-processing with --force
            conn.execute("DELETE FROM event_summaries WHERE event_id = ?", (event_id,))

            conn.execute(
                """
                INSERT INTO event_summaries (event_id, model, prompt_version, summary_json, created_at)
                VALUES (?, ?, ?, ?, ?);
                """,
                (event_id, model, prompt_version, summary_str, now),
            )

            conn.execute(
                """
                UPDATE events
                SET status = 'completed',
                    duration_seconds = ?,
                    sampled_frame_count = ?,
                    error_message = NULL,
                    updated_at = ?
                WHERE event_id = ?;
                """,
                (duration_seconds, sampled_frame_count, now, event_id),
            )

    def fail_event(self, event_id: str, error_message: str) -> None:
        """Mark event as failed with error details."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE events
                SET status = 'failed',
                    error_message = ?,
                    updated_at = ?
                WHERE event_id = ?;
                """,
                (error_message, now, event_id),
            )


# =====================================================================
# Prompt Engineering & Ollama Client
# =====================================================================

def build_ollama_prompt(
    duration_sec: float,
    sampled_timestamps: List[float],
    alert_type: Optional[str] = None,
    alert_offset_sec: Optional[float] = None,
) -> str:
    """
    Construct VLM prompt adhering strictly to evidence and safety rules.
    NEVER put ground truth or expected test labels in prompt!
    """
    timestamps_str = ", ".join(f"{t:.2f}s" for t in sampled_timestamps)
    metadata_notes = [f"Clip duration: {duration_sec:.2f} seconds."]
    if alert_type:
        metadata_notes.append(f"Upstream detector alert type: '{alert_type}'.")
    if alert_offset_sec is not None:
        metadata_notes.append(f"Alert occurred at offset: {alert_offset_sec:.2f} seconds.")

    meta_str = " ".join(metadata_notes)

    prompt = f"""You are an objective CCTV safety video analyst inspecting short surveillance footage.
{meta_str}
You are provided with {len(sampled_timestamps)} sampled frames captured at clip timestamps: [{timestamps_str}].

INSTRUCTIONS & STRICT RULES:
1. Describe ONLY what is directly visible in the sampled frames and metadata. Do not assume or extrapolate.
2. Evaluate PPE compliance for 'helmet' and 'high_visibility_vest' INDEPENDENTLY. Do NOT treat PPE as a single blob.
3. Allowed values:
   - 'alert_assessment': 'supported', 'contradicted', 'inconclusive'
   - 'visible_status': 'present', 'missing', 'uncertain', 'not_observed'
4. Uncertainty and Evidence rules:
   - If a body region (head or torso) is clearly visible and the PPE item is NOT worn (e.g. bare head with no helmet, or shirt without safety vest), report visible_status as 'missing'.
   - If a body region (head or torso) is small, blurred, occluded, poorly lit, or out of frame, report visible_status as 'uncertain'.
   - Do NOT mark an item as missing simply because it is unseen in a single frame.
   - 'not_observed' means the body region or item was not visible or present in any of the sampled frames (e.g. out of camera view).
   - For every observation, include 'evidence' explaining what is visible, and 'evidence_timestamps_sec' listing the timestamps of the frames supporting your finding.
   - Set 'human_review_required' to true if visibility is poor, observations conflict across frames, or evidence is ambiguous.
5. You MUST return ONLY a valid JSON object matching the exact schema below. Do not output markdown fences or explanatory text.

JSON Output Schema:
{{
  "summary": "Short factual description of visible event.",
  "visible_activity": "What the worker visibly does.",
  "alert_assessment": "supported | contradicted | inconclusive",
  "ppe_observations": [
    {{
      "item": "helmet",
      "visible_status": "present | missing | uncertain | not_observed",
      "evidence": "Detailed visible evidence.",
      "evidence_timestamps_sec": [2.0, 4.0]
    }},
    {{
      "item": "high_visibility_vest",
      "visible_status": "present | missing | uncertain | not_observed",
      "evidence": "Detailed visible evidence.",
      "evidence_timestamps_sec": [2.0, 4.0]
    }}
  ],
  "recommended_follow_up": "Actionable follow-up recommendation.",
  "human_review_required": false
}}
"""
    return prompt.strip()


class OllamaClient:
    """Ollama API client for local VLM inference."""

    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_HOST,
        model: str = DEFAULT_MODEL,
        timeout: int = 300,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def check_health(self) -> bool:
        """Verify that Ollama server is reachable."""
        try:
            import requests
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception as exc:
            logger.debug("Ollama health check failed: %s", exc)
            return False

    def generate_summary(
        self,
        frames: List[SampledFrame],
        duration_sec: float,
        alert_type: Optional[str] = None,
        alert_offset_sec: Optional[float] = None,
    ) -> PPESummary:
        """
        Send frames and prompt to Ollama and return validated PPESummary.
        Never logs base64 images!
        """
        import requests

        if not frames:
            raise ValueError("No frames provided for inference.")

        timestamps = [f.timestamp_sec for f in frames]
        prompt = build_ollama_prompt(
            duration_sec=duration_sec,
            sampled_timestamps=timestamps,
            alert_type=alert_type,
            alert_offset_sec=alert_offset_sec,
        )

        logger.info(
            "Sending request to Ollama [model=%s, frames=%d, timestamps=%s]",
            self.model,
            len(frames),
            timestamps,
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [f.image_base64 for f in frames],
                }
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 16384,
            },
        }

        start_time = time.time()
        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Failed to communicate with Ollama at {self.host}: {exc}"
            ) from exc

        elapsed = time.time() - start_time
        logger.info("Ollama responded in %.2f seconds (HTTP %d)", elapsed, resp.status_code)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama returned error HTTP {resp.status_code}: {resp.text[:300]}"
            )

        resp_json = resp.json()
        message_content = resp_json.get("message", {}).get("content", "")
        if not message_content:
            raise ValueError("Empty response received from Ollama.")

        # Validate structured JSON output
        summary = validate_summary_json(message_content, video_duration=duration_sec)
        return summary


# =====================================================================
# End-to-End Summarizer Pipeline
# =====================================================================

def summarize_event(
    video_path: Path,
    event_id: Optional[str] = None,
    alert_type: Optional[str] = None,
    camera_id: Optional[str] = None,
    alert_offset_sec: Optional[float] = None,
    detector_metadata: Optional[Dict[str, Any]] = None,
    db_path: Path = DEFAULT_DB_PATH,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    model: str = DEFAULT_MODEL,
    force: bool = False,
    max_frames: int = MAX_FRAMES_LIMIT,
    timeout: int = 300,
    client: Optional[OllamaClient] = None,
) -> Tuple[bool, Optional[PPESummary]]:
    """
    Execute full pipeline for a single event:
    1. Validate video exists & load metadata.
    2. Extract up to 8 frames with timestamps.
    3. Check DB idempotency.
    4. Call Ollama.
    5. Validate JSON.
    6. Persist results in SQLite.
    Returns (processed_new, summary_object).
    """
    video_path = Path(video_path).resolve()
    if event_id is None:
        event_id = video_path.stem

    now_iso = datetime.now(timezone.utc).isoformat()
    db = PPEDatabase(db_path=db_path)

    should_process, existing_record = db.begin_event(
        event_id=event_id,
        video_path=video_path,
        camera_id=camera_id,
        alert_type=alert_type,
        event_timestamp=now_iso,
        alert_offset_seconds=alert_offset_sec,
        detector_metadata=detector_metadata,
        force=force,
    )

    if not should_process and existing_record:
        logger.info("Event %s already completed in database.", event_id)
        if existing_record.get("summary_json"):
            try:
                summary = validate_summary_json(existing_record["summary_json"])
                return False, summary
            except Exception:
                pass
        return False, None

    try:
        # Step 1: Video metadata
        meta = get_video_metadata(video_path)
        logger.info(
            "Video metadata: %s, duration=%.2fs, fps=%.2f, frames=%d, res=%dx%d",
            video_path.name,
            meta.duration_seconds,
            meta.fps,
            meta.frame_count,
            meta.width,
            meta.height,
        )

        # Step 2: Sampling timestamps
        sample_timestamps = calculate_sample_timestamps(
            duration_sec=meta.duration_seconds,
            max_frames=max_frames,
            alert_offset_sec=alert_offset_sec,
        )
        logger.info("Sampled timestamps (sec): %s", sample_timestamps)

        # Step 3: Extract & resize frames
        frames = extract_frames(video_path, sample_timestamps, max_width=MAX_FRAME_WIDTH)
        if not frames:
            raise ValueError(f"No frames could be extracted from {video_path}")

        # Step 4: Ollama VLM inference
        ollama_client = client or OllamaClient(host=ollama_host, model=model, timeout=timeout)
        summary = ollama_client.generate_summary(
            frames=frames,
            duration_sec=meta.duration_seconds,
            alert_type=alert_type,
            alert_offset_sec=alert_offset_sec,
        )

        # Step 5: Save in DB
        db.complete_event(
            event_id=event_id,
            duration_seconds=meta.duration_seconds,
            sampled_frame_count=len(frames),
            model=model,
            prompt_version=PROMPT_VERSION,
            summary=summary,
        )
        logger.info("Successfully completed and saved summary for event %s", event_id)
        return True, summary

    except Exception as exc:
        logger.error("Failed to process event %s: %s", event_id, exc)
        db.fail_event(event_id, str(exc))
        raise


def print_summary(event_id: str, summary: PPESummary) -> None:
    """Print formatted summary with evidence timestamps for review."""
    print("\n" + "=" * 65)
    print(f" PPE EVENT SUMMARY: {event_id}")
    print("=" * 65)
    print(f" Summary:          {summary.summary}")
    print(f" Visible Activity: {summary.visible_activity}")
    print(f" Alert Assessment: {summary.alert_assessment.value.upper()}")
    print("-" * 65)
    print(" PPE OBSERVATIONS:")
    for obs in summary.ppe_observations:
        ts_str = ", ".join(f"{t:.2f}s" for t in obs.evidence_timestamps_sec)
        print(f"  • {obs.item:20s}: [{obs.visible_status.value.upper()}]")
        print(f"    Evidence:   {obs.evidence}")
        print(f"    Timestamps: [{ts_str}]")
    print("-" * 65)
    print(f" Recommended Follow-up: {summary.recommended_follow_up}")
    print(f" Human Review Required: {'YES' if summary.human_review_required else 'NO'}")
    print("=" * 65 + "\n")


# =====================================================================
# CLI Entry Point
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local PPE Video Event Summarizer using Ollama and OpenCV."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--video",
        type=Path,
        help="Path to single video file to process.",
    )
    group.add_argument(
        "--dataset-dir",
        type=Path,
        help="Directory containing video files to process (e.g., ppe_test_dataset).",
    )

    parser.add_argument(
        "--event-id",
        type=str,
        default=None,
        help="Custom event ID (defaults to video file stem).",
    )
    parser.add_argument(
        "--alert-type",
        type=str,
        default=None,
        help="Upstream detector alert type (e.g., 'no_helmet', 'no_helmet_no_vest').",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="cam-01",
        help="Camera identifier (default: 'cam-01').",
    )
    parser.add_argument(
        "--alert-offset",
        type=float,
        default=None,
        help="Clip-relative alert offset in seconds.",
    )
    parser.add_argument(
        "--detector-meta",
        type=str,
        default=None,
        help="JSON string with detector metadata (e.g. confidence, track ID).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=DEFAULT_OLLAMA_HOST,
        help=f"Ollama server host (default: {DEFAULT_OLLAMA_HOST}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama VLM model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if event is already marked completed.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=MAX_FRAMES_LIMIT,
        help=f"Maximum sampled frames per clip (default: {MAX_FRAMES_LIMIT}, capped at 8).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for Ollama HTTP requests (default: 300).",
    )

    args = parser.parse_args()

    detector_meta_dict = None
    if args.detector_meta:
        try:
            detector_meta_dict = json.loads(args.detector_meta)
        except Exception as exc:
            logger.error("Failed to parse --detector-meta JSON: %s", exc)
            return 1

    videos_to_process: List[Path] = []
    if args.video:
        videos_to_process.append(args.video)
    elif args.dataset_dir:
        dataset_path = Path(args.dataset_dir)
        if not dataset_path.exists():
            logger.error("Dataset directory does not exist: %s", dataset_path)
            return 1
        videos_to_process = sorted(list(dataset_path.glob("*.mp4")))
        if not videos_to_process:
            logger.warning("No .mp4 videos found in %s", dataset_path)
            return 0

    success_count = 0
    failure_count = 0

    for video_file in videos_to_process:
        eid = args.event_id if len(videos_to_process) == 1 and args.event_id else video_file.stem
        # Derive alert_type if not explicitly set for test videos
        derived_alert = args.alert_type
        if derived_alert is None:
            if "helmet_and_vest_missing" in video_file.stem:
                derived_alert = "no_helmet_no_vest"
            elif "helmet_missing" in video_file.stem:
                derived_alert = "no_helmet"

        print(f"\nProcessing: {video_file.name} (Event ID: {eid})")
        try:
            processed_new, summary = summarize_event(
                video_path=video_file,
                event_id=eid,
                alert_type=derived_alert,
                camera_id=args.camera_id,
                alert_offset_sec=args.alert_offset,
                detector_metadata=detector_meta_dict,
                db_path=args.db_path,
                ollama_host=args.ollama_host,
                model=args.model,
                force=args.force,
                max_frames=args.max_frames,
                timeout=args.timeout,
            )
            if summary:
                print_summary(eid, summary)
            success_count += 1
        except Exception as exc:
            print(f"Error processing {video_file.name}: {exc}", file=sys.stderr)
            failure_count += 1

    print(f"\nCompleted run. Success: {success_count}, Failed: {failure_count}")
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
