#!/usr/bin/env python3
"""
ppe_event_summarizer.py — Local Safety Video Summarization

Summarizes short safety incident videos using a local Vision-Language Model (VLM)
via Ollama. Supports three incident categories:
  - PPE Compliance (helmet, high-visibility vest)
  - Restricted Zone Entry
  - OSHA Safety Hazards (fall protection, forklift/pedestrian proximity)
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

PROMPT_VERSION = "v2.0"
DEFAULT_MODEL = "qwen3-vl:4b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_DB_PATH = Path("data/ppe_events.db")
MAX_FRAMES_LIMIT = 8
MAX_FRAME_WIDTH = 1280


# =====================================================================
# Data Models and Enums
# =====================================================================

class IncidentType(str, Enum):
    PPE_COMPLIANCE = "ppe_compliance"
    RESTRICTED_ZONE = "restricted_zone"
    OSHA_HAZARD = "osha_hazard"


# Required observation items per incident type
REQUIRED_OBSERVATIONS: Dict[str, List[str]] = {
    IncidentType.PPE_COMPLIANCE: ["helmet", "high_visibility_vest"],
    IncidentType.RESTRICTED_ZONE: ["zone_boundary_crossing", "authorization_indicators"],
    IncidentType.OSHA_HAZARD: [],  # At least one hazard observation required; validated dynamically
}

# Known OSHA hazard observation item names the model may return
KNOWN_OSHA_ITEMS = {
    "fall_protection_hazard", "fall_hazard", "fall_protection",
    "forklift_pedestrian_proximity", "forklift_hazard", "forklift_pedestrian_hazard",
    "unprotected_edge", "elevated_work", "guardrail",
}


class AlertAssessment(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"


class VisibleStatus(str, Enum):
    PRESENT = "present"
    MISSING = "missing"
    UNCERTAIN = "uncertain"
    NOT_OBSERVED = "not_observed"
    DETECTED = "detected"
    NOT_DETECTED = "not_detected"


class SafetyObservation(BaseModel):
    """A single safety observation (PPE item, hazard, or zone compliance check)."""
    item: str
    visible_status: VisibleStatus
    evidence: str
    evidence_timestamps_sec: List[float] = Field(default_factory=list)

    @field_validator("item")
    @classmethod
    def normalize_item(cls, v: str) -> str:
        v_clean = v.strip().lower().replace("-", "_").replace(" ", "_")
        # PPE normalizations
        if v_clean in ("vest", "high_vis_vest", "hi_vis_vest", "high_visibility_vest"):
            return "high_visibility_vest"
        if v_clean in ("hard_hat", "hardhat", "helmet", "safety_helmet"):
            return "helmet"
        # Zone normalizations
        if v_clean in ("zone_crossing", "boundary_crossing", "restricted_area_entry", "zone_entry"):
            return "zone_boundary_crossing"
        if v_clean in ("authorization", "auth_indicators", "personnel_authorization"):
            return "authorization_indicators"
        # OSHA hazard normalizations
        if v_clean in ("fall_hazard", "fall_protection", "unprotected_edge", "elevated_work_without_protection"):
            return "fall_protection_hazard"
        if v_clean in ("forklift_hazard", "forklift_pedestrian", "forklift_proximity", "vehicle_pedestrian_conflict"):
            return "forklift_pedestrian_proximity"
        return v_clean

    @field_validator("evidence_timestamps_sec")
    @classmethod
    def validate_timestamps(cls, v: List[float]) -> List[float]:
        return [round(float(t), 2) for t in sorted(v)]


# Backward-compatible alias
PPEObservation = SafetyObservation


class SafetySummary(BaseModel):
    """Unified summary schema for all incident types."""
    summary: str
    visible_activity: str
    incident_type: str = IncidentType.PPE_COMPLIANCE
    alert_assessment: AlertAssessment
    safety_observations: List[SafetyObservation]
    recommended_follow_up: str
    human_review_required: bool

    @field_validator("incident_type")
    @classmethod
    def validate_incident_type(cls, v: str) -> str:
        try:
            return IncidentType(v).value
        except ValueError as exc:
            raise ValueError(
                "incident_type must be one of: "
                + ", ".join(item.value for item in IncidentType)
            ) from exc

    def validate_for_incident_type(self, incident_type: IncidentType) -> None:
        """Validate that required observation items are present for the given incident type."""
        items = {obs.item for obs in self.safety_observations}
        required = REQUIRED_OBSERVATIONS.get(incident_type, [])
        for req in required:
            if req not in items:
                raise ValueError(
                    f"Safety observations for incident type '{incident_type}' "
                    f"must contain an entry for '{req}'. Got: {items}"
                )
        if incident_type == IncidentType.OSHA_HAZARD and not items:
            raise ValueError("OSHA hazard summary must include at least one safety observation.")

        # VisibleStatus is a union of all incident vocabularies. Enforce the
        # incident-specific vocabulary before persisting model output.
        restricted_statuses = {
            "zone_boundary_crossing": {
                VisibleStatus.DETECTED,
                VisibleStatus.NOT_DETECTED,
                VisibleStatus.UNCERTAIN,
                VisibleStatus.NOT_OBSERVED,
            },
            "authorization_indicators": {
                VisibleStatus.PRESENT,
                VisibleStatus.MISSING,
                VisibleStatus.UNCERTAIN,
                VisibleStatus.NOT_OBSERVED,
            },
        }
        hazard_statuses = {
            VisibleStatus.DETECTED,
            VisibleStatus.NOT_DETECTED,
            VisibleStatus.UNCERTAIN,
            VisibleStatus.NOT_OBSERVED,
        }
        ppe_statuses = {
            VisibleStatus.PRESENT,
            VisibleStatus.MISSING,
            VisibleStatus.UNCERTAIN,
            VisibleStatus.NOT_OBSERVED,
        }

        for obs in self.safety_observations:
            if incident_type == IncidentType.RESTRICTED_ZONE:
                allowed = restricted_statuses.get(obs.item)
            elif incident_type == IncidentType.OSHA_HAZARD:
                # PPE may be reported as a secondary observation in an OSHA
                # scene; hazard observations use the hazard vocabulary.
                allowed = ppe_statuses if obs.item in {"helmet", "high_visibility_vest"} else hazard_statuses
            else:
                allowed = ppe_statuses

            if allowed is not None and obs.visible_status not in allowed:
                allowed_values = ", ".join(sorted(status.value for status in allowed))
                raise ValueError(
                    f"Observation '{obs.item}' for incident type '{incident_type.value}' "
                    f"must use one of: {allowed_values}. "
                    f"Got '{obs.visible_status.value}'."
                )


# Backward-compatible alias
PPESummary = SafetySummary


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
    incident_type: Optional[IncidentType] = IncidentType.PPE_COMPLIANCE,
) -> SafetySummary:
    """
    Validate raw model output against the SafetySummary schema.
    Accepts both the new 'safety_observations' key and the legacy 'ppe_observations' key.
    Validates required items based on incident_type. If incident_type is None,
    the model's declared incident_type is used after validating it.
    Raises ValueError if validation fails.
    """
    if isinstance(raw_content, str):
        content = raw_content.strip()
        # Strip markdown fences if returned by LLM
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
        data = dict(raw_content)  # shallow copy — never mutate the caller's dict
    else:
        raise ValueError(f"Unsupported content type for validation: {type(raw_content)}")

    # Backward-compat: map legacy 'ppe_observations' key to 'safety_observations'
    if isinstance(data, dict) and "ppe_observations" in data and "safety_observations" not in data:
        data["safety_observations"] = data.pop("ppe_observations")

    # For automatic classification, require and validate the model's declared
    # category before Pydantic's compatibility validator can normalize it.
    if incident_type is None:
        raw_incident_type = data.get("incident_type") if isinstance(data, dict) else None
        try:
            validation_incident_type = IncidentType(raw_incident_type)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Automatic incident classification must return incident_type as one of: "
                + ", ".join(item.value for item in IncidentType)
            ) from exc
    else:
        validation_incident_type = incident_type
        # Inject incident_type if missing from model output for explicit,
        # backwards-compatible callers.
        if isinstance(data, dict) and "incident_type" not in data:
            data["incident_type"] = incident_type.value

    try:
        summary_obj = SafetySummary.model_validate(data)
    except Exception as exc:
        raise ValueError(f"Summary failed schema validation: {exc}") from exc

    # Validate required items for this incident type
    try:
        summary_obj.validate_for_incident_type(validation_incident_type)
    except ValueError:
        raise

    # The requested incident type is authoritative. Do not allow a model to
    # return a valid-looking summary under a different category.
    if incident_type is not None and summary_obj.incident_type != incident_type.value:
        raise ValueError(
            f"Summary incident_type '{summary_obj.incident_type}' does not match "
            f"requested incident type '{incident_type.value}'"
        )

    # Validate evidence timestamps within clip duration
    if video_duration is not None and video_duration > 0:
        tolerance = 1.0
        for obs in summary_obj.safety_observations:
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
    incident_type: Optional[IncidentType] = IncidentType.PPE_COMPLIANCE,
    alert_type: Optional[str] = None,
    alert_offset_sec: Optional[float] = None,
) -> str:
    """
    Construct VLM prompt for the given incident type.
    Uses a distinct template for PPE compliance, restricted zone, and OSHA hazards.
    NEVER puts ground truth or expected labels into the prompt!
    """
    timestamps_str = ", ".join(f"{t:.2f}s" for t in sampled_timestamps)
    metadata_notes = [f"Clip duration: {duration_sec:.2f} seconds."]
    if alert_type:
        metadata_notes.append(f"Upstream detector alert type: '{alert_type}'.")
    if alert_offset_sec is not None:
        metadata_notes.append(f"Alert occurred at offset: {alert_offset_sec:.2f} seconds.")
    meta_str = " ".join(metadata_notes)

    n_frames = len(sampled_timestamps)
    common_header = (
        f"You are an objective CCTV safety video analyst inspecting short surveillance footage.\n"
        f"{meta_str}\n"
        f"You are provided with {n_frames} sampled frames captured at clip timestamps: [{timestamps_str}].\n\n"
        "STRICT RULES (apply to all incident types):\n"
        "- Report ONLY what is directly visible in the frames. Do not assume or extrapolate.\n"
        "- 'not_observed': item or feature not visible in any sampled frame.\n"
        "- 'uncertain': body region or feature is blurred, occluded, small, or poorly lit.\n"
        "- Include 'evidence' text and 'evidence_timestamps_sec' for every observation.\n"
        "- Set 'human_review_required' to true when visibility is poor or evidence conflicts between frames.\n"
        "- Return ONLY a valid JSON object. Do NOT output markdown fences or explanatory text.\n"
    )

    if incident_type is None:
        prompt = common_header + f"""
TASK: Analyze the clip and classify the primary safety incident visible. Do not use the filename.
Choose exactly one incident_type based only on the sampled frames:
- 'ppe_compliance': helmet or high-visibility vest compliance.
- 'restricted_zone': entry into or crossing of a marked restricted/unauthorized area.
- 'osha_hazard': another visible workplace hazard such as fall exposure or forklift/pedestrian proximity.

Classification priority:
1. Choose 'restricted_zone' if a worker is entering or crossing a restricted boundary.
2. Otherwise choose 'osha_hazard' if any non-PPE workplace hazard is visible, such as an unprotected edge or unsafe forklift/pedestrian proximity.
3. Choose 'ppe_compliance' only when no restricted-zone or other OSHA hazard is visible. Treat helmet/vest observations as secondary when the primary incident is restricted-zone or OSHA.

After classification, return the observations required for that category:
- ppe_compliance: exactly 'helmet' and 'high_visibility_vest', with statuses present/missing/uncertain/not_observed.
- restricted_zone: exactly 'zone_boundary_crossing' and 'authorization_indicators'.
- osha_hazard: at least one hazard observation, using detected/not_detected/uncertain/not_observed.

Restricted-zone status rules:
- 'zone_boundary_crossing' uses detected/not_detected/uncertain/not_observed.
- 'authorization_indicators' uses present/missing/uncertain/not_observed.
- A warning sign or restricted-area label describes the boundary; it does NOT prove that the worker is authorized.
- Use 'present' only for a visible worker badge, access credential, escort, or other direct authorization evidence.
- Use 'missing' when the worker is visible but no direct authorization evidence is visible.

PPE status rules:
- Use 'missing' when the head or torso is clearly visible and the PPE item is absent.
- Use 'not_observed' only when that body region is not visible in any sampled frame.

If no incident is clearly supported, choose the closest category, use uncertain or not_observed,
and set 'alert_assessment' to 'inconclusive'. Do not infer facts that are not visible.

JSON Output Schema:
{{
  "summary": "Short factual description of the visible event.",
  "visible_activity": "What workers visibly do.",
  "incident_type": "ppe_compliance | restricted_zone | osha_hazard",
  "alert_assessment": "supported | contradicted | inconclusive",
  "safety_observations": [
    {{
      "item": "Category-specific observation item",
      "visible_status": "Category-specific status",
      "evidence": "Detailed visible evidence.",
      "evidence_timestamps_sec": [2.0, 4.0]
    }}
  ],
  "recommended_follow_up": "Actionable follow-up recommendation.",
  "human_review_required": false
}}
"""
    elif incident_type == IncidentType.PPE_COMPLIANCE:
        prompt = common_header + f"""
TASK: Evaluate Personal Protective Equipment (PPE) compliance for each worker visible.
Evaluate 'helmet' and 'high_visibility_vest' INDEPENDENTLY for each worker.

Allowed values:
- 'alert_assessment': 'supported', 'contradicted', 'inconclusive'
- 'visible_status': 'present', 'missing', 'uncertain', 'not_observed'
  - 'missing': body region is clearly visible but the PPE item is NOT worn.
  - 'present': PPE item is clearly worn.

JSON Output Schema:
{{
  "summary": "Short factual description of visible event.",
  "visible_activity": "What the worker visibly does.",
  "incident_type": "ppe_compliance",
  "alert_assessment": "supported | contradicted | inconclusive",
  "safety_observations": [
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

    elif incident_type == IncidentType.RESTRICTED_ZONE:
        prompt = common_header + f"""
TASK: Assess whether a worker has entered or is at the boundary of a restricted or unauthorized zone.
Look for: marked floor lines or tape, warning signs ("Restricted Area", "Authorized Personnel Only",
"No Entry", etc.), red/yellow safety barriers, locked gates, or visual boundary markers.

Allowed values:
- 'alert_assessment': 'supported' (worker clearly inside or crossing boundary),
  'contradicted' (worker clearly outside), 'inconclusive' (ambiguous position)
- 'visible_status': 'detected', 'not_detected', 'uncertain', 'not_observed'
  - 'detected': boundary crossing or zone entry is clearly visible.
  - 'not_detected': worker is visibly outside the restricted boundary.

Provide two observations:
1. 'zone_boundary_crossing': whether the worker has crossed or is crossing the restricted zone boundary.
2. 'authorization_indicators': any visible PPE, badges, signage or access control features that indicate authorization status.

JSON Output Schema:
{{
  "summary": "Short factual description of the visible event.",
  "visible_activity": "What the worker visibly does near the restricted zone.",
  "incident_type": "restricted_zone",
  "alert_assessment": "supported | contradicted | inconclusive",
  "safety_observations": [
    {{
      "item": "zone_boundary_crossing",
      "visible_status": "detected | not_detected | uncertain | not_observed",
      "evidence": "Describe the boundary markers visible and the worker's position relative to them.",
      "evidence_timestamps_sec": [2.0, 4.0]
    }},
    {{
      "item": "authorization_indicators",
      "visible_status": "present | missing | uncertain | not_observed",
      "evidence": "Describe any visible authorization indicators (PPE, badges, escort, signage).",
      "evidence_timestamps_sec": [2.0, 4.0]
    }}
  ],
  "recommended_follow_up": "Actionable follow-up recommendation.",
  "human_review_required": false
}}
"""

    elif incident_type == IncidentType.OSHA_HAZARD:
        prompt = common_header + f"""
TASK: Identify visible OSHA-related safety hazards. Assess each visible hazard independently.

Common hazard types to look for (use the most appropriate item name):
- 'fall_protection_hazard': worker at height or near unprotected edge without visible harness, guardrail, or fall arrest system.
- 'forklift_pedestrian_proximity': forklift or powered industrial truck operating in close proximity to pedestrians without adequate separation.
If other OSHA hazards are visible, include them with a descriptive item name (e.g. 'unsecured_load', 'blocked_emergency_exit').

Allowed values:
- 'alert_assessment': 'supported', 'contradicted', 'inconclusive'
- 'visible_status': 'detected', 'not_detected', 'uncertain', 'not_observed'
  - 'detected': hazard is clearly visible in the frames.
  - 'not_detected': hazard type was looked for but not observed.

IMPORTANT: Report PPE compliance (helmet, vest) only as secondary observations if clearly relevant to the hazard scene.

JSON Output Schema:
{{
  "summary": "Short factual description of the visible safety hazard.",
  "visible_activity": "What workers and equipment are visibly doing.",
  "incident_type": "osha_hazard",
  "alert_assessment": "supported | contradicted | inconclusive",
  "safety_observations": [
    {{
      "item": "fall_protection_hazard | forklift_pedestrian_proximity | <other_hazard>",
      "visible_status": "detected | not_detected | uncertain | not_observed",
      "evidence": "Describe exactly what makes this a hazard, with visible details.",
      "evidence_timestamps_sec": [2.0, 5.0]
    }}
  ],
  "recommended_follow_up": "Actionable follow-up recommendation referencing OSHA standards where applicable.",
  "human_review_required": false
}}
"""
    else:
        raise ValueError(f"Unknown incident type: {incident_type}")

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
        incident_type: Optional[IncidentType] = IncidentType.PPE_COMPLIANCE,
        alert_type: Optional[str] = None,
        alert_offset_sec: Optional[float] = None,
    ) -> SafetySummary:
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
            incident_type=incident_type,
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
        summary = validate_summary_json(
            message_content,
            video_duration=duration_sec,
            incident_type=incident_type,
        )
        return summary


# =====================================================================
# End-to-End Summarizer Pipeline
# =====================================================================

def summarize_event(
    video_path: Path,
    event_id: Optional[str] = None,
    incident_type: Optional[IncidentType] = None,
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
) -> Tuple[bool, Optional[SafetySummary]]:
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
                summary = validate_summary_json(
                    existing_record["summary_json"],
                    incident_type=incident_type,
                )
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
            incident_type=incident_type,
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


def print_summary(event_id: str, summary: SafetySummary) -> None:
    """Print formatted summary with evidence timestamps for review."""
    incident_label = summary.incident_type.upper().replace("_", " ")
    print("\n" + "=" * 70)
    print(f" SAFETY EVENT SUMMARY [{incident_label}]: {event_id}")
    print("=" * 70)
    print(f" Summary:          {summary.summary}")
    print(f" Visible Activity: {summary.visible_activity}")
    print(f" Alert Assessment: {summary.alert_assessment.value.upper()}")
    print("-" * 70)
    print(" SAFETY OBSERVATIONS:")
    for obs in summary.safety_observations:
        ts_str = ", ".join(f"{t:.2f}s" for t in obs.evidence_timestamps_sec)
        print(f"  • {obs.item:28s}: [{obs.visible_status.value.upper()}]")
        print(f"    Evidence:   {obs.evidence}")
        print(f"    Timestamps: [{ts_str}]")
    print("-" * 70)
    print(f" Recommended Follow-up: {summary.recommended_follow_up}")
    print(f" Human Review Required: {'YES' if summary.human_review_required else 'NO'}")
    print("=" * 70 + "\n")


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
        "--incident-type",
        type=str,
        default="auto",
        choices=["auto", *[t.value for t in IncidentType]],
        help=(
            "Incident category: 'auto' asks the model to classify the clip; or use "
            "'ppe_compliance', 'restricted_zone', or 'osha_hazard'."
        ),
    )
    parser.add_argument(
        "--alert-type",
        type=str,
        default=None,
        help="Upstream detector alert type (e.g., 'no_helmet', 'restricted_zone_entry', 'fall_protection').",
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
        requested_incident_type = (
            None if args.incident_type == "auto" else IncidentType(args.incident_type)
        )
        # Alert metadata is supplied by an upstream detector when available;
        # it is never guessed from the video filename.
        derived_alert = args.alert_type

        logger.info(
            "Event %s → incident_type=%s, alert_type=%s",
            eid, requested_incident_type.value if requested_incident_type else "auto", derived_alert,
        )
        display_type = requested_incident_type.value if requested_incident_type else "auto"
        print(f"\nProcessing: {video_file.name} (Event ID: {eid}, Type: {display_type})")
        try:
            processed_new, summary = summarize_event(
                video_path=video_file,
                event_id=eid,
                incident_type=requested_incident_type,
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
