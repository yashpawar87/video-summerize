"""
tests/test_database.py — Tests for SQLite persistence, idempotency, and status transitions.
"""

from pathlib import Path
import sqlite3
import pytest
from ppe_event_summarizer import (
    PPEDatabase,
    validate_summary_json,
)

SAMPLE_SUMMARY = {
    "summary": "Worker is visible.",
    "visible_activity": "Inspecting gear.",
    "alert_assessment": "supported",
    "ppe_observations": [
        {
            "item": "helmet",
            "visible_status": "missing",
            "evidence": "Bare head.",
            "evidence_timestamps_sec": [1.0],
        },
        {
            "item": "high_visibility_vest",
            "visible_status": "present",
            "evidence": "Vest worn.",
            "evidence_timestamps_sec": [1.0],
        },
    ],
    "recommended_follow_up": "Check.",
    "human_review_required": False,
}


def test_schema_initialization(tmp_path: Path):
    db_file = tmp_path / "test.db"
    db = PPEDatabase(db_path=db_file)

    conn = db._get_connection()
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row["name"] for row in cur.fetchall()]
    conn.close()

    assert "events" in tables
    assert "event_summaries" in tables


def test_foreign_keys_enabled(tmp_path: Path):
    db_file = tmp_path / "test_fk.db"
    db = PPEDatabase(db_path=db_file)
    conn = db._get_connection()
    cur = conn.execute("PRAGMA foreign_keys;")
    assert cur.fetchone()[0] == 1
    conn.close()


def test_idempotent_processing(tmp_path: Path):
    db_file = tmp_path / "test_idempotent.db"
    db = PPEDatabase(db_path=db_file)
    event_id = "test_event_01"
    video_path = tmp_path / "test.mp4"

    # First begin -> should process
    should_proc, _ = db.begin_event(event_id=event_id, video_path=video_path)
    assert should_proc is True

    # Complete event
    summary_obj = validate_summary_json(SAMPLE_SUMMARY)
    db.complete_event(
        event_id=event_id,
        duration_seconds=10.0,
        sampled_frame_count=4,
        model="qwen3-vl:4b",
        prompt_version="v1.0",
        summary=summary_obj,
    )

    # Second begin without force -> should NOT process
    should_proc, existing = db.begin_event(event_id=event_id, video_path=video_path, force=False)
    assert should_proc is False
    assert existing["status"] == "completed"

    # Third begin with force=True -> should process
    should_proc, _ = db.begin_event(event_id=event_id, video_path=video_path, force=True)
    assert should_proc is True


def test_event_failure(tmp_path: Path):
    db_file = tmp_path / "test_fail.db"
    db = PPEDatabase(db_path=db_file)
    event_id = "fail_event"
    video_path = tmp_path / "fail.mp4"

    db.begin_event(event_id=event_id, video_path=video_path)
    error_msg = "Corrupted frame reading error"
    db.fail_event(event_id=event_id, error_message=error_msg)

    event_record = db.get_event(event_id=event_id)
    assert event_record is not None
    assert event_record["status"] == "failed"
    assert event_record["error_message"] == error_msg


def test_cascade_deletion(tmp_path: Path):
    db_file = tmp_path / "test_cascade.db"
    db = PPEDatabase(db_path=db_file)
    event_id = "cascade_event"
    video_path = tmp_path / "test.mp4"

    db.begin_event(event_id=event_id, video_path=video_path)
    summary_obj = validate_summary_json(SAMPLE_SUMMARY)
    db.complete_event(
        event_id=event_id,
        duration_seconds=10.0,
        sampled_frame_count=4,
        model="qwen3-vl:4b",
        prompt_version="v1.0",
        summary=summary_obj,
    )

    with db._get_connection() as conn:
        conn.execute("DELETE FROM events WHERE event_id = ?", (event_id,))
        cur = conn.execute("SELECT COUNT(*) FROM event_summaries WHERE event_id = ?", (event_id,))
        count = cur.fetchone()[0]
        assert count == 0
