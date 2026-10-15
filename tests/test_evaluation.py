"""
tests/test_evaluation.py — Tests for evaluate.py against manifest and ground truth rules.
"""

import json
from pathlib import Path
import pytest
from evaluate import evaluate_manifest, extract_predictions
from ppe_event_summarizer import PPEDatabase, validate_summary_json

SAMPLE_SUMMARY_A = {
    "summary": "Worker 1 summary",
    "visible_activity": "Walking",
    "alert_assessment": "supported",
    "ppe_observations": [
        {
            "item": "helmet",
            "visible_status": "missing",
            "evidence": "No helmet visible.",
            "evidence_timestamps_sec": [1.0],
        },
        {
            "item": "high_visibility_vest",
            "visible_status": "missing",
            "evidence": "Dark shirt, no vest.",
            "evidence_timestamps_sec": [1.0],
        },
    ],
    "recommended_follow_up": "Check.",
    "human_review_required": False,
}

SAMPLE_SUMMARY_B = {
    "summary": "Worker 2 summary",
    "visible_activity": "Carrying tools",
    "alert_assessment": "supported",
    "ppe_observations": [
        {
            "item": "helmet",
            "visible_status": "missing",
            "evidence": "No helmet visible.",
            "evidence_timestamps_sec": [2.0],
        },
        {
            "item": "high_visibility_vest",
            "visible_status": "present",
            "evidence": "Yellow vest visible.",
            "evidence_timestamps_sec": [2.0],
        },
    ],
    "recommended_follow_up": "Check.",
    "human_review_required": False,
}


def test_extract_predictions():
    json_str = json.dumps(SAMPLE_SUMMARY_A)
    preds = extract_predictions(json_str)
    assert "helmet" in preds
    assert "high_visibility_vest" in preds
    assert preds["helmet"][0] == "missing"
    assert preds["high_visibility_vest"][0] == "missing"


def test_evaluation_with_pending_review(tmp_path: Path):
    manifest_data = {
        "clip1.mp4": {
            "intended": {"helmet": "missing", "high_visibility_vest": "missing"},
            "reviewed": {"helmet": None, "high_visibility_vest": None},
            "review_notes": "",
        }
    }
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    db_file = tmp_path / "test_eval.db"
    db = PPEDatabase(db_path=db_file)
    db.begin_event("clip1", tmp_path / "clip1.mp4")
    db.complete_event(
        event_id="clip1",
        duration_seconds=10.0,
        sampled_frame_count=4,
        model="qwen3-vl:4b",
        prompt_version="v1.0",
        summary=validate_summary_json(SAMPLE_SUMMARY_A),
    )

    eval_data = evaluate_manifest(manifest_path=manifest_file, db_path=db_file)
    counts = eval_data["counts"]
    assert counts["total_clips"] == 1
    assert counts["total_items"] == 2
    assert counts["pending_review"] == 2
    assert counts["matches"] == 0
    assert counts["mismatches"] == 0


def test_evaluation_with_reviewed_labels_match_and_uncertain(tmp_path: Path):
    manifest_data = {
        "clip1.mp4": {
            "intended": {"helmet": "missing", "high_visibility_vest": "missing"},
            "reviewed": {"helmet": "missing", "high_visibility_vest": "present"},
            "review_notes": "",
        }
    }
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    db_file = tmp_path / "test_eval.db"
    db = PPEDatabase(db_path=db_file)
    db.begin_event("clip1", tmp_path / "clip1.mp4")
    # In SAMPLE_SUMMARY_A: helmet is missing (matches reviewed), vest is missing (mismatch with reviewed 'present')
    db.complete_event(
        event_id="clip1",
        duration_seconds=10.0,
        sampled_frame_count=4,
        model="qwen3-vl:4b",
        prompt_version="v1.0",
        summary=validate_summary_json(SAMPLE_SUMMARY_A),
    )

    eval_data = evaluate_manifest(manifest_path=manifest_file, db_path=db_file)
    counts = eval_data["counts"]
    assert counts["matches"] == 1
    assert counts["mismatches"] == 1
    assert counts["pending_review"] == 0


def test_uncertain_is_distinct_outcome(tmp_path: Path):
    uncertain_summary = {
        **SAMPLE_SUMMARY_A,
        "ppe_observations": [
            {
                "item": "helmet",
                "visible_status": "uncertain",
                "evidence": "Too blurry.",
                "evidence_timestamps_sec": [1.0],
            },
            {
                "item": "high_visibility_vest",
                "visible_status": "present",
                "evidence": "Clear vest.",
                "evidence_timestamps_sec": [1.0],
            },
        ],
    }

    manifest_data = {
        "clip1.mp4": {
            "intended": {"helmet": "missing", "high_visibility_vest": "present"},
            "reviewed": {"helmet": "missing", "high_visibility_vest": "present"},
            "review_notes": "",
        }
    }
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    db_file = tmp_path / "test_eval.db"
    db = PPEDatabase(db_path=db_file)
    db.begin_event("clip1", tmp_path / "clip1.mp4")
    db.complete_event(
        event_id="clip1",
        duration_seconds=10.0,
        sampled_frame_count=4,
        model="qwen3-vl:4b",
        prompt_version="v1.0",
        summary=validate_summary_json(uncertain_summary),
    )

    eval_data = evaluate_manifest(manifest_path=manifest_file, db_path=db_file)
    counts = eval_data["counts"]
    # helmet should be counted as distinct uncertain, not mismatch or match
    assert counts["uncertain"] == 1
    assert counts["matches"] == 1
    assert counts["mismatches"] == 0
