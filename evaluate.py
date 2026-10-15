#!/usr/bin/env python3
"""
evaluate.py — Evaluation script for Local PPE Video Summarization

Compares model output stored in SQLite against the ground-truth manifest (reviewed labels).
Treats 'uncertain' as a distinct outcome rather than a match or an error.
Reports counts rather than percentages (sanity check on small dataset).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_MANIFEST_PATH = Path("ppe_test_dataset/manifest.json")
DEFAULT_DB_PATH = Path("data/ppe_events.db")
ITEMS_TO_EVALUATE = ["helmet", "high_visibility_vest"]


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load and validate manifest.json."""
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Manifest content must be a JSON object mapping filenames to labels.")

    return data


def get_latest_summary_for_clip(
    conn: sqlite3.Connection,
    clip_filename: str,
) -> Optional[Dict[str, Any]]:
    """Retrieve the latest completed event summary matching the clip filename."""
    stem = Path(clip_filename).stem
    cur = conn.execute(
        """
        SELECT e.event_id, e.status, e.video_path, s.model, s.prompt_version, s.summary_json
        FROM events e
        JOIN event_summaries s ON e.event_id = s.event_id
        WHERE e.status = 'completed'
          AND (
            e.video_path LIKE ?
            OR e.video_path LIKE ?
            OR e.event_id = ?
          )
        ORDER BY s.id DESC
        LIMIT 1;
        """,
        (f"%{clip_filename}", f"%{stem}%", stem),
    )
    row = cur.fetchone()
    if not row:
        return None
    return dict(row)


def extract_predictions(summary_json_str: str) -> Dict[str, Tuple[str, List[float], str]]:
    """
    Extract item predictions from summary JSON.
    Returns mapping: item -> (visible_status, evidence_timestamps, evidence_text).
    """
    try:
        data = json.loads(summary_json_str)
    except Exception as exc:
        raise ValueError(f"Failed to parse summary_json: {exc}") from exc

    observations = data.get("ppe_observations", [])
    mapping = {}
    for obs in observations:
        raw_item = obs.get("item", "").strip().lower().replace("-", "_").replace(" ", "_")
        if raw_item in ("vest", "high_vis_vest", "hi_vis_vest", "high_visibility_vest"):
            item_key = "high_visibility_vest"
        elif raw_item in ("hard_hat", "hardhat", "helmet", "safety_helmet"):
            item_key = "helmet"
        else:
            item_key = raw_item

        status = obs.get("visible_status", "uncertain").strip().lower()
        timestamps = obs.get("evidence_timestamps_sec", [])
        evidence = obs.get("evidence", "")
        mapping[item_key] = (status, timestamps, evidence)

    return mapping


def evaluate_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    db_path: Path = DEFAULT_DB_PATH,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate SQLite database against ground-truth manifest.
    """
    manifest = load_manifest(manifest_path)
    db_path = Path(db_path).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    results_table: List[Dict[str, Any]] = []
    counts = {
        "total_clips": len(manifest),
        "total_items": 0,
        "matches": 0,
        "mismatches": 0,
        "uncertain": 0,
        "pending_review": 0,
        "missing_in_db": 0,
    }

    try:
        for clip_filename, clip_meta in manifest.items():
            intended = clip_meta.get("intended", {})
            reviewed = clip_meta.get("reviewed", {})
            review_notes = clip_meta.get("review_notes", "")

            db_record = get_latest_summary_for_clip(conn, clip_filename)
            if not db_record:
                for item in ITEMS_TO_EVALUATE:
                    counts["total_items"] += 1
                    counts["missing_in_db"] += 1
                    results_table.append({
                        "clip": clip_filename,
                        "item": item,
                        "expected": reviewed.get(item),
                        "predicted": None,
                        "outcome": "NOT_PROCESSED_IN_DB",
                        "timestamps": [],
                        "evidence": "Event not found or not completed in database",
                    })
                continue

            predictions = extract_predictions(db_record["summary_json"])

            for item in ITEMS_TO_EVALUATE:
                counts["total_items"] += 1
                expected_reviewed = reviewed.get(item)
                pred_tuple = predictions.get(item)

                if pred_tuple is None:
                    pred_status = "not_reported"
                    pred_ts: List[float] = []
                    pred_evidence = "Item missing from model observations"
                else:
                    pred_status, pred_ts, pred_evidence = pred_tuple

                # Determine outcome
                # Rules:
                # 1. If reviewed is null -> pending manual review
                # 2. Treat 'uncertain' as distinct outcome rather than match or error
                # 3. Otherwise match or mismatch
                if expected_reviewed is None:
                    outcome = "PENDING_REVIEW"
                    counts["pending_review"] += 1
                elif pred_status == "uncertain":
                    outcome = "UNCERTAIN"
                    counts["uncertain"] += 1
                elif pred_status == expected_reviewed:
                    outcome = "MATCH"
                    counts["matches"] += 1
                else:
                    outcome = "MISMATCH"
                    counts["mismatches"] += 1

                results_table.append({
                    "clip": clip_filename,
                    "item": item,
                    "expected": expected_reviewed,
                    "predicted": pred_status,
                    "outcome": outcome,
                    "timestamps": pred_ts,
                    "evidence": pred_evidence,
                    "review_notes": review_notes,
                })

    finally:
        conn.close()

    return {
        "counts": counts,
        "results": results_table,
    }


def print_evaluation_report(eval_data: Dict[str, Any], verbose: bool = False) -> None:
    """Print readable CLI evaluation table and sanity check summary."""
    results = eval_data["results"]
    counts = eval_data["counts"]

    print("\n" + "=" * 90)
    print(" LOCAL PPE VIDEO EVALUATION REPORT (GROUND-TRUTH vs MODEL)")
    print("=" * 90)
    header = f"{'Clip':<32} {'Item':<22} {'Expected (Reviewed)':<20} {'Predicted':<14} {'Outcome'}"
    print(header)
    print("-" * 90)

    for row in results:
        exp_str = str(row['expected']) if row['expected'] is not None else "null (pending)"
        pred_str = str(row['predicted']) if row['predicted'] is not None else "N/A"
        outcome_str = row['outcome']

        # Color/highlight indicator in text
        if outcome_str == "MATCH":
            outcome_display = "[✔ MATCH]"
        elif outcome_str == "UNCERTAIN":
            outcome_display = "[? UNCERTAIN (distinct)]"
        elif outcome_str == "PENDING_REVIEW":
            outcome_display = "[⏳ PENDING REVIEW]"
        elif outcome_str == "MISMATCH":
            outcome_display = "[✖ MISMATCH]"
        else:
            outcome_display = f"[{outcome_str}]"

        print(f"{row['clip']:<32} {row['item']:<22} {exp_str:<20} {pred_str:<14} {outcome_display}")

        if verbose or outcome_str == "MISMATCH":
            ts_str = ", ".join(f"{t:.2f}s" for t in row['timestamps'])
            print(f"   ↳ Timestamps: [{ts_str}]")
            print(f"   ↳ Evidence:   {row['evidence']}")
            if row.get("review_notes"):
                print(f"   ↳ Notes:      {row['review_notes']}")

    print("=" * 90)
    print(" SANITY CHECK SUMMARY COUNTS (Note: small dataset, reporting counts not percentages):")
    print(f"   • Total clips evaluated:     {counts['total_clips']}")
    print(f"   • Total items assessed:      {counts['total_items']}")
    print(f"   • Matches:                   {counts['matches']}")
    print(f"   • Mismatches:                {counts['mismatches']}")
    print(f"   • Distinct Uncertain:        {counts['uncertain']}")
    print(f"   • Pending Manual Review:     {counts['pending_review']}")
    print(f"   • Missing in Database:       {counts['missing_in_db']}")
    print("=" * 90 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate completed PPE event summaries against manifest reviewed ground truth."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Path to manifest.json (default: {DEFAULT_MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose evidence explanations and timestamps for all items.",
    )

    args = parser.parse_args()

    try:
        eval_data = evaluate_manifest(
            manifest_path=args.manifest,
            db_path=args.db_path,
            verbose=args.verbose,
        )
        print_evaluation_report(eval_data, verbose=args.verbose)

        # Non-zero exit if there are mismatches or missing in DB
        counts = eval_data["counts"]
        if counts["missing_in_db"] > 0 or counts["mismatches"] > 0:
            return 1
        return 0

    except Exception as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
