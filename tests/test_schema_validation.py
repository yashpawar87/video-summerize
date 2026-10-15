"""
tests/test_schema_validation.py — Tests for Pydantic schema validation of model output.
"""

import pytest
from ppe_event_summarizer import (
    validate_summary_json,
    PPESummary,
    AlertAssessment,
    VisibleStatus,
)

VALID_SUMMARY_DICT = {
    "summary": "Worker is visible in the corridor.",
    "visible_activity": "Walking across the work area.",
    "alert_assessment": "supported",
    "ppe_observations": [
        {
            "item": "helmet",
            "visible_status": "missing",
            "evidence": "Worker's head has no hardhat or helmet visible.",
            "evidence_timestamps_sec": [1.5, 3.0],
        },
        {
            "item": "high_visibility_vest",
            "visible_status": "present",
            "evidence": "Fluorescent high-vis vest worn over dark shirt.",
            "evidence_timestamps_sec": [1.5, 3.0],
        },
    ],
    "recommended_follow_up": "Check worker safety badge.",
    "human_review_required": False,
}


def test_valid_summary_validation():
    summary = validate_summary_json(VALID_SUMMARY_DICT, video_duration=10.0)
    assert isinstance(summary, PPESummary)
    assert summary.alert_assessment == AlertAssessment.SUPPORTED
    assert len(summary.ppe_observations) == 2
    assert summary.ppe_observations[0].visible_status == VisibleStatus.MISSING
    assert summary.ppe_observations[1].visible_status == VisibleStatus.PRESENT
    assert not summary.human_review_required


def test_markdown_codeblock_parsing():
    import json
    json_str = f"```json\n{json.dumps(VALID_SUMMARY_DICT)}\n```"
    summary = validate_summary_json(json_str, video_duration=10.0)
    assert summary.alert_assessment == AlertAssessment.SUPPORTED


def test_missing_helmet_fails():
    invalid_dict = {
        **VALID_SUMMARY_DICT,
        "ppe_observations": [
            {
                "item": "high_visibility_vest",
                "visible_status": "present",
                "evidence": "Vest worn.",
                "evidence_timestamps_sec": [1.0],
            }
        ],
    }
    with pytest.raises(ValueError, match="must contain an entry for 'helmet'"):
        validate_summary_json(invalid_dict)


def test_missing_vest_fails():
    invalid_dict = {
        **VALID_SUMMARY_DICT,
        "ppe_observations": [
            {
                "item": "helmet",
                "visible_status": "missing",
                "evidence": "No helmet.",
                "evidence_timestamps_sec": [1.0],
            }
        ],
    }
    with pytest.raises(ValueError, match="must contain an entry for 'high_visibility_vest'"):
        validate_summary_json(invalid_dict)


def test_invalid_alert_assessment_enum():
    invalid_dict = {
        **VALID_SUMMARY_DICT,
        "alert_assessment": "unsupported_value",
    }
    with pytest.raises(ValueError, match="(validation error|Input should be)"):
        validate_summary_json(invalid_dict)


def test_invalid_visible_status_enum():
    invalid_dict = {
        **VALID_SUMMARY_DICT,
        "ppe_observations": [
            {
                "item": "helmet",
                "visible_status": "absent",  # Invalid enum, must be 'missing'
                "evidence": "Not there.",
                "evidence_timestamps_sec": [1.0],
            },
            {
                "item": "high_visibility_vest",
                "visible_status": "present",
                "evidence": "Vest worn.",
                "evidence_timestamps_sec": [1.0],
            },
        ],
    }
    with pytest.raises(ValueError, match="(validation error|Input should be)"):
        validate_summary_json(invalid_dict)


def test_out_of_range_evidence_timestamps():
    # Clip duration is 5.0 seconds, but evidence timestamp is 15.0 seconds
    invalid_dict = {
        **VALID_SUMMARY_DICT,
        "ppe_observations": [
            {
                "item": "helmet",
                "visible_status": "missing",
                "evidence": "No helmet.",
                "evidence_timestamps_sec": [15.0],
            },
            {
                "item": "high_visibility_vest",
                "visible_status": "present",
                "evidence": "Vest worn.",
                "evidence_timestamps_sec": [2.0],
            },
        ],
    }
    with pytest.raises(ValueError, match="falls outside clip duration"):
        validate_summary_json(invalid_dict, video_duration=5.0)


def test_item_name_normalization():
    # Model returns "vest" and "hardhat" instead of exact keys
    dict_with_variants = {
        **VALID_SUMMARY_DICT,
        "ppe_observations": [
            {
                "item": "hardhat",
                "visible_status": "missing",
                "evidence": "No hardhat.",
                "evidence_timestamps_sec": [1.0],
            },
            {
                "item": "hi-vis vest",
                "visible_status": "present",
                "evidence": "Vest worn.",
                "evidence_timestamps_sec": [1.0],
            },
        ],
    }
    summary = validate_summary_json(dict_with_variants, video_duration=10.0)
    items = [obs.item for obs in summary.ppe_observations]
    assert "helmet" in items
    assert "high_visibility_vest" in items
