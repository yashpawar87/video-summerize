"""
tests/test_schema_validation.py — Tests for Pydantic schema validation of model output.
"""

import pytest
from ppe_event_summarizer import (
    validate_summary_json,
    PPESummary,
    AlertAssessment,
    IncidentType,
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
    assert len(summary.safety_observations) == 2
    assert summary.safety_observations[0].visible_status == VisibleStatus.MISSING
    assert summary.safety_observations[1].visible_status == VisibleStatus.PRESENT
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
    items = [obs.item for obs in summary.safety_observations]
    assert "helmet" in items
    assert "high_visibility_vest" in items


def _base_safety_summary(incident_type, observations):
    return {
        "summary": "Visible safety event.",
        "visible_activity": "Worker activity is visible.",
        "incident_type": incident_type,
        "alert_assessment": "supported",
        "safety_observations": observations,
        "recommended_follow_up": "Review the event.",
        "human_review_required": False,
    }


def test_restricted_zone_summary_validation():
    data = _base_safety_summary("restricted_zone", [
        {
            "item": "zone_boundary_crossing",
            "visible_status": "detected",
            "evidence": "Worker crosses the marked boundary.",
            "evidence_timestamps_sec": [2.0],
        },
        {
            "item": "authorization_indicators",
            "visible_status": "missing",
            "evidence": "No badge or escort is visible.",
            "evidence_timestamps_sec": [2.0],
        },
    ])
    summary = validate_summary_json(data, video_duration=5.0, incident_type=IncidentType.RESTRICTED_ZONE)
    assert summary.incident_type == "restricted_zone"


def test_osha_hazard_summary_validation():
    data = _base_safety_summary("osha_hazard", [
        {
            "item": "fall_protection_hazard",
            "visible_status": "detected",
            "evidence": "An unprotected elevated edge is visible.",
            "evidence_timestamps_sec": [2.0],
        },
    ])
    summary = validate_summary_json(data, video_duration=5.0, incident_type=IncidentType.OSHA_HAZARD)
    assert summary.safety_observations[0].visible_status == VisibleStatus.DETECTED


def test_auto_validation_uses_model_incident_type():
    data = _base_safety_summary("restricted_zone", [
        {
            "item": "zone_boundary_crossing",
            "visible_status": "detected",
            "evidence": "Worker crosses the marked boundary.",
            "evidence_timestamps_sec": [2.0],
        },
        {
            "item": "authorization_indicators",
            "visible_status": "missing",
            "evidence": "No authorization indicator is visible.",
            "evidence_timestamps_sec": [2.0],
        },
    ])
    summary = validate_summary_json(data, video_duration=5.0, incident_type=None)
    assert summary.incident_type == "restricted_zone"


def test_auto_validation_requires_valid_model_incident_type():
    data = _base_safety_summary("unknown_category", [])
    with pytest.raises(ValueError, match="Automatic incident classification"):
        validate_summary_json(data, incident_type=None)


def test_incident_type_must_match_requested_type():
    data = _base_safety_summary("ppe_compliance", [
        {
            "item": "zone_boundary_crossing",
            "visible_status": "detected",
            "evidence": "Boundary crossing is visible.",
            "evidence_timestamps_sec": [2.0],
        },
        {
            "item": "authorization_indicators",
            "visible_status": "missing",
            "evidence": "No authorization indicator is visible.",
            "evidence_timestamps_sec": [2.0],
        },
    ])
    with pytest.raises(ValueError, match="does not match requested incident type"):
        validate_summary_json(data, video_duration=5.0, incident_type=IncidentType.RESTRICTED_ZONE)


def test_incident_specific_status_is_enforced():
    data = _base_safety_summary("osha_hazard", [
        {
            "item": "forklift_pedestrian_proximity",
            "visible_status": "present",
            "evidence": "Forklift is near a pedestrian.",
            "evidence_timestamps_sec": [2.0],
        },
    ])
    with pytest.raises(ValueError, match="must use one of"):
        validate_summary_json(data, video_duration=5.0, incident_type=IncidentType.OSHA_HAZARD)
