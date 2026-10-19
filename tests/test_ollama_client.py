"""
tests/test_ollama_client.py — Unit tests mocking Ollama client and checking prompt construction.
"""

from unittest.mock import MagicMock, patch
import pytest
import requests
from ppe_event_summarizer import (
    IncidentType,
    OllamaClient,
    SampledFrame,
    PPESummary,
    build_ollama_prompt,
)

MOCK_RAW_RESPONSE = {
    "summary": "Worker is visible in the corridor.",
    "visible_activity": "Walking across the work area.",
    "alert_assessment": "supported",
    "ppe_observations": [
        {
            "item": "helmet",
            "visible_status": "missing",
            "evidence": "Head bare.",
            "evidence_timestamps_sec": [1.0, 2.0],
        },
        {
            "item": "high_visibility_vest",
            "visible_status": "present",
            "evidence": "Vest worn.",
            "evidence_timestamps_sec": [1.0, 2.0],
        },
    ],
    "recommended_follow_up": "Check credentials.",
    "human_review_required": False,
}


def test_prompt_does_not_contain_expected_labels():
    prompt = build_ollama_prompt(
        duration_sec=10.0,
        sampled_timestamps=[1.0, 3.0, 5.0],
        alert_type="no_helmet",
        alert_offset_sec=3.0,
    )
    # Must NOT bias the model with expected answers
    assert "Worker without helmet and without vest" not in prompt
    assert "Intended condition" not in prompt
    assert "ground truth" not in prompt.lower()
    # Must instruct independent assessment
    assert "INDEPENDENTLY" in prompt


def test_auto_prompt_classifies_without_filename():
    prompt = build_ollama_prompt(
        duration_sec=10.0,
        sampled_timestamps=[1.0, 5.0, 9.0],
        incident_type=None,
    )
    assert "Do not use the filename" in prompt
    assert "ppe_compliance | restricted_zone | osha_hazard" in prompt
    assert "A warning sign or restricted-area label describes the boundary" in prompt
    assert "choose 'osha_hazard' if any non-PPE workplace hazard is visible" in prompt


def test_explicit_incident_type_remains_supported():
    prompt = build_ollama_prompt(
        duration_sec=10.0,
        sampled_timestamps=[1.0],
        incident_type=IncidentType.OSHA_HAZARD,
    )
    assert "Identify visible OSHA-related safety hazards" in prompt


@patch("requests.post")
def test_ollama_client_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    import json
    mock_resp.json.return_value = {
        "message": {"content": json.dumps(MOCK_RAW_RESPONSE)}
    }
    mock_post.return_value = mock_resp

    client = OllamaClient(host="http://localhost:11434", model="qwen3-vl:4b")
    dummy_frame = SampledFrame(
        timestamp_sec=1.0,
        frame_index=30,
        image_bytes=b"fake_jpeg",
        image_base64="ZmFrZV9qcGVn",
    )

    summary = client.generate_summary(
        frames=[dummy_frame],
        duration_sec=5.0,
        alert_type="no_helmet",
    )

    assert isinstance(summary, PPESummary)
    assert summary.alert_assessment.value == "supported"
    assert len(summary.safety_observations) == 2

    # Check payload sent to mock_post
    call_args = mock_post.call_args[1]
    payload = call_args["json"]
    assert payload["model"] == "qwen3-vl:4b"
    assert payload["format"] == "json"
    assert len(payload["messages"][0]["images"]) == 1
    # Verify base64 was sent to Ollama
    assert payload["messages"][0]["images"][0] == "ZmFrZV9qcGVn"


@patch("requests.post")
def test_ollama_client_network_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    client = OllamaClient(host="http://localhost:11434")
    dummy_frame = SampledFrame(
        timestamp_sec=1.0,
        frame_index=30,
        image_bytes=b"fake",
        image_base64="ZmFrZQ==",
    )

    with pytest.raises(RuntimeError, match="Failed to communicate with Ollama"):
        client.generate_summary(frames=[dummy_frame], duration_sec=5.0)


@patch("requests.post")
def test_ollama_client_malformed_json_response(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {"content": "This is definitely not JSON!"}
    }
    mock_post.return_value = mock_resp

    client = OllamaClient(host="http://localhost:11434")
    dummy_frame = SampledFrame(
        timestamp_sec=1.0,
        frame_index=30,
        image_bytes=b"fake",
        image_base64="ZmFrZQ==",
    )

    with pytest.raises(ValueError, match="Malformed JSON"):
        client.generate_summary(frames=[dummy_frame], duration_sec=5.0)
