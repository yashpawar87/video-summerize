"""
tests/test_frame_extraction.py — Tests for frame extraction, timestamps, and resizing.
"""

from pathlib import Path
import pytest
from ppe_event_summarizer import (
    calculate_sample_timestamps,
    extract_frames,
    MAX_FRAMES_LIMIT,
    MAX_FRAME_WIDTH,
)


def test_calculate_sample_timestamps_even():
    duration = 10.0
    ts = calculate_sample_timestamps(duration_sec=duration, max_frames=8, alert_offset_sec=None)
    assert len(ts) <= 8
    assert all(0.0 <= t <= duration for t in ts)
    # Ensure sorted strictly increasing
    assert ts == sorted(ts)


def test_calculate_sample_timestamps_with_alert_offset():
    duration = 10.0
    alert_offset = 4.5
    ts = calculate_sample_timestamps(duration_sec=duration, max_frames=8, alert_offset_sec=alert_offset)
    assert len(ts) <= 8
    assert all(0.0 <= t <= duration for t in ts)
    # Check that timestamps near the alert offset are included
    near_alert = [t for t in ts if abs(t - alert_offset) <= 1.5]
    assert len(near_alert) >= 1


def test_calculate_sample_timestamps_bounds():
    # Test edge cases: 0 duration, negative duration
    assert calculate_sample_timestamps(0.0) == [0.0]
    assert calculate_sample_timestamps(-5.0) == [0.0]

    # Test max_frames constraint capped at 8
    ts = calculate_sample_timestamps(20.0, max_frames=16)
    assert len(ts) <= MAX_FRAMES_LIMIT


def test_extract_frames_valid_video():
    video_path = Path("ppe_test_dataset/helmet_missing.mp4")
    if not video_path.exists():
        pytest.skip(f"Test video not found: {video_path}")

    timestamps = [1.0, 3.0, 5.0]
    frames = extract_frames(video_path, timestamps=timestamps, max_width=MAX_FRAME_WIDTH)

    assert len(frames) == len(timestamps)
    for frame in frames:
        assert frame.timestamp_sec in timestamps
        assert len(frame.image_bytes) > 0
        assert len(frame.image_base64) > 0
        # Check valid base64
        import base64
        decoded = base64.b64decode(frame.image_base64)
        assert decoded == frame.image_bytes


def test_extract_frames_limit_enforced():
    video_path = Path("ppe_test_dataset/helmet_missing.mp4")
    if not video_path.exists():
        pytest.skip(f"Test video not found: {video_path}")

    # Pass 12 timestamps, must be capped at 8
    excess_timestamps = [float(i) for i in range(12)]
    frames = extract_frames(video_path, timestamps=excess_timestamps)
    assert len(frames) <= MAX_FRAMES_LIMIT
