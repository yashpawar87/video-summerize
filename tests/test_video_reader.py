"""
tests/test_video_reader.py — Tests for video loading and metadata extraction.
"""

from pathlib import Path
import pytest
from ppe_event_summarizer import get_video_metadata, VideoMetadata


def test_missing_video_path(tmp_path: Path):
    missing_path = tmp_path / "non_existent_video.mp4"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        get_video_metadata(missing_path)


def test_corrupt_or_non_video_file(tmp_path: Path):
    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_text("This is not a video file", encoding="utf-8")
    with pytest.raises(ValueError, match=".*(could not open video file|Invalid video stream).*"):
        get_video_metadata(corrupt_file)


def test_valid_video_metadata():
    video_path = Path("ppe_test_dataset/helmet_missing.mp4")
    if not video_path.exists():
        pytest.skip(f"Test video not found: {video_path}")

    meta = get_video_metadata(video_path)
    assert isinstance(meta, VideoMetadata)
    assert meta.duration_seconds > 0.0
    assert meta.fps > 0.0
    assert meta.frame_count > 0
    assert meta.width > 0
    assert meta.height > 0
    assert meta.video_path == video_path.resolve()
