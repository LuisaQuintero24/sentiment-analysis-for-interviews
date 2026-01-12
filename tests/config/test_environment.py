"""Tests for config.environment module."""

import os
from unittest.mock import patch
from src.config.environment import setup_hf_token, setup_ffmpeg


def test_setup_hf_token_from_env():
    """Test setting HF token from environment variable."""
    test_token = "hf_test_token_123"
    with patch.dict(os.environ, {'HF_TOKEN': test_token}):
        setup_hf_token()
        assert os.environ['HF_TOKEN'] == test_token


def test_setup_ffmpeg_path_exists(tmp_path):
    """Test FFmpeg setup when path exists."""
    script_dir = tmp_path
    ffmpeg_dir = (
        script_dir / "engines" / "ffmpeg-2026-01-05-git-2892815c45-full_build" / "bin"
    )
    ffmpeg_dir.mkdir(parents=True)

    original_path = os.environ.get('PATH', '')
    setup_ffmpeg(script_dir)

    assert str(ffmpeg_dir) in os.environ['PATH']
    os.environ['PATH'] = original_path
