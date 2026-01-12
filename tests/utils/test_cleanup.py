"""Tests for utils.cleanup module."""

from src.utils.cleanup import cleanup_folders


def test_cleanup_creates_interim_folder(tmp_path):
    """Test that interim folder is created."""
    script_dir = tmp_path
    interim_dir = script_dir / "data" / "interim"

    cleanup_folders(script_dir)

    assert interim_dir.exists()
    assert interim_dir.is_dir()


def test_cleanup_removes_files_from_interim(tmp_path):
    """Test that files are removed from interim folder."""
    script_dir = tmp_path
    interim_dir = script_dir / "data" / "interim"
    interim_dir.mkdir(parents=True)
    test_file = interim_dir / "test.rttm"
    test_file.write_text("test content")

    cleanup_folders(script_dir)

    assert interim_dir.exists()
    assert not test_file.exists()


def test_cleanup_removes_specific_output_files(tmp_path):
    """Test that specific output files are removed."""
    script_dir = tmp_path
    output_dir = script_dir / "data" / "output"
    output_dir.mkdir(parents=True)

    files_to_clean = ["audio_diarizado.json", "transcriptions.json"]

    for filename in files_to_clean:
        (output_dir / filename).write_text("test")

    cleanup_folders(script_dir)

    for filename in files_to_clean:
        assert not (output_dir / filename).exists()
