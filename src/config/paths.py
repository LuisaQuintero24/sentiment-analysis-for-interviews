"""Centralized path configuration for the audio processing pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectPaths:
    script_dir: Path
    raw_dir: Path
    audio_wav: Path
    rttm_file: Path
    segments_json: Path
    parts_dir: Path
    transcriptions_json: Path
    final_json: Path
    sentiment_json: Path


def get_project_paths(script_dir: Path) -> ProjectPaths:
    return ProjectPaths(
        script_dir=script_dir,
        raw_dir=script_dir / "data" / "raw",
        audio_wav=script_dir / "data" / "refined" / "audio.wav",
        rttm_file=script_dir / "data" / "interim" / "audio.rttm",
        segments_json=script_dir / "data" / "output" / "audio_diarizado.json",
        parts_dir=script_dir / "data" / "output" / "parts",
        transcriptions_json=script_dir / "data" / "output" / "transcriptions.json",
        final_json=script_dir / "data" / "output" / "audio_diarizado_transcribed.json",
        sentiment_json=script_dir / "data" / "output" / "sentiment_analysis.json",
    )
