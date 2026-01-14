""" Configuration settings for the audio processing and analysis pipeline

Args:
    audio: settings related to audio processing and transcription
    analysis: settings related to next analysis tasks
    thresholds: confidence thresholds for various classifiers
    output: settings for output formatting and content
    logging: settings for logging verbosity and format

Returns:
    Settings: dataclass encapsulating all configuration settings

Raises:
    None. Uses default values if no config file is found.

Note:
    - Loads settings from config.yaml if present
    - Uses pydantic for data validation and default values
    - Caches settings instance for performance

Example:
    >>> settings = get_settings()
    >>> print( settings.audio.whisper_model)

   """

import logging
from pathlib import Path
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field
from rich.logging import RichHandler

from src.utils.progress import console


class AudioSettings(BaseModel):
    whisper_model: str = "small"
    sample_rate: int = 16000
    min_segment_duration: float = 0.1


class AnalysisSettings(BaseModel):
    question_model: str = "facebook/bart-large-mnli"
    default_language: str = "auto"


class ThresholdSettings(BaseModel):
    question_confidence: float = Field(default=0.5, ge=0, le=1)


class OutputSettings(BaseModel):
    format: str = "json"
    include_probabilities: bool = True


class LoggingSettings(BaseModel):
    level: str = "INFO"


class Settings(BaseModel):
    audio: AudioSettings = AudioSettings()
    analysis: AnalysisSettings = AnalysisSettings()
    thresholds: ThresholdSettings = ThresholdSettings()
    output: OutputSettings = OutputSettings()
    logging: LoggingSettings = LoggingSettings()


@lru_cache
def get_settings() -> Settings:
    config_path = Path(__file__).parent.parent.parent / "config.yaml"

    if config_path.exists():
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
            return Settings(**config_data)

    return Settings()


def setup_logging(settings: Settings | None = None) -> logging.Logger:
    if settings is None:
        settings = get_settings()

    level = getattr(logging, settings.logging.level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    return logging.getLogger("interview_analyzer")
