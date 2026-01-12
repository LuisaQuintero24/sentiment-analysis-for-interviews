import logging
from pathlib import Path

import whisper
from langdetect import detect, LangDetectException
from pydub import AudioSegment

from src.config.settings import get_settings
from src.models.segment import Segment, TranscribedSegment

logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def transcribe_segments(
    segments: list[Segment],
    clip_paths: list[Path],
    model_name: str | None = None,
    language: str | None = None,
) -> tuple[list[TranscribedSegment], str]:
    """
    Transcribe audio clips using Whisper.

    Returns tuple of (transcribed_segments, detected_language).
    """
    settings = get_settings()
    model_name = model_name or settings.audio.whisper_model
    min_duration = settings.audio.min_segment_duration

    logger.info(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)

    # Auto-detect language from first substantial clip
    detected_lang = language
    if detected_lang is None or detected_lang == "auto":
        for path in clip_paths:
            clip = AudioSegment.from_wav(str(path))
            if len(clip) / 1000 >= min_duration:
                result = model.transcribe(str(path))
                detected_lang = result.get("language", "en")
                logger.info(f"Detected language: {detected_lang}")
                break
        else:
            detected_lang = "en"

    transcribed = []
    for idx, (seg, path) in enumerate(zip(segments, clip_paths)):
        clip = AudioSegment.from_wav(str(path))
        duration = len(clip) / 1000

        if duration < min_duration:
            text = ""
            logger.debug(f"Skipped {path.name} (too short: {duration:.3f}s)")
        else:
            result = model.transcribe(str(path), language=detected_lang)
            text = result.get("text", "").strip()
            logger.debug(f"Transcribed {path.name} ({duration:.3f}s)")

        transcribed.append(
            TranscribedSegment(
                start=seg.start,
                end=seg.end,
                speaker=seg.speaker,
                text=text,
                language=detected_lang,
            )
        )

    logger.info(f"Transcription complete: {len(transcribed)} segments")
    return transcribed, detected_lang
