"""
    Transcribe audio segments using OpenAI's whisper model.
    
    Args:
        segments (list[Segment]): List of segments with start, end times and speaker labels,
        clip_paths (list¨[Path]): list of paths to audio clips corresponding to segments 
        model_name (str | None): Whisper model name to use. If None, uses default from settings.
        language (str | None): Language code for transcription. If None or "auto", detects language
        
    Returns:
        tuple[list[TranscribedSegment], str]: list of transcribed segments and detected language code.
        
    Raises:
        None. Assumes valid input segments and clip paths.
    
    Note:
        - Uses OpenAI's whisper for transcription
        - Auto-detects language from first substantial clip if language is None or "auto"
        - Skips transcription for clips shorter than minimum duration from settings
        - Logs key steps for transparency
        
    Example:
        >>> segments = [Segment(start=0.0, end=5,0, speaker="Speaker 1"), Segment(start=5.0, end=10.0, speaker="Speaker 2"
        >>> clip_paths = [Path("/path/to/part_0.wav"), Path("/path/to/part_1.wav")]
        >>> transcribed_segments, detected_lang = transcribe_segments(segments, clip_paths, model_name="base", language="auto")
        >>> for seg in transcribed_segments:
        >>>     print(seg)
"""

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


def transcribe_segments(segments: list[Segment],clip_paths: list[Path],
model_name: str | None = None,
    language: str | None = None,
) -> tuple[list[TranscribedSegment], str]:

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
