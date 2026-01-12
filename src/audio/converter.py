import logging
from pathlib import Path

from pydub import AudioSegment

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {
    ".mp3": "mp3",
    ".m4a": "m4a",
    ".mp4": "mp4",
    ".ogg": "ogg",
    ".flac": "flac",
    ".aac": "aac",
    ".wma": "wma",
    ".aiff": "aiff",
    ".webm": "webm",
}


def find_audio_file(raw_dir: Path) -> Path | None:
    for ext in SUPPORTED_FORMATS:
        candidates = list(raw_dir.glob(f"*{ext}"))
        if candidates:
            return candidates[0]
    return None


def convert_to_wav(source_path: Path, wav_path: Path) -> bool:
    try:
        ext = source_path.suffix.lower()
        fmt = SUPPORTED_FORMATS.get(ext)
        if not fmt:
            logger.error(f"Unsupported format: {ext}")
            return False

        logger.info(f"Converting {source_path.name} to WAV")
        audio = AudioSegment.from_file(str(source_path), format=fmt)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        audio.export(str(wav_path), format="wav")
        logger.info(f"Conversion complete: {wav_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to convert audio to WAV: {e}")
        return False


def ensure_wav_audio(raw_dir: Path, audio_wav: Path) -> bool:
    if audio_wav.exists():
        logger.debug(f"WAV file found: {audio_wav}")
        return True

    source = find_audio_file(raw_dir)
    if source:
        logger.info(f"Found audio file: {source.name}")
        return convert_to_wav(source, audio_wav)

    logger.error(f"No supported audio file found in {raw_dir}")
    logger.error(f"Supported formats: {', '.join(SUPPORTED_FORMATS.keys())}")
    return False
