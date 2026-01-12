import logging
from collections import Counter

from src.models.segment import TranscribedSegment

logger = logging.getLogger(__name__)


def map_speakers(segments: list[TranscribedSegment]) -> dict[str, str]:
    """
    Map speaker codes to roles based on speaking frequency.
    The speaker with fewer segments is assumed to be the Interviewer.
    """
    if not segments:
        return {}

    speaker_counts = Counter(seg.speaker for seg in segments)
    sorted_speakers = speaker_counts.most_common()

    # Reverse: least frequent speaker first (interviewer)
    sorted_speakers = sorted_speakers[::-1]

    speaker_map = {}
    for i, (speaker, count) in enumerate(sorted_speakers):
        if i == 0:
            speaker_map[speaker] = "Interviewer"
            logger.debug(f"Mapped {speaker} -> Interviewer ({count} segments)")
        else:
            speaker_map[speaker] = "Interviewee"
            logger.debug(f"Mapped {speaker} -> Interviewee ({count} segments)")

    return speaker_map


def get_speaker_role(speaker: str, speaker_map: dict[str, str]) -> str:
    return speaker_map.get(speaker, speaker)
