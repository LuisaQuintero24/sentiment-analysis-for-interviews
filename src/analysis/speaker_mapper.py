import logging
from collections import Counter

from src.models.segment import TranscribedSegment

logger = logging.getLogger(__name__)

""" Module for mapping speakers in interview transcripts.
    Provides functions to map speakers to roles based on frequency.
    
    Args: 
        segments (list[TranscribedSegment]): list of transcribed segments with speaker information
    
    Returns: 
        dict¨[str, str]: mapping of original speaker names to roles ("Interviewer", "Interviewee")
        
    Raises: 
        None. Assumes valid input segments.
    
    Note:
        - Maps least frequent speaker to "Interviewer" and others to "Interviewee"
        - Uses Counter to count ocurrences of each speaker
        - Logs mapping decisions for transparency
        - If no segments provided, returns empty mapping. 
        
    Example:
        >>> segments = ¨[TranscribedSegment(speaker="Alice", text="..."), TranscribedSegment(speaker=" Bob", text="..."), TranscribedSegment(speaker="Alice", text="...")]
        >>> speaker_map = map_speakers(segments)
        >>> print(speaker_map)"""




def map_speakers(segments: list[TranscribedSegment]) -> dict[str, str]:
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
