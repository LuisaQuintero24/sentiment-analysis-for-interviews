from pydantic import BaseModel, Field

""" Data models for basic segment structures in the interview analysis pipeline.
These models define the foundational attributes of segments, including timing,
speaker information, and transcription details.
Args:
    start (float): Start time of the segment in seconds
    end (float): End time of the segment in seconds
    speaker (str): Identifier for the speaker of the segment

Returns:
    Segment: dataclass representing a basic audio segment with timing and speaker
    TranscribedSegment: dataclass extending Segment to include transcription text
    and detected language

Raises:
    None. Assumes valid input data for segment creation.
    
Note:
    - Uses pydantic for data validation and default values
    - Facilitates easy extension for additional segment attributes
    
Example:
    >>> segment = Segment(start=0.0, end=5.0, speaker="Speaker 1")
    >>> transcribed_segment = TranscribedSegment(
            start=0.0,
            end=5.0,
            speaker="Speaker 1",
            text="Hello, how are you?",
            language="en") 
    >>> print(segment)
    >>> print(transcribed_segment)
    
        """
class Segment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker: str


class TranscribedSegment(Segment):
    text: str
    language: str = "unknown"
