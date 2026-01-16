from pydantic import BaseModel, Field
from src.models.segment import TranscribedSegment


""" Data models for analyzed segments in the interview analysis pipeline.
These models extend the base segment structures to include analysis results
such as sentiment and emotion classification.

Args:
    segment_id (int): Unique identifier for the segment
    role (str): Role of the segment (question, statement)
    speaker_role (str): Role of the speaker (Interviewer, Interviewee)
    sentiment (SentimentResult | None): Sentiment analysis result
    emotion (EmotionResult | None): Emotion analysis result
    paired_with (int | None): ID of paired segment if applicable

Returns:
    AnalyzedSegment: dataclass representing an analyzed segment with sentiment
    and emotion results

Raises:
    None. Assumes valid input data for segment analysis.
    Note:
        - Inherits from TranscribedSegment to include base transcription data
        - Includes optional sentiment and emotion analysis results
        - Paired segments are linked via segment IDs
    
Example:
    >>> analyzed_segment = AnalyzedSegment(
            segment_id=1,
            speaker="Alice",
            text="¿Cómo te llamas?",
            role="question",
            speaker_role= "Interviewer",
            sentiment=None, 
            emotion=None,
            paired_with=None
        )
        
        print
    """

class SentimentResult(BaseModel):
    label: str  # POS, NEG, NEU
    score: float = Field(ge=0, le=1)
    probabilities: dict[str, float]


class EmotionResult(BaseModel):
    label: str  # joy, anger, sadness, fear, disgust, surprise, others
    score: float = Field(ge=0, le=1)
    probabilities: dict[str, float]


class AnalyzedSegment(TranscribedSegment):
    segment_id: int
    role: str  # question, statement
    speaker_role: str  # Interviewer, Interviewee
    sentiment: SentimentResult | None = None
    emotion: EmotionResult | None = None
    paired_with: int | None = None
