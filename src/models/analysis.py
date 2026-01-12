from pydantic import BaseModel, Field
from src.models.segment import TranscribedSegment


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
