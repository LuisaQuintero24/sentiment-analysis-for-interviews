from pydantic import BaseModel
from src.models.analysis import AnalyzedSegment


class SentimentDistribution(BaseModel):
    count: int
    percentage: float


class InterviewMetadata(BaseModel):
    date: str
    participants: list[str]
    duration_seconds: float
    language: str


class InterviewReport(BaseModel):
    total_segments: int
    total_questions: int
    total_statements: int
    sentiment_distribution: dict[str, SentimentDistribution]
    emotion_distribution: dict[str, SentimentDistribution]
    average_sentiment_score: float
    dominant_sentiment: str
    dominant_emotion: str


class InterviewAnalysis(BaseModel):
    interview_id: str
    metadata: InterviewMetadata
    segments: list[AnalyzedSegment]
    report: InterviewReport
