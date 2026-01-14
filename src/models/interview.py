from pydantic import BaseModel
from src.models.analysis import AnalyzedSegment

""" Data models for interview metadata, reports, and overall analysis.
    These models encapsulate the structure of interview-related data,
    including metadata, segment analyses, and summary reports.
    
    Args:
        date (str): Date of the interview
        participants (list[str]): List of participants in the interview
        duration_seconds (float): Duration of the interview in seconds
        language (str): Language of the interview
        total_segments (int): Total number of segments in the interview
        total_questions (int): Total number of questions asked
        total_statements (int): Total number of statements made
        sentiment_distribution (dict[str, SentimentDistribution]): Distribution of sentiments
        emotion_distribution (dict[str, SentimentDistribution]): Distribution of emotions
        average_sentiment_score (float): Average sentiment score across segments
        dominant_sentiment (str): Dominant sentiment in the interview
        dominant_emotion (str): Dominant emotion in the interview
        interview_id (str): Unique identifier for the interview
        metadata (InterviewMetadata): Metadata about the interview
        segments (list[AnalyzedSegment]): List of analyzed segments
        report (InterviewReport): Summary report of the interview analysis
        
    Returns: 
        InterviewMetadata: dataclass for interview metadata
        InterviewReport: dataclass for interview summary report
        InterviewAnalysis: dataclass encapsulating the full interview analysis 
    
    Raises:
        None. Assumes valid input data for interview analysis.
    
    Note:
        - Uses nested dataclasses for structured representation
        - Facilitates easy access to interview data and analysis results
        - Designed for extensibility to include additional analysis metrics
    
    Example:
        >>> metadata = InterviewMetadata(
                date="2024-01-15",
                participants=["Alice", "Bob"],
                duration_seconds=3600,
                language="es"
            )
        >>> report = InterviewReport(
                total_segments=100,
                total_questions=40,
                total_statements=60,
                sentiment_distribution={"POS": SentimentDistribution(count=50, percentage=50.0)},
                emotion_distribution={"joy": SentimentDistribution(count=30, percentage=30.0)},
                average_sentiment_score=0.65,
                dominant_sentiment="POS", 
                dominant_emotion="joy")

        >>> analysis = InterviewAnalysis(
                interview_id="interview_001",
                metadata=metadata,
                segments=[],
                report=report
            )
        >>> print(analysis)
    """


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
