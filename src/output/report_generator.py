import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.models.analysis import AnalyzedSegment
from src.models.interview import (
    InterviewAnalysis,
    InterviewMetadata,
    InterviewReport,
    SentimentDistribution,
)

logger = logging.getLogger(__name__)


def _calculate_distribution(
    items: list[str], total: int
) -> dict[str, SentimentDistribution]:
    counts = Counter(items)
    return {
        label: SentimentDistribution(
            count=count, percentage=round(100 * count / total, 1) if total > 0 else 0.0
        )
        for label, count in counts.items()
    }


def generate_report(
    segments: list[AnalyzedSegment],
    duration_seconds: float,
    language: str,
    interview_id: str = "interview_001",
) -> InterviewAnalysis:

    participants = list(set(seg.speaker_role for seg in segments))
    questions = [s for s in segments if s.role == "question"]
    statements = [s for s in segments if s.role == "statement"]

    # Sentiment stats (only for statements/answers)
    sentiments = [s.sentiment.label for s in statements if s.sentiment]
    sentiment_dist = _calculate_distribution(sentiments, len(statements))

    # Emotion stats
    emotions = [s.emotion.label for s in statements if s.emotion]
    emotion_dist = _calculate_distribution(emotions, len(statements))

    # Average sentiment score
    scores = [s.sentiment.score for s in statements if s.sentiment]
    avg_score = round(sum(scores) / len(scores), 3) if scores else 0.5

    # Dominant labels
    dominant_sentiment = max(
        sentiment_dist, key=lambda k: sentiment_dist[k].count, default="N/A"
    )
    dominant_emotion = max(
        emotion_dist, key=lambda k: emotion_dist[k].count, default="N/A"
    )

    metadata = InterviewMetadata(
        date=datetime.now().strftime("%Y-%m-%d"),
        participants=participants,
        duration_seconds=round(duration_seconds, 2),
        language=language,
    )

    report = InterviewReport(
        total_segments=len(segments),
        total_questions=len(questions),
        total_statements=len(statements),
        sentiment_distribution=sentiment_dist,
        emotion_distribution=emotion_dist,
        average_sentiment_score=avg_score,
        dominant_sentiment=dominant_sentiment,
        dominant_emotion=dominant_emotion,
    )

    return InterviewAnalysis(
        interview_id=interview_id,
        metadata=metadata,
        segments=segments,
        report=report,
    )


def save_analysis(analysis: InterviewAnalysis, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis.model_dump(), f, ensure_ascii=False, indent=2)

    logger.info(f"Analysis saved to: {output_path}")
