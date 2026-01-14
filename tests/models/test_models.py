"""Tests for data models with validation and edge cases."""

import pytest
from pydantic import ValidationError

from src.models.segment import Segment, TranscribedSegment
from src.models.analysis import SentimentResult, EmotionResult, AnalyzedSegment
from src.models.interview import (
    SentimentDistribution,
    InterviewMetadata,
    InterviewReport,
    InterviewAnalysis,
)


class TestSegment:
    """Tests for Segment model."""

    def test_segment_creation(self):
        """Test basic segment creation."""
        seg = Segment(start=0.0, end=1.5, speaker="SPEAKER_00")
        assert seg.start == 0.0
        assert seg.end == 1.5
        assert seg.speaker == "SPEAKER_00"

    def test_segment_rejects_negative_start(self):
        """Test that negative start time is rejected."""
        with pytest.raises(ValidationError):
            Segment(start=-1.0, end=1.5, speaker="SPEAKER_00")

    def test_segment_rejects_negative_end(self):
        """Test that negative end time is rejected."""
        with pytest.raises(ValidationError):
            Segment(start=0.0, end=-1.0, speaker="SPEAKER_00")

    def test_segment_allows_zero_duration(self):
        """Test that zero-duration segments are allowed."""
        seg = Segment(start=1.0, end=1.0, speaker="SPEAKER_00")
        assert seg.start == seg.end

    def test_segment_serialization(self):
        """Test segment can be serialized to dict."""
        seg = Segment(start=0.0, end=1.5, speaker="SPEAKER_00")
        data = seg.model_dump()
        assert data == {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"}

    def test_segment_from_dict(self):
        """Test segment can be created from dict."""
        data = {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"}
        seg = Segment(**data)
        assert seg.start == 0.0


class TestTranscribedSegment:
    """Tests for TranscribedSegment model."""

    def test_transcribed_segment_inherits_from_segment(self):
        """Test that TranscribedSegment has all Segment fields."""
        seg = TranscribedSegment(
            start=0.0, end=1.5, speaker="SPEAKER_00", text="Hello", language="en"
        )
        assert seg.start == 0.0
        assert seg.text == "Hello"
        assert seg.language == "en"

    def test_transcribed_segment_default_language(self):
        """Test default language is 'unknown'."""
        seg = TranscribedSegment(
            start=0.0, end=1.5, speaker="SPEAKER_00", text="Hello"
        )
        assert seg.language == "unknown"

    def test_transcribed_segment_empty_text(self):
        """Test empty text is allowed."""
        seg = TranscribedSegment(
            start=0.0, end=1.5, speaker="SPEAKER_00", text="", language="en"
        )
        assert seg.text == ""

    def test_transcribed_segment_unicode_text(self):
        """Test unicode text is handled correctly."""
        seg = TranscribedSegment(
            start=0.0, end=1.5, speaker="SPEAKER_00",
            text="¿Cómo estás? 你好 ", language="mixed"
        )
        assert "¿Cómo" in seg.text
        assert "你好" in seg.text


class TestSentimentResult:
    """Tests for SentimentResult model."""

    def test_sentiment_result_creation(self):
        """Test basic creation."""
        result = SentimentResult(
            label="POS", score=0.85,
            probabilities={"POS": 0.85, "NEG": 0.1, "NEU": 0.05}
        )
        assert result.label == "POS"
        assert result.score == 0.85

    def test_sentiment_result_score_bounds(self):
        """Test score must be between 0 and 1."""
        with pytest.raises(ValidationError):
            SentimentResult(label="POS", score=1.5, probabilities={})

        with pytest.raises(ValidationError):
            SentimentResult(label="POS", score=-0.1, probabilities={})

    def test_sentiment_result_edge_scores(self):
        """Test boundary scores 0 and 1 are valid."""
        result_zero = SentimentResult(label="NEU", score=0.0, probabilities={})
        result_one = SentimentResult(label="POS", score=1.0, probabilities={})
        assert result_zero.score == 0.0
        assert result_one.score == 1.0


class TestEmotionResult:
    """Tests for EmotionResult model."""

    def test_emotion_result_creation(self):
        """Test basic creation."""
        result = EmotionResult(
            label="joy", score=0.7,
            probabilities={"joy": 0.7, "others": 0.3}
        )
        assert result.label == "joy"

    def test_emotion_result_score_bounds(self):
        """Test score validation."""
        with pytest.raises(ValidationError):
            EmotionResult(label="joy", score=2.0, probabilities={})


class TestAnalyzedSegment:
    """Tests for AnalyzedSegment model."""

    def test_analyzed_segment_creation(self):
        """Test full analyzed segment creation."""
        sentiment = SentimentResult(
            label="POS", score=0.8, probabilities={"POS": 0.8}
        )
        emotion = EmotionResult(
            label="joy", score=0.7, probabilities={"joy": 0.7}
        )
        seg = AnalyzedSegment(
            start=0.0, end=2.0, speaker="SPEAKER_00",
            text="I'm happy!", language="en",
            segment_id=0, role="statement", speaker_role="Interviewee",
            sentiment=sentiment, emotion=emotion, paired_with=None
        )
        assert seg.segment_id == 0
        assert seg.role == "statement"
        assert seg.sentiment.label == "POS"

    def test_analyzed_segment_optional_sentiment(self):
        """Test that sentiment and emotion can be None."""
        seg = AnalyzedSegment(
            start=0.0, end=2.0, speaker="SPEAKER_00",
            text="Short", language="en",
            segment_id=0, role="statement", speaker_role="Interviewee",
            sentiment=None, emotion=None, paired_with=None
        )
        assert seg.sentiment is None
        assert seg.emotion is None

    def test_analyzed_segment_paired_with(self):
        """Test paired_with field."""
        seg = AnalyzedSegment(
            start=0.0, end=2.0, speaker="A",
            text="Question?", language="en",
            segment_id=0, role="question", speaker_role="Interviewer",
            paired_with=1
        )
        assert seg.paired_with == 1


class TestInterviewMetadata:
    """Tests for InterviewMetadata model."""

    def test_metadata_creation(self):
        """Test basic metadata creation."""
        meta = InterviewMetadata(
            date="2024-01-15",
            participants=["Interviewer", "Candidate"],
            duration_seconds=1800.0,
            language="en"
        )
        assert meta.date == "2024-01-15"
        assert len(meta.participants) == 2
        assert meta.duration_seconds == 1800.0

    def test_metadata_empty_participants(self):
        """Test empty participants list is allowed."""
        meta = InterviewMetadata(
            date="2024-01-15", participants=[],
            duration_seconds=0.0, language="en"
        )
        assert meta.participants == []


class TestSentimentDistribution:
    """Tests for SentimentDistribution model."""

    def test_distribution_creation(self):
        """Test basic distribution creation."""
        dist = SentimentDistribution(count=10, percentage=0.5)
        assert dist.count == 10
        assert dist.percentage == 0.5


class TestInterviewReport:
    """Tests for InterviewReport model."""

    def test_report_creation(self):
        """Test full report creation."""
        report = InterviewReport(
            total_segments=20,
            total_questions=5,
            total_statements=15,
            sentiment_distribution={
                "POS": SentimentDistribution(count=10, percentage=0.5),
                "NEG": SentimentDistribution(count=5, percentage=0.25),
                "NEU": SentimentDistribution(count=5, percentage=0.25),
            },
            emotion_distribution={
                "joy": SentimentDistribution(count=8, percentage=0.4),
            },
            average_sentiment_score=0.65,
            dominant_sentiment="POS",
            dominant_emotion="joy"
        )
        assert report.total_segments == 20
        assert report.total_questions == 5
        assert report.dominant_sentiment == "POS"


class TestInterviewAnalysis:
    """Tests for InterviewAnalysis model."""

    def test_analysis_creation(self):
        """Test full interview analysis creation."""
        meta = InterviewMetadata(
            date="2024-01-15", participants=["A", "B"],
            duration_seconds=1800.0, language="en"
        )
        report = InterviewReport(
            total_segments=1, total_questions=0, total_statements=1,
            sentiment_distribution={}, emotion_distribution={},
            average_sentiment_score=0.5,
            dominant_sentiment="NEU", dominant_emotion="others"
        )
        seg = AnalyzedSegment(
            start=0, end=1, speaker="A", text="Hi", language="en",
            segment_id=0, role="statement", speaker_role="Interviewer"
        )
        analysis = InterviewAnalysis(
            interview_id="test-123",
            metadata=meta,
            segments=[seg],
            report=report
        )
        assert analysis.interview_id == "test-123"
        assert len(analysis.segments) == 1

    def test_analysis_serialization_roundtrip(self):
        """Test that analysis can be serialized and deserialized."""
        meta = InterviewMetadata(
            date="2024-01-15", participants=["A"],
            duration_seconds=100.0, language="en"
        )
        report = InterviewReport(
            total_segments=0, total_questions=0, total_statements=0,
            sentiment_distribution={}, emotion_distribution={},
            average_sentiment_score=0.0,
            dominant_sentiment="NEU", dominant_emotion="others"
        )
        analysis = InterviewAnalysis(
            interview_id="roundtrip-test",
            metadata=meta, segments=[], report=report
        )

        json_str = analysis.model_dump_json()
        restored = InterviewAnalysis.model_validate_json(json_str)

        assert restored.interview_id == "roundtrip-test"
        assert restored.metadata.language == "en"
