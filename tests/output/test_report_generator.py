"""Tests for report generation module."""

import json
from datetime import datetime

import pytest

from src.models.analysis import AnalyzedSegment, SentimentResult, EmotionResult
from src.models.interview import InterviewAnalysis
from src.output.report_generator import (
    generate_report,
    save_analysis,
    _calculate_distribution,
)


class TestCalculateDistribution:
    """Tests for distribution calculation helper."""

    def test_calculate_distribution_basic(self):
        """Test basic distribution calculation."""
        items = ["POS", "POS", "NEG", "NEU"]
        result = _calculate_distribution(items, total=4)

        assert result["POS"].count == 2
        assert result["POS"].percentage == 50.0
        assert result["NEG"].count == 1
        assert result["NEG"].percentage == 25.0

    def test_calculate_distribution_empty(self):
        """Test distribution with empty list."""
        result = _calculate_distribution([], total=0)
        assert result == {}

    def test_calculate_distribution_zero_total(self):
        """Test distribution handles zero total gracefully."""
        items = ["POS"]
        result = _calculate_distribution(items, total=0)
        assert result["POS"].percentage == 0.0


class TestGenerateReport:
    """Tests for report generation."""

    @pytest.fixture
    def analyzed_segments(self):
        """Create sample analyzed segments for testing."""
        sentiment_pos = SentimentResult(
            label="POS", score=0.9, probabilities={"POS": 0.9}
        )
        sentiment_neg = SentimentResult(
            label="NEG", score=0.7, probabilities={"NEG": 0.7}
        )
        emotion_joy = EmotionResult(
            label="joy", score=0.8, probabilities={"joy": 0.8}
        )
        emotion_sad = EmotionResult(
            label="sadness", score=0.6, probabilities={"sadness": 0.6}
        )

        return [
            AnalyzedSegment(
                start=0, end=2, speaker="A", text="How are you?", language="en",
                segment_id=0, role="question", speaker_role="Interviewer",
                sentiment=None, emotion=None,
            ),
            AnalyzedSegment(
                start=2, end=5, speaker="B", text="I'm great!", language="en",
                segment_id=1, role="statement", speaker_role="Interviewee",
                sentiment=sentiment_pos, emotion=emotion_joy,
            ),
            AnalyzedSegment(
                start=5, end=8, speaker="A", text="What challenges?", language="en",
                segment_id=2, role="question", speaker_role="Interviewer",
                sentiment=None, emotion=None,
            ),
            AnalyzedSegment(
                start=8, end=12, speaker="B", text="It was hard.", language="en",
                segment_id=3, role="statement", speaker_role="Interviewee",
                sentiment=sentiment_neg, emotion=emotion_sad,
            ),
        ]

    def test_generate_report_returns_interview_analysis(self, analyzed_segments):
        """Test that generate_report returns InterviewAnalysis."""
        result = generate_report(
            segments=analyzed_segments,
            duration_seconds=120.5,
            language="en",
            interview_id="test-001",
        )

        assert isinstance(result, InterviewAnalysis)
        assert result.interview_id == "test-001"

    def test_generate_report_counts_segments(self, analyzed_segments):
        """Test segment counting."""
        result = generate_report(
            segments=analyzed_segments,
            duration_seconds=120.0,
            language="en",
        )

        assert result.report.total_segments == 4
        assert result.report.total_questions == 2
        assert result.report.total_statements == 2

    def test_generate_report_calculates_sentiment_distribution(self, analyzed_segments):
        """Test sentiment distribution calculation."""
        result = generate_report(
            segments=analyzed_segments,
            duration_seconds=120.0,
            language="en",
        )

        assert "POS" in result.report.sentiment_distribution
        assert "NEG" in result.report.sentiment_distribution
        assert result.report.sentiment_distribution["POS"].count == 1
        assert result.report.sentiment_distribution["NEG"].count == 1

    def test_generate_report_calculates_emotion_distribution(self, analyzed_segments):
        """Test emotion distribution calculation."""
        result = generate_report(
            segments=analyzed_segments,
            duration_seconds=120.0,
            language="en",
        )

        assert "joy" in result.report.emotion_distribution
        assert "sadness" in result.report.emotion_distribution

    def test_generate_report_calculates_average_score(self, analyzed_segments):
        """Test average sentiment score calculation."""
        result = generate_report(
            segments=analyzed_segments,
            duration_seconds=120.0,
            language="en",
        )

        expected_avg = (0.9 + 0.7) / 2
        assert result.report.average_sentiment_score == expected_avg

    def test_generate_report_identifies_dominant_sentiment(self):
        """Test dominant sentiment identification."""
        sentiment_pos = SentimentResult(label="POS", score=0.8, probabilities={})
        segments = [
            AnalyzedSegment(
                start=0, end=1, speaker="A", text="Good", language="en",
                segment_id=i, role="statement", speaker_role="Interviewee",
                sentiment=sentiment_pos, emotion=None,
            )
            for i in range(3)
        ]

        result = generate_report(segments, 10.0, "en")

        assert result.report.dominant_sentiment == "POS"

    def test_generate_report_extracts_participants(self, analyzed_segments):
        """Test participant extraction."""
        result = generate_report(
            segments=analyzed_segments,
            duration_seconds=120.0,
            language="en",
        )

        assert "Interviewer" in result.metadata.participants
        assert "Interviewee" in result.metadata.participants

    def test_generate_report_sets_metadata(self, analyzed_segments):
        """Test metadata population."""
        result = generate_report(
            segments=analyzed_segments,
            duration_seconds=1800.5,
            language="es",
        )

        assert result.metadata.duration_seconds == 1800.5
        assert result.metadata.language == "es"
        assert result.metadata.date == datetime.now().strftime("%Y-%m-%d")

    def test_generate_report_empty_segments(self):
        """Test report generation with empty segments."""
        result = generate_report(
            segments=[],
            duration_seconds=0.0,
            language="en",
        )

        assert result.report.total_segments == 0
        assert result.report.average_sentiment_score == 0.5
        assert result.report.dominant_sentiment == "N/A"

    def test_generate_report_no_sentiment_data(self):
        """Test report with segments that have no sentiment."""
        segments = [
            AnalyzedSegment(
                start=0, end=1, speaker="A", text="Hi", language="en",
                segment_id=0, role="statement", speaker_role="Interviewee",
                sentiment=None, emotion=None,
            )
        ]

        result = generate_report(segments, 10.0, "en")

        assert result.report.dominant_sentiment == "N/A"
        assert result.report.dominant_emotion == "N/A"


class TestSaveAnalysis:
    """Tests for saving analysis to file."""

    def test_save_analysis_creates_file(self, tmp_path):
        """Test that analysis is saved to file."""
        from src.models.interview import InterviewMetadata, InterviewReport

        meta = InterviewMetadata(
            date="2024-01-15", participants=["A"],
            duration_seconds=100.0, language="en"
        )
        report = InterviewReport(
            total_segments=0, total_questions=0, total_statements=0,
            sentiment_distribution={}, emotion_distribution={},
            average_sentiment_score=0.5,
            dominant_sentiment="N/A", dominant_emotion="N/A"
        )
        analysis = InterviewAnalysis(
            interview_id="save-test", metadata=meta, segments=[], report=report
        )

        output_path = tmp_path / "output" / "analysis.json"
        save_analysis(analysis, output_path)

        assert output_path.exists()

    def test_save_analysis_content_is_valid_json(self, tmp_path):
        """Test that saved content is valid JSON."""
        from src.models.interview import InterviewMetadata, InterviewReport

        meta = InterviewMetadata(
            date="2024-01-15", participants=["A"],
            duration_seconds=100.0, language="en"
        )
        report = InterviewReport(
            total_segments=0, total_questions=0, total_statements=0,
            sentiment_distribution={}, emotion_distribution={},
            average_sentiment_score=0.5,
            dominant_sentiment="N/A", dominant_emotion="N/A"
        )
        analysis = InterviewAnalysis(
            interview_id="json-test", metadata=meta, segments=[], report=report
        )

        output_path = tmp_path / "analysis.json"
        save_analysis(analysis, output_path)

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["interview_id"] == "json-test"
        assert loaded["metadata"]["language"] == "en"

    def test_save_analysis_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created."""
        from src.models.interview import InterviewMetadata, InterviewReport

        meta = InterviewMetadata(
            date="2024-01-15", participants=[],
            duration_seconds=0.0, language="en"
        )
        report = InterviewReport(
            total_segments=0, total_questions=0, total_statements=0,
            sentiment_distribution={}, emotion_distribution={},
            average_sentiment_score=0.0,
            dominant_sentiment="N/A", dominant_emotion="N/A"
        )
        analysis = InterviewAnalysis(
            interview_id="test", metadata=meta, segments=[], report=report
        )

        deep_path = tmp_path / "a" / "b" / "c" / "analysis.json"
        save_analysis(analysis, deep_path)

        assert deep_path.exists()
