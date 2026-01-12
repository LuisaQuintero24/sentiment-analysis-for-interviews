"""Tests for pipeline runner including integration test."""

from unittest.mock import MagicMock

import pytest

from src.models.segment import Segment, TranscribedSegment
from src.models.analysis import SentimentResult, EmotionResult
from src.models.interview import InterviewAnalysis


class TestRunPipelineIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.fixture
    def mock_all_dependencies(self, tmp_path, mocker):
        """Mock all external dependencies for integration testing."""
        mock_segments = [
            Segment(start=0.0, end=2.0, speaker="SPEAKER_00"),
            Segment(start=2.1, end=4.0, speaker="SPEAKER_01"),
            Segment(start=4.1, end=6.0, speaker="SPEAKER_00"),
        ]

        mock_transcribed = [
            TranscribedSegment(
                start=0.0, end=2.0, speaker="SPEAKER_00",
                text="How are you feeling?", language="en"
            ),
            TranscribedSegment(
                start=2.1, end=4.0, speaker="SPEAKER_01",
                text="I feel great today!", language="en"
            ),
            TranscribedSegment(
                start=4.1, end=6.0, speaker="SPEAKER_00",
                text="That's wonderful.", language="en"
            ),
        ]

        mock_sentiment = SentimentResult(
            label="POS", score=0.9, probabilities={"POS": 0.9}
        )
        mock_emotion = EmotionResult(
            label="joy", score=0.8, probabilities={"joy": 0.8}
        )

        mocker.patch(
            "src.pipeline.runner.ensure_wav_audio",
            return_value=True
        )
        mocker.patch(
            "src.pipeline.runner.diarize_audio",
            return_value=mock_segments
        )
        mocker.patch(
            "src.pipeline.runner.split_audio_segments",
            return_value=[tmp_path / f"part_{i}.wav" for i in range(3)]
        )
        mocker.patch(
            "src.pipeline.runner.transcribe_segments",
            return_value=(mock_transcribed, "en")
        )
        mocker.patch(
            "src.pipeline.runner.classify_question",
            side_effect=[("question", 0.9), ("statement", 0.8), ("statement", 0.7)]
        )
        mocker.patch(
            "src.pipeline.runner.analyze_text",
            return_value=(mock_sentiment, mock_emotion)
        )

        mocker.patch(
            "src.pipeline.runner.get_settings",
            return_value=MagicMock(
                analysis=MagicMock(default_language="auto")
            )
        )

        mocker.patch("src.pipeline.runner.pipeline_progress")

        return {
            "segments": mock_segments,
            "transcribed": mock_transcribed,
        }

    def test_run_pipeline_returns_interview_analysis(
        self, tmp_path, mock_all_dependencies
    ):
        """Test that pipeline returns InterviewAnalysis on success."""
        from src.pipeline.runner import run_pipeline

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        audio_wav = tmp_path / "refined" / "audio.wav"
        parts_dir = tmp_path / "parts"
        output_path = tmp_path / "output" / "analysis.json"

        result = run_pipeline(
            raw_dir=raw_dir,
            audio_wav=audio_wav,
            parts_dir=parts_dir,
            output_path=output_path,
            device="cpu",
            interview_id="test-integration-001",
        )

        assert isinstance(result, InterviewAnalysis)
        assert result.interview_id == "test-integration-001"

    def test_run_pipeline_produces_correct_segment_count(
        self, tmp_path, mock_all_dependencies
    ):
        """Test that pipeline processes all segments."""
        from src.pipeline.runner import run_pipeline

        result = run_pipeline(
            raw_dir=tmp_path / "raw",
            audio_wav=tmp_path / "refined" / "a.wav",
            parts_dir=tmp_path / "parts",
            output_path=tmp_path / "out.json",
        )

        assert len(result.segments) == 3
        assert result.report.total_segments == 3

    def test_run_pipeline_classifies_questions_and_statements(
        self, tmp_path, mock_all_dependencies
    ):
        """Test question/statement classification."""
        from src.pipeline.runner import run_pipeline

        result = run_pipeline(
            raw_dir=tmp_path / "raw",
            audio_wav=tmp_path / "refined" / "a.wav",
            parts_dir=tmp_path / "parts",
            output_path=tmp_path / "out.json",
        )

        assert result.report.total_questions == 1
        assert result.report.total_statements == 2

    def test_run_pipeline_maps_speaker_roles(
        self, tmp_path, mock_all_dependencies
    ):
        """Test that speakers are mapped to roles."""
        from src.pipeline.runner import run_pipeline

        result = run_pipeline(
            raw_dir=tmp_path / "raw",
            audio_wav=tmp_path / "refined" / "a.wav",
            parts_dir=tmp_path / "parts",
            output_path=tmp_path / "out.json",
        )

        roles = {seg.speaker_role for seg in result.segments}
        assert "Interviewer" in roles or "Interviewee" in roles

    def test_run_pipeline_saves_output_file(
        self, tmp_path, mock_all_dependencies
    ):
        """Test that output file is created."""
        from src.pipeline.runner import run_pipeline

        output_path = tmp_path / "output" / "analysis.json"

        run_pipeline(
            raw_dir=tmp_path / "raw",
            audio_wav=tmp_path / "refined" / "a.wav",
            parts_dir=tmp_path / "parts",
            output_path=output_path,
        )

        assert output_path.exists()

    def test_run_pipeline_returns_none_on_audio_failure(self, tmp_path, mocker):
        """Test that pipeline returns None when audio conversion fails."""
        mocker.patch(
            "src.pipeline.runner.ensure_wav_audio",
            return_value=False
        )
        mocker.patch("src.pipeline.runner.pipeline_progress")
        mocker.patch("src.pipeline.runner.get_settings")

        from src.pipeline.runner import run_pipeline

        result = run_pipeline(
            raw_dir=tmp_path / "raw",
            audio_wav=tmp_path / "refined" / "a.wav",
            parts_dir=tmp_path / "parts",
            output_path=tmp_path / "out.json",
        )

        assert result is None

    def test_run_pipeline_returns_none_on_empty_diarization(self, tmp_path, mocker):
        """Test that pipeline returns None when diarization finds no speakers."""
        mocker.patch(
            "src.pipeline.runner.ensure_wav_audio",
            return_value=True
        )
        mocker.patch(
            "src.pipeline.runner.diarize_audio",
            return_value=[]
        )
        mocker.patch("src.pipeline.runner.pipeline_progress")
        mocker.patch("src.pipeline.runner.get_settings")

        from src.pipeline.runner import run_pipeline

        result = run_pipeline(
            raw_dir=tmp_path / "raw",
            audio_wav=tmp_path / "refined" / "a.wav",
            parts_dir=tmp_path / "parts",
            output_path=tmp_path / "out.json",
        )

        assert result is None


class TestPipelineImports:
    """Basic import tests for pipeline module."""

    def test_run_pipeline_is_callable(self):
        """Test that run_pipeline can be imported."""
        from src.pipeline.runner import run_pipeline

        assert callable(run_pipeline)

    def test_import_pipeline_module(self):
        """Test package-level import."""
        from src.pipeline import run_pipeline

        assert callable(run_pipeline)
