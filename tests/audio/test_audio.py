"""Behavioral tests for audio processing modules."""

from unittest.mock import MagicMock

import pytest

from src.models.segment import Segment


class TestConverter:
    """Tests for audio converter module."""

    def test_find_audio_file_finds_mp3(self, tmp_path):
        """Test finding MP3 file in directory."""
        mp3_path = tmp_path / "interview.mp3"
        mp3_path.touch()

        from src.audio.converter import find_audio_file

        result = find_audio_file(tmp_path)

        assert result == mp3_path

    def test_find_audio_file_finds_m4a(self, tmp_path):
        """Test finding M4A file in directory."""
        m4a_path = tmp_path / "interview.m4a"
        m4a_path.touch()

        from src.audio.converter import find_audio_file

        result = find_audio_file(tmp_path)

        assert result == m4a_path

    def test_find_audio_file_returns_none_when_empty(self, tmp_path):
        """Test returning None when no audio file found."""
        from src.audio.converter import find_audio_file

        result = find_audio_file(tmp_path)

        assert result is None

    def test_convert_to_wav_success(self, tmp_path, mocker):
        """Test successful audio to WAV conversion."""
        source_path = tmp_path / "input.mp3"
        wav_path = tmp_path / "output" / "audio.wav"
        source_path.touch()

        mock_audio = MagicMock()
        mock_from_file = mocker.patch(
            "src.audio.converter.AudioSegment.from_file",
            return_value=mock_audio,
        )

        from src.audio.converter import convert_to_wav

        result = convert_to_wav(source_path, wav_path)

        assert result is True
        mock_from_file.assert_called_once_with(str(source_path), format="mp3")
        mock_audio.export.assert_called_once_with(str(wav_path), format="wav")

    def test_convert_to_wav_failure(self, tmp_path, mocker):
        """Test conversion failure returns False."""
        source_path = tmp_path / "input.mp3"
        wav_path = tmp_path / "output.wav"

        mocker.patch(
            "src.audio.converter.AudioSegment.from_file",
            side_effect=Exception("Conversion failed"),
        )

        from src.audio.converter import convert_to_wav

        result = convert_to_wav(source_path, wav_path)

        assert result is False

    def test_convert_to_wav_unsupported_format(self, tmp_path):
        """Test conversion fails for unsupported format."""
        source_path = tmp_path / "input.xyz"
        wav_path = tmp_path / "output.wav"
        source_path.touch()

        from src.audio.converter import convert_to_wav

        result = convert_to_wav(source_path, wav_path)

        assert result is False

    def test_ensure_wav_audio_wav_exists(self, tmp_path):
        """Test that existing WAV file is detected."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        wav_path = tmp_path / "refined" / "audio.wav"
        wav_path.parent.mkdir(parents=True)
        wav_path.touch()

        from src.audio.converter import ensure_wav_audio

        result = ensure_wav_audio(raw_dir, wav_path)

        assert result is True

    def test_ensure_wav_audio_converts_from_source(self, tmp_path, mocker):
        """Test conversion when WAV doesn't exist but source does."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        mp3_path = raw_dir / "audio.mp3"
        mp3_path.touch()
        wav_path = tmp_path / "refined" / "audio.wav"

        mock_audio = MagicMock()
        mocker.patch(
            "src.audio.converter.AudioSegment.from_file",
            return_value=mock_audio,
        )

        from src.audio.converter import ensure_wav_audio

        result = ensure_wav_audio(raw_dir, wav_path)

        assert result is True

    def test_ensure_wav_audio_no_files(self, tmp_path):
        """Test failure when no source files exist."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        wav_path = tmp_path / "refined" / "audio.wav"

        from src.audio.converter import ensure_wav_audio

        result = ensure_wav_audio(raw_dir, wav_path)

        assert result is False


class TestSegmenter:
    """Tests for audio segmenter module."""

    def test_split_audio_segments_creates_clips(self, tmp_path, mocker):
        """Test that segments are split into individual clips."""
        audio_path = tmp_path / "audio.wav"
        output_dir = tmp_path / "clips"
        audio_path.touch()

        mock_audio = MagicMock()
        mock_audio.__getitem__ = MagicMock(return_value=MagicMock())
        mocker.patch(
            "src.audio.segmenter.AudioSegment.from_wav",
            return_value=mock_audio,
        )

        segments = [
            Segment(start=0.0, end=1.5, speaker="SPEAKER_00"),
            Segment(start=2.0, end=3.5, speaker="SPEAKER_01"),
        ]

        from src.audio.segmenter import split_audio_segments

        result = split_audio_segments(audio_path, segments, output_dir)

        assert len(result) == 2
        assert result[0] == output_dir / "part_0.wav"
        assert result[1] == output_dir / "part_1.wav"
        assert output_dir.exists()

    def test_split_audio_segments_correct_timestamps(self, tmp_path, mocker):
        """Test that correct time slices are extracted."""
        audio_path = tmp_path / "audio.wav"
        output_dir = tmp_path / "clips"
        audio_path.touch()

        mock_clip = MagicMock()
        mock_audio = MagicMock()
        mock_audio.__getitem__ = MagicMock(return_value=mock_clip)
        mocker.patch(
            "src.audio.segmenter.AudioSegment.from_wav",
            return_value=mock_audio,
        )

        segments = [Segment(start=1.5, end=3.0, speaker="SPEAKER_00")]

        from src.audio.segmenter import split_audio_segments

        split_audio_segments(audio_path, segments, output_dir)

        mock_audio.__getitem__.assert_called_once_with(slice(1500, 3000))

    def test_split_audio_segments_empty_list(self, tmp_path, mocker):
        """Test handling of empty segment list."""
        audio_path = tmp_path / "audio.wav"
        output_dir = tmp_path / "clips"
        audio_path.touch()

        mock_audio = MagicMock()
        mocker.patch(
            "src.audio.segmenter.AudioSegment.from_wav",
            return_value=mock_audio,
        )

        from src.audio.segmenter import split_audio_segments

        result = split_audio_segments(audio_path, [], output_dir)

        assert result == []


class TestDiarizer:
    """Tests for speaker diarization module."""

    def test_diarize_audio_returns_segments(
        self, tmp_path, mock_pyannote_pipeline, mock_soundfile, mocker
    ):
        """Test that diarization returns properly formatted segments."""
        mocker.patch.dict("os.environ", {"HF_TOKEN": "test_token"})
        audio_path = tmp_path / "audio.wav"
        audio_path.touch()

        from src.audio.diarizer import diarize_audio

        result = diarize_audio(audio_path)

        assert len(result) == 3
        assert all(isinstance(seg, Segment) for seg in result)

    def test_diarize_audio_raises_without_token(self, tmp_path, mocker):
        """Test that missing HF_TOKEN raises ValueError."""
        mocker.patch.dict("os.environ", {}, clear=True)
        if "HF_TOKEN" in mocker.patch.dict("os.environ", {}).keys():
            del mocker.patch.dict("os.environ", {})["HF_TOKEN"]

        audio_path = tmp_path / "audio.wav"

        from src.audio.diarizer import diarize_audio

        with pytest.raises(ValueError, match="HF_TOKEN not found"):
            diarize_audio(audio_path, hf_token=None)

    def test_write_rttm_creates_file(self, tmp_path, sample_segments):
        """Test that RTTM file is created with correct format."""
        rttm_path = tmp_path / "output" / "diarization.rttm"

        from src.audio.diarizer import write_rttm

        write_rttm(sample_segments, rttm_path, uri="test_audio")

        assert rttm_path.exists()
        content = rttm_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3
        assert "SPEAKER test_audio" in lines[0]
        assert "SPEAKER_00" in lines[0]


class TestTranscriber:
    """Tests for audio transcription module."""

    def test_transcribe_segments_returns_transcribed_segments(
        self, tmp_path, mock_whisper, mock_pydub, mock_settings
    ):
        """Test that transcription returns TranscribedSegment objects."""
        segments = [Segment(start=0.0, end=2.0, speaker="SPEAKER_00")]
        clip_paths = [tmp_path / "clip_0.wav"]
        clip_paths[0].touch()

        from src.audio.transcriber import transcribe_segments
        from src.models.segment import TranscribedSegment

        result, lang = transcribe_segments(segments, clip_paths)

        assert len(result) == 1
        assert isinstance(result[0], TranscribedSegment)
        assert result[0].text == "This is transcribed text."
        assert lang == "en"

    def test_transcribe_segments_skips_short_clips(
        self, tmp_path, mock_whisper, mock_settings, mocker
    ):
        """Test that clips shorter than min_duration are skipped."""
        mock_short_clip = MagicMock()
        mock_short_clip.__len__ = MagicMock(return_value=50)  # 50ms
        mocker.patch(
            "src.audio.transcriber.AudioSegment.from_wav",
            return_value=mock_short_clip,
        )

        segments = [Segment(start=0.0, end=0.05, speaker="SPEAKER_00")]
        clip_paths = [tmp_path / "clip_0.wav"]
        clip_paths[0].touch()

        from src.audio.transcriber import transcribe_segments

        result, _ = transcribe_segments(segments, clip_paths)

        assert result[0].text == ""
        mock_whisper.transcribe.assert_not_called()

    def test_detect_language_returns_detected(self, mocker):
        """Test language detection with valid text."""
        mocker.patch("src.audio.transcriber.detect", return_value="es")

        from src.audio.transcriber import detect_language

        result = detect_language("Hola, ¿cómo estás?")

        assert result == "es"

    def test_detect_language_fallback_on_error(self, mocker):
        """Test language detection falls back to 'en' on error."""
        from langdetect import LangDetectException

        mocker.patch(
            "src.audio.transcriber.detect",
            side_effect=LangDetectException(code=0, message="Error"),
        )

        from src.audio.transcriber import detect_language

        result = detect_language("")

        assert result == "en"
