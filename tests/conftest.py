"""Es lo que utiliza el mock para los tests. Son objetos mock ( objetos de mentiritas) para testear
funcionalidades sin tener que depender de los modelos reales o archivos grandes.
También incluye datos de ejemplo para usar en los tests."""



from unittest.mock import MagicMock

import pytest

from src.models.segment import Segment, TranscribedSegment
from src.models.analysis import SentimentResult, EmotionResult, AnalyzedSegment


# --- Sample Data Fixtures ---

@pytest.fixture
def sample_segment():
    return Segment(start=0.0, end=1.5, speaker="SPEAKER_00")


@pytest.fixture
def sample_segments():
    return [
        Segment(start=0.0, end=2.0, speaker="SPEAKER_00"),
        Segment(start=2.1, end=4.5, speaker="SPEAKER_01"),
        Segment(start=4.6, end=6.0, speaker="SPEAKER_00"),
    ]


@pytest.fixture
def sample_transcribed_segment():
    return TranscribedSegment(
        start=0.0,
        end=2.0,
        speaker="SPEAKER_00",
        text="Hello, how are you today?",
        language="en",
    )


@pytest.fixture
def sample_transcribed_segments():
    return [
        TranscribedSegment(
            start=0.0, end=2.0, speaker="SPEAKER_00",
            text="How are you feeling today?", language="en"
        ),
        TranscribedSegment(
            start=2.1, end=4.5, speaker="SPEAKER_01",
            text="I'm feeling great, thank you.", language="en"
        ),
        TranscribedSegment(
            start=4.6, end=6.0, speaker="SPEAKER_00",
            text="That's wonderful to hear.", language="en"
        ),
    ]


@pytest.fixture
def sample_sentiment_result():
    return SentimentResult(
        label="POS",
        score=0.85,
        probabilities={"POS": 0.85, "NEG": 0.1, "NEU": 0.05},
    )


@pytest.fixture
def sample_emotion_result():
    return EmotionResult(
        label="joy",
        score=0.7,
        probabilities={"joy": 0.7, "sadness": 0.1, "anger": 0.05, "others": 0.15},
    )


@pytest.fixture
def sample_analyzed_segment(sample_sentiment_result, sample_emotion_result):
    return AnalyzedSegment(
        start=0.0,
        end=2.0,
        speaker="SPEAKER_00",
        text="I'm feeling great today!",
        language="en",
        segment_id=0,
        role="statement",
        speaker_role="Interviewee",
        sentiment=sample_sentiment_result,
        emotion=sample_emotion_result,
        paired_with=None,
    )


# --- Mock Fixtures for ML Models ---

@pytest.fixture
def mock_whisper(mocker):
    """Mock whisper model and its transcribe method."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "This is transcribed text.",
        "language": "en",
    }
    mock_load = mocker.patch("src.audio.transcriber.whisper.load_model")
    mock_load.return_value = mock_model
    return mock_model


@pytest.fixture
def mock_pyannote_pipeline(mocker):
    """Mock pyannote speaker diarization pipeline."""
    mock_annotation = MagicMock()
    mock_annotation.__iter__ = lambda self: iter([
        (MagicMock(start=0.0, end=2.0), "SPEAKER_00"),
        (MagicMock(start=2.1, end=4.5), "SPEAKER_01"),
        (MagicMock(start=4.6, end=6.0), "SPEAKER_00"),
    ])

    mock_output = MagicMock()
    mock_output.speaker_diarization = mock_annotation

    mock_pipeline = MagicMock()
    mock_pipeline.return_value = mock_output

    mock_from_pretrained = mocker.patch("src.audio.diarizer.Pipeline.from_pretrained")
    mock_from_pretrained.return_value = mock_pipeline
    return mock_pipeline


@pytest.fixture
def mock_pysentimiento(mocker):
    """Mock pysentimiento analyzer."""
    mock_result = MagicMock()
    mock_result.output = "POS"
    mock_result.probas = {"POS": 0.8, "NEG": 0.1, "NEU": 0.1}

    mock_analyzer = MagicMock()
    mock_analyzer.predict.return_value = mock_result

    mock_create = mocker.patch("src.analysis.sentiment.create_analyzer")
    mock_create.return_value = mock_analyzer
    return mock_analyzer


@pytest.fixture
def mock_soundfile(mocker):
    """Mock soundfile.read for audio loading."""
    import numpy as np
    mock_read = mocker.patch("src.audio.diarizer.sf.read")
    mock_read.return_value = (np.zeros(16000), 16000)
    return mock_read


@pytest.fixture
def mock_pydub(mocker):
    """Mock pydub AudioSegment for audio processing."""
    mock_segment = MagicMock()
    mock_segment.__len__ = lambda self: 2000  # 2 seconds in ms
    mock_from_wav = mocker.patch("src.audio.transcriber.AudioSegment.from_wav")
    mock_from_wav.return_value = mock_segment
    return mock_from_wav


# --- File System Fixtures ---

@pytest.fixture
def temp_audio_file(tmp_path):
    """Create a temporary WAV file path."""
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 40)  # Minimal WAV header
    return audio_file


@pytest.fixture
def temp_project_structure(tmp_path):
    """Create a temporary project directory structure."""
    (tmp_path / "data" / "input").mkdir(parents=True)
    (tmp_path / "data" / "interim").mkdir(parents=True)
    (tmp_path / "data" / "output").mkdir(parents=True)
    return tmp_path


# --- Settings Fixtures ---

@pytest.fixture
def mock_settings(mocker):
    """Mock settings with test defaults."""
    from src.config.settings import Settings, AudioSettings, AnalysisSettings

    settings = Settings(
        audio=AudioSettings(whisper_model="tiny", min_segment_duration=0.1),
        analysis=AnalysisSettings(default_language="en"),
    )
    mocker.patch("src.config.settings.get_settings", return_value=settings)
    mocker.patch("src.audio.transcriber.get_settings", return_value=settings)
    return settings
