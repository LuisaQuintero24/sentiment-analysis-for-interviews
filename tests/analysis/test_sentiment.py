"""Behavioral tests for analysis modules."""

from unittest.mock import MagicMock

from src.models.segment import TranscribedSegment
from src.models.analysis import SentimentResult, EmotionResult, AnalyzedSegment


class TestSentimentAnalysis:
    """Tests for sentiment analysis functions."""

    def test_analyze_sentiment_returns_result(self, mocker):
        """Test that analyze_sentiment returns a SentimentResult."""
        mock_result = MagicMock()
        mock_result.output = "POS"
        mock_result.probas = {"POS": 0.8, "NEG": 0.1, "NEU": 0.1}

        mock_analyzer = MagicMock()
        mock_analyzer.predict.return_value = mock_result

        mocker.patch(
            "src.analysis.sentiment.create_analyzer",
            return_value=mock_analyzer,
        )
        mocker.patch("src.analysis.sentiment._get_analyzer.cache_clear")

        from src.analysis.sentiment import _get_analyzer
        _get_analyzer.cache_clear()

        from src.analysis.sentiment import analyze_sentiment

        result = analyze_sentiment("I love this!", lang="en")

        assert isinstance(result, SentimentResult)
        assert result.label == "POS"
        assert result.score == 0.8
        assert "POS" in result.probabilities

    def test_analyze_emotion_returns_result(self, mocker):
        """Test that analyze_emotion returns an EmotionResult."""
        mock_result = MagicMock()
        mock_result.output = "joy"
        mock_result.probas = {"joy": 0.7, "sadness": 0.1, "anger": 0.2}

        mock_analyzer = MagicMock()
        mock_analyzer.predict.return_value = mock_result

        mocker.patch(
            "src.analysis.sentiment.create_analyzer",
            return_value=mock_analyzer,
        )

        from src.analysis.sentiment import _get_analyzer
        _get_analyzer.cache_clear()

        from src.analysis.sentiment import analyze_emotion

        result = analyze_emotion("I'm so happy!", lang="en")

        assert isinstance(result, EmotionResult)
        assert result.label == "joy"
        assert result.score == 0.7

    def test_analyze_text_returns_both(self, mocker):
        """Test that analyze_text returns both sentiment and emotion."""
        mock_result = MagicMock()
        mock_result.output = "POS"
        mock_result.probas = {"POS": 0.8, "NEG": 0.1, "NEU": 0.1}

        mock_analyzer = MagicMock()
        mock_analyzer.predict.return_value = mock_result

        mocker.patch(
            "src.analysis.sentiment.create_analyzer",
            return_value=mock_analyzer,
        )

        from src.analysis.sentiment import _get_analyzer
        _get_analyzer.cache_clear()

        from src.analysis.sentiment import analyze_text

        sentiment, emotion = analyze_text("Great day!", lang="en")

        assert isinstance(sentiment, SentimentResult)
        assert isinstance(emotion, EmotionResult)


class TestQuestionClassifier:
    """Tests for question classification."""

    def test_classify_question_identifies_question(self, mocker):
        """Test that questions are correctly classified."""
        mock_classifier = MagicMock()
        mock_classifier.return_value = {
            "labels": ["question", "statement"],
            "scores": [0.9, 0.1],
        }

        mocker.patch(
            "src.analysis.question_classifier.pipeline",
            return_value=mock_classifier,
        )
        mocker.patch(
            "src.analysis.question_classifier.get_settings",
            return_value=MagicMock(
                analysis=MagicMock(question_model="test-model"),
                thresholds=MagicMock(question_confidence=0.5),
            ),
        )

        from src.analysis.question_classifier import _get_classifier
        _get_classifier.cache_clear()

        from src.analysis.question_classifier import classify_question

        role, score = classify_question("How are you?")

        assert role == "question"
        assert score == 0.9

    def test_classify_question_identifies_statement(self, mocker):
        """Test that statements are correctly classified."""
        mock_classifier = MagicMock()
        mock_classifier.return_value = {
            "labels": ["statement", "question"],
            "scores": [0.85, 0.15],
        }

        mocker.patch(
            "src.analysis.question_classifier.pipeline",
            return_value=mock_classifier,
        )
        mocker.patch(
            "src.analysis.question_classifier.get_settings",
            return_value=MagicMock(
                analysis=MagicMock(question_model="test-model"),
                thresholds=MagicMock(question_confidence=0.5),
            ),
        )

        from src.analysis.question_classifier import _get_classifier
        _get_classifier.cache_clear()

        from src.analysis.question_classifier import classify_question

        role, score = classify_question("I am fine.")

        assert role == "statement"
        assert score == 0.85

    def test_classify_question_below_threshold_is_statement(self, mocker):
        """Test that low-confidence questions are classified as statements."""
        mock_classifier = MagicMock()
        mock_classifier.return_value = {
            "labels": ["question", "statement"],
            "scores": [0.4, 0.6],  # Below 0.5 threshold
        }

        mocker.patch(
            "src.analysis.question_classifier.pipeline",
            return_value=mock_classifier,
        )
        mocker.patch(
            "src.analysis.question_classifier.get_settings",
            return_value=MagicMock(
                analysis=MagicMock(question_model="test-model"),
                thresholds=MagicMock(question_confidence=0.5),
            ),
        )

        from src.analysis.question_classifier import _get_classifier
        _get_classifier.cache_clear()

        from src.analysis.question_classifier import classify_question

        role, score = classify_question("Maybe this is a question")

        assert role == "statement"

    def test_is_question_returns_boolean(self, mocker):
        """Test is_question helper function."""
        mock_classifier = MagicMock()
        mock_classifier.return_value = {
            "labels": ["question", "statement"],
            "scores": [0.9, 0.1],
        }

        mocker.patch(
            "src.analysis.question_classifier.pipeline",
            return_value=mock_classifier,
        )
        mocker.patch(
            "src.analysis.question_classifier.get_settings",
            return_value=MagicMock(
                analysis=MagicMock(question_model="test-model"),
                thresholds=MagicMock(question_confidence=0.5),
            ),
        )

        from src.analysis.question_classifier import _get_classifier
        _get_classifier.cache_clear()

        from src.analysis.question_classifier import is_question

        assert is_question("What is your name?") is True


class TestSpeakerMapper:
    """Tests for speaker role mapping."""

    def test_map_speakers_assigns_interviewer_to_less_frequent(self):
        """Test that the less frequent speaker becomes Interviewer."""
        segments = [
            TranscribedSegment(
                start=0, end=1, speaker="A", text="Hi", language="en"
            ),
            TranscribedSegment(
                start=1, end=2, speaker="B", text="Hello", language="en"
            ),
            TranscribedSegment(
                start=2, end=3, speaker="B", text="How are you?", language="en"
            ),
            TranscribedSegment(
                start=3, end=4, speaker="B", text="Great", language="en"
            ),
        ]

        from src.analysis.speaker_mapper import map_speakers

        result = map_speakers(segments)

        assert result["A"] == "Interviewer"  # 1 segment
        assert result["B"] == "Interviewee"  # 3 segments

    def test_map_speakers_empty_list(self):
        """Test that empty input returns empty dict."""
        from src.analysis.speaker_mapper import map_speakers

        result = map_speakers([])

        assert result == {}

    def test_map_speakers_single_speaker(self):
        """Test handling of single speaker."""
        segments = [
            TranscribedSegment(
                start=0, end=1, speaker="A", text="Hi", language="en"
            ),
            TranscribedSegment(
                start=1, end=2, speaker="A", text="Hello", language="en"
            ),
        ]

        from src.analysis.speaker_mapper import map_speakers

        result = map_speakers(segments)

        assert result["A"] == "Interviewer"

    def test_get_speaker_role_returns_mapped_role(self):
        """Test get_speaker_role with valid mapping."""
        from src.analysis.speaker_mapper import get_speaker_role

        speaker_map = {"SPEAKER_00": "Interviewer", "SPEAKER_01": "Interviewee"}

        assert get_speaker_role("SPEAKER_00", speaker_map) == "Interviewer"
        assert get_speaker_role("SPEAKER_01", speaker_map) == "Interviewee"

    def test_get_speaker_role_returns_original_if_not_mapped(self):
        """Test get_speaker_role falls back to original speaker code."""
        from src.analysis.speaker_mapper import get_speaker_role

        result = get_speaker_role("UNKNOWN", {})

        assert result == "UNKNOWN"


class TestQAPairer:
    """Tests for question-answer pairing."""

    def test_pair_questions_answers_pairs_question_to_next_answer(self):
        """Questions are paired with the next answer from different speaker."""
        segments = [
            AnalyzedSegment(
                start=0, end=1, speaker="A", text="How are you?", language="en",
                segment_id=0, role="question", speaker_role="Interviewer",
            ),
            AnalyzedSegment(
                start=1, end=2, speaker="B", text="I'm fine.", language="en",
                segment_id=1, role="statement", speaker_role="Interviewee",
            ),
        ]

        from src.analysis.qa_pairer import pair_questions_answers

        result = pair_questions_answers(segments)

        assert result[0].paired_with == 1
        assert result[1].paired_with == 0

    def test_pair_questions_answers_only_last_question_pairs(self):
        """Test that only the last Interviewer question before answers pairs."""
        segments = [
            AnalyzedSegment(
                start=0, end=1, speaker="A", text="Question one?", language="en",
                segment_id=0, role="question", speaker_role="Interviewer",
            ),
            AnalyzedSegment(
                start=1, end=2, speaker="A", text="Question two?", language="en",
                segment_id=1, role="question", speaker_role="Interviewer",
            ),
            AnalyzedSegment(
                start=2, end=3, speaker="B", text="Answer here.", language="en",
                segment_id=2, role="statement", speaker_role="Interviewee",
            ),
        ]

        from src.analysis.qa_pairer import pair_questions_answers

        result = pair_questions_answers(segments)

        assert result[0].paired_with is None  # No Interviewee answers before next Interviewer
        assert result[1].paired_with == 2     # Last question pairs with first answer
        assert result[2].paired_with == 1     # Answer pairs back to question

    def test_pair_questions_answers_empty_list(self):
        """Test that empty list returns empty list."""
        from src.analysis.qa_pairer import pair_questions_answers

        result = pair_questions_answers([])

        assert result == []

    def test_pair_questions_answers_skips_empty_text(self):
        """Test that segments with empty text are not paired."""
        segments = [
            AnalyzedSegment(
                start=0, end=1, speaker="A", text="Question?", language="en",
                segment_id=0, role="question", speaker_role="Interviewer",
            ),
            AnalyzedSegment(
                start=1, end=2, speaker="B", text="", language="en",
                segment_id=1, role="statement", speaker_role="Interviewee",
            ),
            AnalyzedSegment(
                start=2, end=3, speaker="B", text="Real answer.", language="en",
                segment_id=2, role="statement", speaker_role="Interviewee",
            ),
        ]

        from src.analysis.qa_pairer import pair_questions_answers

        result = pair_questions_answers(segments)

        assert result[0].paired_with == 2  # Skipped empty segment
        assert result[1].paired_with is None  # Empty text not paired
        assert result[2].paired_with == 0  # Answer pairs to question

    def test_pair_questions_answers_multiple_answers_pair_to_same_question(self):
        """Test that all Interviewee segments pair to the same question."""
        segments = [
            AnalyzedSegment(
                start=0, end=1, speaker="A", text="Tell me about yourself.", language="en",
                segment_id=0, role="question", speaker_role="Interviewer",
            ),
            AnalyzedSegment(
                start=1, end=2, speaker="B", text="Well, I work in tech.", language="en",
                segment_id=1, role="statement", speaker_role="Interviewee",
            ),
            AnalyzedSegment(
                start=2, end=3, speaker="B", text="I love coding.", language="en",
                segment_id=2, role="statement", speaker_role="Interviewee",
            ),
            AnalyzedSegment(
                start=3, end=4, speaker="B", text="It's been great.", language="en",
                segment_id=3, role="statement", speaker_role="Interviewee",
            ),
        ]

        from src.analysis.qa_pairer import pair_questions_answers

        result = pair_questions_answers(segments)

        assert result[0].paired_with == 1  # Question pairs to first answer
        assert result[1].paired_with == 0  # First answer to question
        assert result[2].paired_with == 0  # Second answer to question
        assert result[3].paired_with == 0  # Third answer to question

    def test_pair_questions_answers_interviewee_question_not_paired(self):
        """Test that questions from Interviewee are not paired as questions."""
        segments = [
            AnalyzedSegment(
                start=0, end=1, speaker="A", text="How are you?", language="en",
                segment_id=0, role="question", speaker_role="Interviewer",
            ),
            AnalyzedSegment(
                start=1, end=2, speaker="B", text="Good, and you?", language="en",
                segment_id=1, role="question", speaker_role="Interviewee",
            ),
        ]

        from src.analysis.qa_pairer import pair_questions_answers

        result = pair_questions_answers(segments)

        assert result[0].paired_with == 1  # Question pairs to Interviewee segment
        assert result[1].paired_with == 0  # Interviewee pairs back to Interviewer question
