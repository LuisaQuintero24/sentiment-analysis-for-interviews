import logging
from functools import lru_cache

from pysentimiento import create_analyzer

from src.models.analysis import SentimentResult, EmotionResult

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _get_analyzer(task: str, lang: str):
    logger.info(f"Loading {task} analyzer for language: {lang}")
    return create_analyzer(task=task, lang=lang)


def analyze_sentiment(text: str, lang: str = "es") -> SentimentResult:
    analyzer = _get_analyzer("sentiment", lang)
    result = analyzer.predict(text)

    return SentimentResult(
        label=str(result.output),
        score=result.probas.get(str(result.output), 0.0),
        probabilities={str(k): float(v) for k, v in result.probas.items()},
    )


def analyze_emotion(text: str, lang: str = "es") -> EmotionResult:
    analyzer = _get_analyzer("emotion", lang)
    result = analyzer.predict(text)

    return EmotionResult(
        label=str(result.output),
        score=result.probas.get(str(result.output), 0.0),
        probabilities={str(k): float(v) for k, v in result.probas.items()},
    )


def analyze_text(text: str, lang: str = "es") -> tuple[SentimentResult, EmotionResult]:
    return analyze_sentiment(text, lang), analyze_emotion(text, lang)
