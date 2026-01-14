""" Moduke for classifying text segments as questions or statements.
   Provides functions to classify text using a zero-shot classification model.

   Args:
       text: the input text segment to classify

    Returns:
        tuple: a tuple containing the classified role("question", "statement")
        and the confidence score.

    Raises:
        None. Assumes valid input text.

    Note:
        - Uses transformers pipeline for zero-shot classification
        - Caches classifier instances for performance
        - Classification threshold is configurable via settings
        - Default candidate labels are "question" and "statement"

    Example:
        >>> role, score = classify_question("¿Cómo te llamas?"
        >>> print(role, score)"""


import logging
from functools import lru_cache

from transformers import pipeline

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


CANDIDATE_LABELS = ["question", "statement"]


@lru_cache(maxsize=1)
def _get_classifier():
    settings = get_settings()
    model_name = settings.analysis.question_model
    logger.info(f"Loading question classifier: {model_name}")
    return pipeline("zero-shot-classification", model=model_name)


def classify_question(text: str) -> tuple[str, float]:
    classifier = _get_classifier()
    settings = get_settings()

    result = classifier(text, candidate_labels=CANDIDATE_LABELS)

    top_label = result["labels"][0]
    top_score = result["scores"][0]

    if top_label == "question" and top_score >= settings.thresholds.question_confidence:
        return "question", top_score

    return "statement", result["scores"][1] if top_label == "question" else top_score


def is_question(text: str) -> bool:
    role, _ = classify_question(text)
    return role == "question"
