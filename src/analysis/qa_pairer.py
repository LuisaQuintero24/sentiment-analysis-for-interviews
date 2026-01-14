
"""Pairs interview questions with their corresponding answers.
    
    This function iterates through a list of analyzed segments and links questions
    (identified by "Interviewer" role) with the answers that follow them. It updates
    the `paired_with` attribute of both questions and answers to establish the
    relationship between them.

    Args:
         A list of AnalyzedSegment objects containing interview data with
        speaker roles and segment classifications.

    Returns:
        The same list of segments with updated `paired_with` attributes that link
        questions to their first answer and answers to their corresponding question.

    Raises:
        None. Returns the original segments list if it's empty.

    Note:
        - Questions are identified by speaker_role="Interviewer" and role="question"
        - Answers are matched by finding consecutive non-Interviewer segments
        - Empty answer text is skipped during pairing
        - Questions are linked to their first non-empty answer
        - Multiple answers can be paired to a single question
    """

import logging

from src.models.analysis import AnalyzedSegment

logger = logging.getLogger(__name__)


def pair_questions_answers(segments: list[AnalyzedSegment]) -> list[AnalyzedSegment]:

    if not segments:
        return segments

    i = 0
    while i < len(segments):
        seg = segments[i]
        if seg.speaker_role == "Interviewer" and seg.role == "question":
            question_id = seg.segment_id
            first_answer_id = None

            j = i + 1
            while j < len(segments) and segments[j].speaker_role != "Interviewer":
                answer = segments[j]
                if answer.text.strip():
                    answer.paired_with = question_id
                    logger.debug(
                        f"Paired answer {answer.segment_id} -> question {question_id}"
                    )
                    if first_answer_id is None:
                        first_answer_id = answer.segment_id
                j += 1

            if first_answer_id:
                seg.paired_with = first_answer_id
                logger.debug(
                    f"Paired question {question_id} -> first answer {first_answer_id}"
                )

            i = j
        else:
            i += 1

    return segments
