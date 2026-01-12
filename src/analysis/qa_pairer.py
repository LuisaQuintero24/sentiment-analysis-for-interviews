import logging

from src.models.analysis import AnalyzedSegment

logger = logging.getLogger(__name__)


def pair_questions_answers(segments: list[AnalyzedSegment]) -> list[AnalyzedSegment]:
    """
    Pair Interviewer questions with Interviewee answers.
    Each Interviewer question pairs with all subsequent Interviewee segments
    until the next Interviewer segment.
    """
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
