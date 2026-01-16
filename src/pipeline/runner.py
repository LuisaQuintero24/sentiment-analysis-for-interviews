""" Pipeline runner module for orchestrating the audio processing workflow.
    This module defines the main function `run_pipeline` which executes a series of
    steps to process interview audio data, including conversion, diarization,
    segmentation, transcription, analysis and report generation.
    
    Args:
        raw_dir (Path): Directory containing raw audio files.
        audio_wav (Path): Path to save the converted WAV audio file.
        parts_dir (Path): Directory to save segmented audio clips.
        output_path (Path): Path to save the final analysis report.
        device (str): Device to use for processing (e.g., "cpu" or "cuda").
        interview_id (str): Unique identifier for the interview.

    Returns:
           The final analysis report object or None if any step fails.

    Raises:
        None. Each step handles its own errors and returns None if it fails.
    
    Note:
        - The pipeline consists of multiple phases, each responsible for a specific task.
        - Progress is tracked and reported for each phase.
        - The final analysis report is saved to the specified output path.
        - The function returns None if any critical step fails.
        - The pipeline is designed to be modular and extensible for future enhancements.
"""


from pathlib import Path

from src.audio.converter import ensure_wav_audio
from src.audio.diarizer import diarize_audio
from src.audio.segmenter import split_audio_segments
from src.audio.transcriber import transcribe_segments
from src.analysis.sentiment import analyze_text
from src.analysis.question_classifier import classify_question
from src.analysis.speaker_mapper import map_speakers
from src.analysis.qa_pairer import pair_questions_answers
from src.config.settings import get_settings
from src.models.analysis import AnalyzedSegment, SentimentResult, EmotionResult
from src.models.interview import InterviewAnalysis
from src.output.report_generator import generate_report, save_analysis
from src.utils.progress import pipeline_progress

def run_pipeline(
    raw_dir: Path,
    audio_wav: Path,
    parts_dir: Path,
    output_path: Path,
    device: str = "cpu",
    interview_id: str = "interview_001",
) -> InterviewAnalysis | None:
    settings = get_settings()

    with pipeline_progress() as progress:
        progress.start_phase("Audio Conversion")
        if not ensure_wav_audio(raw_dir, audio_wav):
            return None
        progress.complete_phase("Audio Conversion")

        # Step 2: Diarize
        progress.start_phase("Speaker Diarization")
        segments = diarize_audio(audio_wav, device=device)
        if not segments:
            return None
        progress.complete_phase("Speaker Diarization")

        # Step 3: Split audio
        progress.start_phase("Audio Segmentation", total=len(segments))
        clip_paths = split_audio_segments(audio_wav, segments, parts_dir)
        progress.complete_phase("Audio Segmentation")

        # Step 4: Transcribe
        progress.start_phase("Transcription", total=len(segments))
        transcribed, detected_lang = transcribe_segments(
            segments,
            clip_paths,
            language=settings.analysis.default_language,)
        progress.complete_phase("Transcription")

        lang = detected_lang if detected_lang in ["es", "en", "it", "pt"] else "en"

        # Step 5: Classify questions
        progress.start_phase("Question Classification", total=len(transcribed))
        # Step 6: Sentiment analysis
        progress.start_phase("Sentiment Analysis", total=len(transcribed))
        # Step 7: Speaker mapping
        progress.start_phase("Speaker Mapping")
        speaker_map = map_speakers(transcribed)
        progress.complete_phase("Speaker Mapping")

        analyzed_segments: list[AnalyzedSegment] = []
        for idx, seg in enumerate(transcribed):
            if not seg.text.strip():
                role = "statement"
            else:
                role, _ = classify_question(seg.text)
            progress.advance("Question Classification")

            speaker_role = speaker_map.get(seg.speaker, seg.speaker)

            if not seg.text.strip():
                sentiment = None
                emotion = None
            elif role == "statement":
                sentiment, emotion = analyze_text(seg.text, lang)
            else:
                sentiment = SentimentResult(
                    label="NEU",
                    score=0.95,
                    probabilities={"NEG": 0.025, "NEU": 0.95, "POS": 0.025},
                )
                emotion = EmotionResult(
                    label="others",
                    score=0.95,
                    probabilities={
                        "others": 0.95,
                        "joy": 0.008,
                        "sadness": 0.008,
                        "anger": 0.008,
                        "surprise": 0.008,
                        "disgust": 0.008,
                        "fear": 0.008,},)
            progress.advance("Sentiment Analysis")

            analyzed_segments.append(
                AnalyzedSegment(
                    segment_id=idx + 1,
                    start=seg.start,
                    end=seg.end,
                    speaker=seg.speaker,
                    text=seg.text,
                    language=seg.language,
                    role=role,
                    speaker_role=speaker_role,
                    sentiment=sentiment,
                    emotion=emotion,
                    paired_with=None,))

        progress.complete_phase("Question Classification")
        progress.complete_phase("Sentiment Analysis")

        # Step 8: Pair Q&A
        progress.start_phase("Q&A Pairing")
        analyzed_segments = pair_questions_answers(analyzed_segments)
        progress.complete_phase("Q&A Pairing")

        # Step 9: Generate report
        progress.start_phase("Report Generation")
        duration = segments[-1].end - segments[0].start if segments else 0

        analysis = generate_report(
            segments=analyzed_segments,
            duration_seconds=duration,
            language=detected_lang,
            interview_id=interview_id,
        )
        save_analysis(analysis, output_path)
        progress.complete_phase("Report Generation")

    return analysis
