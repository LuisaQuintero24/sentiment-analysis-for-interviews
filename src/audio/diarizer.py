"""Speaker diarization using pyannote audio pipeline.
    Perform speaker diarization on the given audio file using pyannote.

    Args:
        audio_path (Path): Path to the input audio file.
        hf_token (str | None): Hugging Face token for authentication. If None, reads from HF_TOKEN env variable.
        device (str): Device to run the model on ( "cpu" or "cuda)

    Returns: 
        list[Segment]: list of diarized segments with start, end times and speaker labels.

    Raises:
        ValueError: if HF token is not provided and not found in environment

    Note:
        - Uses pyannote's speaker-diarization-3.1 pipeline
        - Loads audio using soundfile and converts to torch tensor
        -Output segments are rounded to 3 decimal places
        - Logs key steps for transparency

    Example:
        >>> audio_path = Path(" /path/to/audio.wav")
        >>> segments = diarize_audio(audio_path, hf_token="your_hf_token", device= "cuda"
        >>> for seg in segments:
        >>>   print(seg)
"""

import logging
import os
from pathlib import Path

import torch
import soundfile as sf
from pyannote.audio import Pipeline

from src.models.segment import Segment

logger = logging.getLogger(__name__)


def diarize_audio(audio_path: Path,hf_token: str | None = None,device: str = "cpu",) -> list[Segment]:
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found. Set it via environment or pass hf_token.")

    logger.info(f"Loading audio: {audio_path}")
    data, sample_rate = sf.read(str(audio_path))
    waveform = torch.from_numpy(data).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T

    logger.info("Loading pyannote speaker-diarization-3.1 pipeline")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,
    )
    pipeline.to(torch.device(device))

    logger.info(f"Running diarization on device: {device}")
    output = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    annotation = output.speaker_diarization

    segments = []
    for turn, speaker in annotation:
        segments.append(
            Segment(
                start=round(turn.start, 3),
                end=round(turn.end, 3),
                speaker=str(speaker),
            )
        )

    logger.info(f"Diarization complete: {len(segments)} segments found")
    return segments


def write_rttm(segments: list[Segment], rttm_path: Path, uri: str = "audio") -> None:
    """Write segments to RTTM format for compatibility."""
    rttm_path.parent.mkdir(parents=True, exist_ok=True)

    with open(rttm_path, "w", encoding="utf-8") as f:
        for seg in segments:
            duration = seg.end - seg.start
            f.write(
                f"SPEAKER {uri} 1 {seg.start:.3f} {duration:.3f} "
                f"<NA> <NA> {seg.speaker} <NA> <NA>\n"
            )

    logger.info(f"RTTM saved to: {rttm_path}")
