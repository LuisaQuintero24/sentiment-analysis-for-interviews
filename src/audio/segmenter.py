import logging
from pathlib import Path

from pydub import AudioSegment

from src.models.segment import Segment

logger = logging.getLogger(__name__)


def split_audio_segments(audio_path: Path,segments: list[Segment],output_dir: Path,) -> list[Path]:
    """Split audio file into clips based on provided segments.

    Args: 
        audio_path (Path): Path to the input audio file.
        segments (list[Segment]): List of segments with start and end times.
        output_dir (Path): Directory to save the output clips.
    
    Returns:
        list[Path]: List of paths to the generated audio clips.
        
    Raises:
        None. Assumes valid input paths and segments.
        
    Note:
        - Uses pydub to handle audio splitting
        - Exports clips in WAV format
        -Creates output directory if it does not exist
        - Logs key steps for transparency
        
    Example:
        >>> audio_path = Path("/path/to/audio.wav")
        >>> segments = [Segment(start=0.0, end=5.0), Segment(start=10.0, end= 15.0
        >>> output_fir = Path("/path/to/output_clips"))]
        >>> clip_paths = split_audio_segments(audio_path, segments, output_dir)
        >>> for path in clip_paths:
        >>>     print(path)
        """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading audio: {audio_path}")
    audio = AudioSegment.from_wav(str(audio_path))

    clip_paths = []
    for idx, seg in enumerate(segments):
        start_ms = int(seg.start * 1000)
        end_ms = int(seg.end * 1000)
        clip = audio[start_ms:end_ms]

        out_path = output_dir / f"part_{idx}.wav"
        clip.export(str(out_path), format="wav")
        clip_paths.append(out_path)

        duration = (end_ms - start_ms) / 1000
        logger.debug(f"Exported: {out_path.name} ({duration:.3f}s)")

    logger.info(f"Split complete: {len(clip_paths)} clips")
    return clip_paths
