""" Dataclass encapsulating all key paths used in the project.

   Args:
       script_dir (Path): Base directory of the script.
       raw_dir (Path): Directory for raw input audio files.
       audio_wav (Path): Path to the converted WAV audio file.
       rttm_file (Path): Path to the RTTM file for diarization output.
       segments_json (Path): Path to the JSON file with diarization segments.
       parts_dir (Path): Directory to store split audio segments.
       transcriptions_json (Path): Path to the JSON file with transcriptions.
       final_json (Path): Path to the final output JSON file.
       sentiment_json (Path): Path to the sentiment analysis JSON file.
    
   Returns:
       ProjectPaths: dataclass instance with all configured paths.
       
    Raises:
        None. All paths are constructed based on the script_dir.
        
    Note:
        - Centralizes all path configurations for easy management
        - Uses pathlib for path manipulations
        - Paths are relative to the provided script_dir
        
    Example:
        >>> script_dir = Path("/path/to/project")
        >>> paths = get_project_paths(script_dir)
        >>> print(paths.audio_wav) 
        
        """


from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectPaths:
    script_dir: Path
    raw_dir: Path
    audio_wav: Path
    rttm_file: Path
    segments_json: Path
    parts_dir: Path
    transcriptions_json: Path
    final_json: Path
    sentiment_json: Path


def get_project_paths(script_dir: Path) -> ProjectPaths:
    return ProjectPaths(
        script_dir=script_dir,
        raw_dir=script_dir / "data" / "raw",
        audio_wav=script_dir / "data" / "refined" / "audio.wav",
        rttm_file=script_dir / "data" / "interim" / "audio.rttm",
        segments_json=script_dir / "data" / "output" / "audio_diarizado.json",
        parts_dir=script_dir / "data" / "output" / "parts",
        transcriptions_json=script_dir / "data" / "output" / "transcriptions.json",
        final_json=script_dir / "data" / "output" / "audio_diarizado_transcribed.json",
        sentiment_json=script_dir / "data" / "output" / "sentiment_analysis.json",
    )
