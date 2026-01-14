import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def setup_hf_token():
    """
    Configure HuggingFace token for pyannote.

    Uses HF_TOKEN from environment variables.
    Get your token at: https://huggingface.co/settings/tokens
    
    Raises:
        ValueError: if HF_TOKEN is not set in environment variables.
        
    Note:
        - Essential for accessing HuggingFace models
        - Loads token from environment variable HF_TOKEN
        - Provides user guidance if token is missing
    
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is required."
            "Get your token at https://huggingface.co/settings/tokens"
        )
    print(" HF_TOKEN configurado")


def setup_ffmpeg(script_dir: Path):
    """ Metod to set up FFmpeg path for audio processing.
    Args:
        script_dir (Path): Base directory of the script.
    
    Returns: 
        None. Sets FFmpeg path in environment variables.
    
    Raises: 
        None. Warns if FFmpeg path is not found.
    
    Note:
        - FFmpeg is required for audio format conversions
        - Assumes FFmpeg is located in 'engines/ffmpeg-<version>/bin' relative to script_dir
        - Updates system PATH to include FFmpeg binaries
        - Provides user feedback on setup status
        """    
        
    ffmpeg_path = (
        script_dir / "engines" / "ffmpeg-2026-01-05-git-2892815c45-full_build" / "bin"
    )
    if ffmpeg_path.exists():
        os.environ["PATH"] = f"{os.environ['PATH']};{ffmpeg_path}"
        print(f"FFmpeg anadido al PATH: {ffmpeg_path}")
    else:
        print(f" Advertencia: No se encontro FFmpeg en {ffmpeg_path}")


def setup_environment(script_dir: Path):
    setup_hf_token()
    setup_ffmpeg(script_dir)
