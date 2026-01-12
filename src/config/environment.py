"""Environment configuration for the audio processing pipeline."""

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
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is required. "
            "Get your token at https://huggingface.co/settings/tokens"
        )
    print(" HF_TOKEN configurado")


def setup_ffmpeg(script_dir: Path):
    """
    Add FFmpeg to PATH for pydub and whisper.

    Parameters
    ----------
    script_dir : Path
        Root directory of the project
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
    """
    Configure all necessary environment variables.

    Parameters
    ----------
    script_dir : Path
        Root directory of the project
    """
    setup_hf_token()
    setup_ffmpeg(script_dir)
