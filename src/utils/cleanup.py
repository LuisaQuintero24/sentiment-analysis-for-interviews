"""
Utilidades para limpieza de archivos y carpetas del pipeline.
"""

import shutil

from .progress import console


def cleanup_folders(script_dir):
    """
    Limpia carpetas y archivos de ejecuciones previas.

    Parameters
    ----------
    script_dir : Path
        Directorio raíz del proyecto donde se ejecuta el script
    """
    folders_to_clean = [
        script_dir / "data" / "interim",
        script_dir / "data" / "refined",
        script_dir / "data" / "output" / "parts"
    ]

    files_to_clean = [
        script_dir / "data" / "output" / "audio_diarizado.json",
        script_dir / "data" / "output" / "audio_diarizado_transcribed.json",
        script_dir / "data" / "output" / "transcriptions.json",
        script_dir / "data" / "output" / "sentiment_analysis.json"
    ]

    with console.status("[bold blue]Cleaning previous run files..."):
        for folder in folders_to_clean:
            if folder.exists():
                for item in folder.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            else:
                folder.mkdir(parents=True, exist_ok=True)

        for file in files_to_clean:
            if file.exists():
                file.unlink()

    console.print("[green]✓[/green] Cleanup complete")
