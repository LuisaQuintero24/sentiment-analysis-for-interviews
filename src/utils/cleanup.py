"""
    Clean up specific folders and files used in previous pipeline runs.


    Args:
        script_dir (Path): The base directory of the script where data folders are located.

    Returns:
        None. The specified folders and files are deleted if they exist.

    Raises:
        None. Any errors during deletion are silently ignored.

    Note:
                                                                                                                                                            - Folders cleaned: interim, refined, output/parts
        - Files cleaned: audio_diarizado.json, audio_diarizado_transcribed.json, transcriptions.json, sentiment_analysis.json"""



import shutil
from .progress import console


def cleanup_folders(script_dir):       
    folders_to_clean = [
        script_dir / "data" / "interim",
        script_dir / "data" / "refined",
        script_dir / "data" / "output" / "parts"]

    files_to_clean = [
        script_dir / "data" / "output" / "audio_diarizado.json",
        script_dir / "data" / "output" / "audio_diarizado_transcribed.json",
        script_dir / "data" / "output" / "transcriptions.json",
        script_dir / "data" / "output" / "sentiment_analysis.json"]

    with console.status("Cleaning previous run files..."):
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

    console.print("[green]Cleanup complete[/green]")
