"""
Main script to run the interview analysis pipeline.
Sets up the environment, cleans previous runs, and executes the pipeline,
logging key information and results.
Handles exceptions and ensures proper logging throughout the process.

Args:
    None. The script is executed directly.

Returns:
    Exit code 0 on success, 1 on failure.

Raises:
    None. All exceptions are caught and logged.

Note:
    - The script uses rich for console output and logging for detailed logs.
    - Cleans up previous run files before starting a new pipeline execution.
    - Configures environment settings such as HF token and FFmpeg path.
    - Logs key statistics from the pipeline upon successful completion.
    - Handles warnings related to FP16 in CPU environments.

"""

import sys
import warnings
from pathlib import Path

from rich.panel import Panel

from src.config.settings import setup_logging, get_settings
from src.config.paths import get_project_paths
from src.config.environment import setup_environment
from src.pipeline.runner import run_pipeline
from src.utils import cleanup_folders, console

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")


def main() -> bool:
    script_dir = Path(__file__).parent.resolve()

    # Print title first
    console.print(Panel.fit(
        "[bold cyan]Interview Analysis Pipeline[/bold cyan]",
        border_style="cyan"
    ))

    # Setup logging (will appear below title)
    logger = setup_logging()

    settings = get_settings()
    logger.info(f"Working directory: {script_dir}")
    logger.info(f"Whisper model: {settings.audio.whisper_model}")
    logger.info(f"Question classifier: {settings.analysis.question_model}")

    # Clean previous runs
    cleanup_folders(script_dir)

    # Setup environment (HF token, FFmpeg path)
    setup_environment(script_dir)

    # Get paths
    paths = get_project_paths(script_dir)

    # Run pipeline
    try:
        result = run_pipeline(
            raw_dir=paths.raw_dir,
            audio_wav=paths.audio_wav,
            parts_dir=paths.parts_dir,
            output_path=paths.sentiment_json,
            device="cpu",
        )

        if result:
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Output: {paths.sentiment_json}")
            logger.info(f"Total segments: {result.report.total_segments}")
            logger.info(f"Questions: {result.report.total_questions}")
            logger.info(f"Statements: {result.report.total_statements}")
            logger.info(f"Dominant sentiment: {result.report.dominant_sentiment}")
            logger.info(f"Dominant emotion: {result.report.dominant_emotion}")
            return True

        logger.error("Pipeline failed")
        return False

    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
