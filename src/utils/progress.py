"""This module provides a progress tracking utility for monitoring the 
various phases of the interview processing pipeline. It defines a 
`PipelineProgress` class that uses the `rich` library to display progress bars
and status updates in the console.

Args:
    None. The module defines classes and functions for use in other parts of the application.
    
Returns:
    None. The module provides functionality for tracking and displaying progress.

Raises:
    None. The module is designed to be used without raising exceptions.
    
Note:
    - The `PipelineProgress` class can be used as a context manager to automatically
    handle the start and stop of the progress display.
    
    - each phase of the pipeline can be started, advanced and completed with dedicated methods.
    
    - The `pipeline_progress` function provides a convenient way to use the `PipelineProgress` class
     within a `with` statement.
    
    """

from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)

console = Console()

PIPELINE_PHASES = [
    ("Audio Conversion", None),
    ("Speaker Diarization", None),
    ("Audio Segmentation", None),
    ("Transcription", None),
    ("Question Classification", None),
    ("Sentiment Analysis", None),
    ("Speaker Mapping", None),
    ("Q&A Pairing", None),
    ("Report Generation", None),
]


class PipelineProgress:
    def __init__(self):
        self.console = console
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            expand=False,
        )
        self.tasks: dict[str, int] = {}
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False

    def start(self):
        if self._started:
            return
        self._started = True
        self.progress.start()
        for name, _ in PIPELINE_PHASES:
            task_id = self.progress.add_task(name, total=None, visible=False)
            self.tasks[name] = task_id

    def start_phase(self, name: str, total: int | None = None):
        if name not in self.tasks:
            return
        task_id = self.tasks[name]
        self.progress.update(task_id, visible=True, total=total)
        self.progress.start_task(task_id)

    def advance(self, name: str, advance: int = 1):
        if name not in self.tasks:
            return
        self.progress.advance(self.tasks[name], advance)

    def complete_phase(self, name: str):
        if name not in self.tasks:
            return
        task_id = self.tasks[name]
        task = self.progress.tasks[task_id]
        if task.total is None:
            self.progress.update(task_id, total=1, completed=1)
        else:
            self.progress.update(task_id, completed=task.total)

    def finish(self):
        if not self._started:
            return
        self.progress.stop()
        self.console.print("[bold green]Pipeline complete![/bold green]")


@contextmanager
def pipeline_progress() -> Generator[PipelineProgress, None, None]:
    progress = PipelineProgress()
    with progress:
        yield progress
