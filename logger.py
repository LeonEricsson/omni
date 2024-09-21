from typing import Dict

from rich.console import Console
from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn

import wandb


class TrainingLogger:
    """Handles logging for both training and validation."""

    def __init__(self, num_epochs: int, train_batches: int):
        self.num_epochs = num_epochs
        self.train_batches = train_batches
        self.console = Console()

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        self.epoch_task = None
        self.val_task = None

    def __enter__(self):
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()

    def start_training(self, device):
        self.console.print(f"[blue]Using device: {device}")
        self.console.print("[green]Starting training...")

    def end_training(self):
        self.console.print("[green]Training complete")

    def start_epoch(self, epoch: int, step: int) -> None:
        self.console.rule(f"[bold cyan]Epoch {epoch + 1}/{self.num_epochs}")
        wandb.log({"Epoch": epoch + 1}, step=step)

        self.epoch_task = self.progress.add_task(
            f"[green]Epoch {epoch + 1}/{self.num_epochs}", total=self.train_batches
        )

    def end_epoch(self, epoch: int, total_tokens: int) -> None:
        self.console.rule(f"[bold green]Epoch {epoch + 1} Complete")
        self.console.print(f"Total tokens processed: {total_tokens:,}")

    def log_training_step(self, metrics: Dict[str, float], step: int) -> None:
        wandb.log(metrics, step=step)

    def advance_train(self) -> None:
        self.progress.advance(self.epoch_task)

    def start_validation(self, num_batches: int) -> None:
        self.val_task = self.progress.add_task(
            "[yellow]Validating...", total=num_batches
        )

    def advance_validation(self) -> None:
        self.progress.advance(self.val_task)

    def log_validation_step(self, metrics: Dict[str, float], step: int) -> None:
        wandb.log(metrics, step=step)
        self.progress.remove_task(self.val_task)
