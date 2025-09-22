"""
Modern Progress Display System

Provides beautiful, informative progress indicators with contextual information,
tips, and next steps for all CLI operations.
"""

import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from contextlib import contextmanager

try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn, 
        TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
    )
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.live import Live
    from rich.layout import Layout
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .framework import CLIContext


@dataclass
class ProgressStep:
    """Represents a step in a multi-step process."""
    
    name: str
    description: str
    emoji: str = "⚙️"
    estimated_duration: int = 30  # seconds
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress_percentage: float = 0.0
    tips: List[str] = None
    
    def __post_init__(self):
        if self.tips is None:
            self.tips = []


@dataclass
class ProcessTip:
    """Represents a contextual tip to show during long operations."""
    
    content: str
    timing: str  # when to show: "start", "middle", "end", "always"
    category: str = "general"  # general, educational, technical, cultural


class ModernProgress:
    """Modern progress display with rich formatting and contextual information."""
    
    def __init__(self, context: CLIContext):
        self.context = context
        self.console = Console() if RICH_AVAILABLE else None
        self._current_operation: Optional[str] = None
        self._start_time: Optional[datetime] = None
        self._tips_database = self._initialize_tips_database()
    
    def show_operation_progress(
        self, 
        operation_name: str,
        steps: List[ProgressStep],
        show_tips: bool = True
    ) -> 'ProgressContext':
        """Show progress for a multi-step operation."""
        
        return ProgressContext(
            progress_display=self,
            operation_name=operation_name,
            steps=steps,
            show_tips=show_tips
        )
    
    def show_simple_progress(
        self,
        description: str,
        total: Optional[int] = None,
        unit: str = "items"
    ) -> 'SimpleProgressContext':
        """Show simple progress bar for basic operations."""
        
        return SimpleProgressContext(
            progress_display=self,
            description=description,
            total=total,
            unit=unit
        )
    
    def show_training_progress(
        self,
        epoch: int,
        total_epochs: int,
        batch: int,
        total_batches: int,
        loss: float,
        learning_rate: float,
        eta: str
    ):
        """Show specialized training progress with metrics."""
        
        if not RICH_AVAILABLE:
            print(f"Training: Epoch {epoch}/{total_epochs}, Batch {batch}/{total_batches}, Loss: {loss:.4f}")
            return
        
        # Create training progress layout
        layout = Layout()
        layout.split_column(
            Layout(self._create_training_header(epoch, total_epochs), name="header", size=3),
            Layout(self._create_training_metrics(loss, learning_rate, eta), name="metrics", size=5),
            Layout(self._create_training_progress_bar(batch, total_batches), name="progress", size=3)
        )
        
        return layout
    
    def _create_training_header(self, epoch: int, total_epochs: int) -> Panel:
        """Create training header panel."""
        
        header_text = Text()
        header_text.append("🎓 ", style="bold yellow")
        header_text.append("Training Educational Content Model", style="bold blue")
        header_text.append(f" - Epoch {epoch}/{total_epochs}", style="cyan")
        
        return Panel(header_text, border_style="blue")
    
    def _create_training_metrics(self, loss: float, lr: float, eta: str) -> Table:
        """Create training metrics table."""
        
        metrics_table = Table(show_header=False, box=None, padding=(0, 2))
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("Value", style="cyan")
        metrics_table.add_column("Status", style="green")
        
        # Loss with interpretation
        loss_status = "Excellent" if loss < 0.1 else "Good" if loss < 0.5 else "Training"
        metrics_table.add_row("📉 Loss", f"{loss:.4f}", loss_status)
        
        # Learning rate
        metrics_table.add_row("🎯 Learning Rate", f"{lr:.2e}", "")
        
        # Time remaining
        metrics_table.add_row("⏱️  Time Remaining", eta, "")
        
        return metrics_table
    
    def _create_training_progress_bar(self, batch: int, total_batches: int) -> Progress:
        """Create training progress bar."""
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Processing batch..."),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console
        )
        
        task = progress.add_task("Training", total=total_batches)
        progress.update(task, completed=batch)
        
        return progress
    
    def show_status_summary(self, components: Dict[str, Any]):
        """Show system status in a beautiful format."""
        
        if not RICH_AVAILABLE:
            for component, status in components.items():
                print(f"{component}: {status}")
            return
        
        # Create status table
        status_table = Table(title="🔍 System Status", show_header=True, header_style="bold magenta")
        status_table.add_column("Component", style="bold")
        status_table.add_column("Status", justify="center")
        status_table.add_column("Details", style="dim")
        
        for component, info in components.items():
            if info.get("ok", False):
                status_icon = "✅"
                status_style = "green"
            else:
                status_icon = "⚠️"
                status_style = "yellow"
            
            status_table.add_row(
                component,
                f"[{status_style}]{status_icon}[/{status_style}]",
                info.get("message", "")
            )
        
        self.console.print(status_table)
    
    def show_next_steps(self, steps: List[Dict[str, str]]):
        """Show suggested next steps in an attractive format."""
        
        if not RICH_AVAILABLE:
            print("\nSuggested next steps:")
            for i, step in enumerate(steps, 1):
                print(f"{i}. {step['title']}: {step['command']}")
            return
        
        steps_text = Text()
        steps_text.append("🎯 Recommended next actions:\n\n", style="bold cyan")
        
        for i, step in enumerate(steps, 1):
            steps_text.append(f"{i}. ", style="bold white")
            steps_text.append(f"{step.get('emoji', '▶️')} {step['title']}", style="cyan")
            steps_text.append(f"\n   {step['description']}\n", style="dim")
            steps_text.append(f"   💻 Command: ", style="dim")
            steps_text.append(f"toaripi {step['command']}", style="bold green")
            steps_text.append("\n\n")
        
        self.console.print(Panel(steps_text, title="🧭 What's Next?", border_style="cyan"))
    
    def show_celebration(self, message: str, details: List[str] = None):
        """Show celebration message for completed operations."""
        
        if details is None:
            details = []
        
        if not RICH_AVAILABLE:
            print(f"🎉 {message}")
            for detail in details:
                print(f"  • {detail}")
            return
        
        celebration_text = Text()
        celebration_text.append("🎉 ", style="bold yellow")
        celebration_text.append(message, style="bold green")
        celebration_text.append("\n\n")
        
        for detail in details:
            celebration_text.append("✨ ", style="yellow")
            celebration_text.append(f"{detail}\n", style="green")
        
        self.console.print(Panel(
            celebration_text,
            title="🌟 Success!",
            border_style="green",
            padding=(1, 2)
        ))
    
    def _initialize_tips_database(self) -> Dict[str, List[ProcessTip]]:
        """Initialize database of contextual tips."""
        
        return {
            "training": [
                ProcessTip(
                    "💡 Training creates an AI that learns Toaripi language patterns from your parallel text data.",
                    "start",
                    "educational"
                ),
                ProcessTip(
                    "🎯 The model is learning to generate age-appropriate educational content for primary school students.",
                    "middle",
                    "educational"
                ),
                ProcessTip(
                    "🌍 All generated content will be automatically validated for cultural sensitivity and appropriateness.",
                    "middle",
                    "cultural"
                ),
                ProcessTip(
                    "⏱️ Training time varies based on data size and computer speed. Larger datasets generally produce better models.",
                    "start",
                    "technical"
                )
            ],
            "data_preparation": [
                ProcessTip(
                    "📚 Good training data should have clear, accurate translations between English and Toaripi.",
                    "start",
                    "educational"
                ),
                ProcessTip(
                    "🔍 We automatically check for cultural appropriateness and age-suitable content.",
                    "middle",
                    "cultural"
                ),
                ProcessTip(
                    "✅ Quality is more important than quantity - clean, accurate data works better than large, messy datasets.",
                    "start",
                    "general"
                )
            ],
            "model_export": [
                ProcessTip(
                    "📦 Exporting creates a compressed model that can run offline on devices like Raspberry Pi.",
                    "start",
                    "technical"
                ),
                ProcessTip(
                    "🏫 Exported models are perfect for classroom use where internet connectivity may be limited.",
                    "middle",
                    "educational"
                )
            ]
        }


class ProgressContext:
    """Context manager for multi-step progress operations."""
    
    def __init__(
        self,
        progress_display: ModernProgress,
        operation_name: str,
        steps: List[ProgressStep],
        show_tips: bool = True
    ):
        self.progress_display = progress_display
        self.operation_name = operation_name
        self.steps = steps
        self.show_tips = show_tips
        self._current_step_index = 0
        self._progress = None
        self._live = None
        
    def __enter__(self):
        self._start_operation()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_operation(exc_type is None)
    
    def _start_operation(self):
        """Start the progress operation."""
        
        if not RICH_AVAILABLE:
            print(f"Starting {self.operation_name}...")
            return
        
        # Create progress display
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.progress_display.console
        )
        
        # Add tasks for each step
        for step in self.steps:
            self._progress.add_task(
                f"{step.emoji} {step.description}",
                total=100
            )
        
        self._live = Live(self._progress, console=self.progress_display.console)
        self._live.start()
    
    def advance_step(self, step_name: str, progress: float = 100.0, message: str = ""):
        """Advance to the next step."""
        
        if not RICH_AVAILABLE:
            print(f"✅ Completed: {step_name}")
            return
        
        # Update current step
        if self._current_step_index < len(self.steps):
            task_id = list(self._progress.tasks.keys())[self._current_step_index]
            self._progress.update(task_id, completed=progress)
            
            # Show tip if available
            if self.show_tips:
                self._show_contextual_tip()
            
            self._current_step_index += 1
    
    def _show_contextual_tip(self):
        """Show a contextual tip for the current operation."""
        
        # This would show tips based on operation type and timing
        pass
    
    def _end_operation(self, success: bool):
        """End the progress operation."""
        
        if self._live:
            self._live.stop()
        
        if success and RICH_AVAILABLE:
            self.progress_display.show_celebration(
                f"{self.operation_name} completed successfully!",
                [f"✅ {step.description}" for step in self.steps]
            )


class SimpleProgressContext:
    """Context manager for simple progress operations."""
    
    def __init__(
        self,
        progress_display: ModernProgress,
        description: str,
        total: Optional[int] = None,
        unit: str = "items"
    ):
        self.progress_display = progress_display
        self.description = description
        self.total = total
        self.unit = unit
        self._progress = None
        self._task_id = None
    
    def __enter__(self):
        if not RICH_AVAILABLE:
            print(f"Starting {self.description}...")
            return self
        
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn() if self.total else SpinnerColumn(),
            TextColumn(f"{{task.completed}} {self.unit}"),
            TimeElapsedColumn(),
            console=self.progress_display.console
        )
        
        self._task_id = self._progress.add_task(self.description, total=self.total)
        self._progress.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._progress:
            self._progress.stop()
    
    def update(self, advance: int = 1, description: str = None):
        """Update progress."""
        
        if not RICH_AVAILABLE:
            return
        
        if self._progress and self._task_id:
            self._progress.update(self._task_id, advance=advance)
            if description:
                self._progress.update(self._task_id, description=description)


class ProgressManager:
    """Manages all progress-related functionality."""
    
    def __init__(self, context: CLIContext):
        self.context = context
        self.modern_progress = ModernProgress(context)
    
    def create_operation_progress(self, operation: str, steps: List[ProgressStep]) -> ProgressContext:
        """Create progress context for multi-step operations."""
        return self.modern_progress.show_operation_progress(operation, steps)
    
    def create_simple_progress(self, description: str, total: int = None) -> SimpleProgressContext:
        """Create simple progress context."""
        return self.modern_progress.show_simple_progress(description, total)
    
    def show_status(self, components: Dict[str, Any]):
        """Show system status."""
        self.modern_progress.show_status_summary(components)
    
    def show_celebration(self, message: str, details: List[str] = None):
        """Show celebration."""
        self.modern_progress.show_celebration(message, details)
    
    def show_next_steps(self, steps: List[Dict[str, str]]):
        """Show next steps.""" 
        self.modern_progress.show_next_steps(steps)