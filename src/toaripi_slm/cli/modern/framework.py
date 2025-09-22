"""
Modern CLI Framework - Core Infrastructure

Provides the base framework for a sleek, user-friendly CLI experience with
rich formatting, intelligent guidance, and adaptive interface.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.emoji import Emoji


@dataclass
class CLIContext:
    """Enhanced CLI context for managing global state with user experience focus."""
    
    # Core configuration
    config_file: Optional[Path] = None
    verbose: bool = False
    quiet: bool = False
    working_directory: Path = field(default_factory=Path.cwd)
    
    # User experience
    user_profile: Optional['UserProfile'] = None
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    interface_style: str = "modern"  # modern, classic, minimal
    
    # Educational focus
    educational_mode: bool = True
    cultural_validation: bool = True
    age_group_focus: str = "primary"
    
    # Runtime state
    current_workflow: Optional[str] = None
    last_command: Optional[str] = None
    
    # Modern CLI components (lazy loaded)
    modern_cli: Optional['ModernCLI'] = None
    guidance_engine: Optional['GuidanceEngine'] = None
    progress_manager: Optional['ProgressManager'] = None
    error_context: Dict[str, Any] = field(default_factory=dict)
    
    def log(self, message: str, level: str = "info", emoji: str = ""):
        """Enhanced logging with emoji and rich formatting."""
        if self.quiet and level not in ["error", "warning"]:
            return
        
        console = Console()
        
        # Add emoji if provided
        if emoji:
            message = f"{emoji} {message}"
        
        # Style based on level
        if level == "error":
            console.print(f"[bold red]❌ ERROR:[/bold red] {message}")
        elif level == "warning":
            console.print(f"[bold yellow]⚠️  WARNING:[/bold yellow] {message}")
        elif level == "success":
            console.print(f"[bold green]✅ SUCCESS:[/bold green] {message}")
        elif level == "info":
            console.print(f"[bold blue]ℹ️  INFO:[/bold blue] {message}")
        elif level == "debug" and self.verbose:
            console.print(f"[dim]🔍 DEBUG:[/dim] {message}")
        elif level == "tip":
            console.print(f"[bold cyan]💡 TIP:[/bold cyan] {message}")
        elif level == "celebration":
            console.print(f"[bold magenta]🎉 {message}[/bold magenta]")
        else:
            console.print(message)


class ModernCLI:
    """
    Modern CLI framework with rich user experience features.
    
    Provides:
    - Beautiful terminal interface with rich formatting
    - Intelligent user guidance and suggestions
    - Adaptive complexity based on user experience
    - Cultural sensitivity for educational content
    - Seamless error recovery and help system
    """
    
    def __init__(self, context: CLIContext):
        self.context = context
        self.console = Console()
        self._guidance_engine = None
        self._progress_manager = None
        self._error_handler = None
        
    @property
    def guidance_engine(self):
        """Lazy load guidance engine to avoid circular imports."""
        if self._guidance_engine is None:
            from .guidance_system import GuidanceEngine
            self._guidance_engine = GuidanceEngine(self.context)
        return self._guidance_engine
    
    @property
    def progress_manager(self):
        """Lazy load progress manager."""
        if self._progress_manager is None:
            from .progress_display import ProgressManager
            self._progress_manager = ProgressManager(self.context)
        return self._progress_manager
    
    @property
    def error_handler(self):
        """Lazy load error handler."""
        if self._error_handler is None:
            from .error_handling import SmartErrorRecovery
            self._error_handler = SmartErrorRecovery(self.context)
        return self._error_handler
    
    def show_welcome(self, first_time: bool = False) -> None:
        """Show modern welcome interface with system status."""
        
        # Create welcome content based on user profile
        if first_time:
            self._show_first_time_welcome()
        else:
            self._show_returning_user_welcome()
        
        # Show system status
        self._show_system_status()
        
        # Show contextual next steps
        self._show_next_steps()
    
    def _show_first_time_welcome(self) -> None:
        """Welcome screen for first-time users."""
        
        welcome_text = Text()
        welcome_text.append("🌟 ", style="bold yellow")
        welcome_text.append("Welcome to Toaripi SLM", style="bold blue")
        welcome_text.append(" 🌟\n\n", style="bold yellow")
        
        welcome_text.append("🎓 ", style="bold")
        welcome_text.append("Mission: ", style="bold")
        welcome_text.append("Generate educational content for Toaripi language learners\n")
        
        welcome_text.append("👥 ", style="bold")
        welcome_text.append("For: ", style="bold") 
        welcome_text.append("Teachers, students, and community members\n")
        
        welcome_text.append("🌍 ", style="bold")
        welcome_text.append("Focus: ", style="bold")
        welcome_text.append("Cultural preservation and age-appropriate learning\n\n")
        
        welcome_text.append("✨ ", style="bold cyan")
        welcome_text.append("Let's get you started with a quick setup!", style="cyan")
        
        panel = Panel(
            welcome_text,
            title="🎉 First Time Setup",
            border_style="blue",
            padding=(1, 2)
        )
        
        self.console.print()
        self.console.print(Align.center(panel))
        self.console.print()
    
    def _show_returning_user_welcome(self) -> None:
        """Welcome screen for returning users."""
        
        user_name = "there"  # Will be enhanced with user profile
        if self.context.user_profile:
            user_name = self.context.user_profile.display_name or "there"
        
        welcome_text = Text()
        welcome_text.append(f"👋 Welcome back, {user_name}!\n\n", style="bold blue")
        welcome_text.append("Ready to create educational content in Toaripi? ", style="cyan")
        
        panel = Panel(
            welcome_text,
            title="🌟 Toaripi SLM",
            border_style="green",
            padding=(1, 2)
        )
        
        self.console.print()
        self.console.print(panel)
    
    def _show_system_status(self) -> None:
        """Show current system status with visual indicators."""
        
        # Check system components
        status_items = self._check_system_components()
        
        # Create status table
        status_table = Table(show_header=False, box=None, padding=(0, 1))
        status_table.add_column("Component", style="bold")
        status_table.add_column("Status", justify="left")
        
        for component, status in status_items.items():
            if status["ok"]:
                status_table.add_row(
                    f"✅ {component}",
                    f"[green]{status['message']}[/green]"
                )
            else:
                status_table.add_row(
                    f"⚠️  {component}",
                    f"[yellow]{status['message']}[/yellow]"
                )
        
        status_panel = Panel(
            status_table,
            title="📊 System Status",
            border_style="cyan"
        )
        
        self.console.print(status_panel)
    
    def _check_system_components(self) -> Dict[str, Dict[str, Any]]:
        """Check status of key system components."""
        
        components = {}
        
        # Check Python environment
        try:
            import transformers, torch
            components["Python Environment"] = {
                "ok": True,
                "message": "All required packages installed"
            }
        except ImportError:
            components["Python Environment"] = {
                "ok": False,
                "message": "Missing required packages"
            }
        
        # Check data directory
        data_dir = self.context.working_directory / "data"
        if data_dir.exists():
            data_files = list(data_dir.rglob("*.csv"))
            if data_files:
                components["Training Data"] = {
                    "ok": True,
                    "message": f"Found {len(data_files)} data files"
                }
            else:
                components["Training Data"] = {
                    "ok": False,
                    "message": "No training data found"
                }
        else:
            components["Training Data"] = {
                "ok": False,
                "message": "Data directory not found"
            }
        
        # Check models directory
        models_dir = self.context.working_directory / "models"
        if models_dir.exists():
            model_files = list(models_dir.rglob("*.bin")) + list(models_dir.rglob("*.gguf"))
            if model_files:
                components["Trained Models"] = {
                    "ok": True,
                    "message": f"Found {len(model_files)} trained models"
                }
            else:
                components["Trained Models"] = {
                    "ok": False,
                    "message": "No trained models yet"
                }
        else:
            components["Trained Models"] = {
                "ok": False,
                "message": "Models directory not found"
            }
        
        return components
    
    def _show_next_steps(self) -> None:
        """Show contextual next steps based on current state."""
        
        next_steps = self.guidance_engine.suggest_next_actions()
        
        if not next_steps:
            return
        
        steps_text = Text()
        steps_text.append("🎯 Suggested next steps:\n\n", style="bold cyan")
        
        for i, step in enumerate(next_steps[:3], 1):  # Show top 3 suggestions
            steps_text.append(f"{i}. ", style="bold white")
            steps_text.append(f"{step['emoji']} {step['title']}", style="cyan")
            steps_text.append(f" - {step['description']}\n", style="dim")
            steps_text.append(f"   Command: ", style="dim")
            steps_text.append(f"toaripi {step['command']}", style="bold green")
            steps_text.append("\n\n")
        
        steps_panel = Panel(
            steps_text,
            title="🧭 What's Next?",
            border_style="cyan"
        )
        
        self.console.print(steps_panel)
    
    def handle_natural_language_input(self, user_input: str) -> Optional[str]:
        """
        Parse natural language input and suggest appropriate commands.
        
        Examples:
        - "train a model" -> "train start"
        - "check my data" -> "data validate"
        - "generate a story" -> "interact --content-type story"
        """
        
        # Simple keyword matching (can be enhanced with NLP)
        user_input = user_input.lower().strip()
        
        # Training related
        if any(word in user_input for word in ["train", "create model", "build model"]):
            return "train start"
        
        # Data related
        if any(word in user_input for word in ["check data", "validate data", "prepare data"]):
            return "data validate"
        
        # Generation related
        if any(word in user_input for word in ["generate", "create story", "make content"]):
            return "interact"
        
        # Status related
        if any(word in user_input for word in ["status", "check system", "health"]):
            return "status"
        
        # Help related
        if any(word in user_input for word in ["help", "how to", "what can"]):
            return "help"
        
        return None
    
    def show_command_help(self, command: str, context: str = "") -> None:
        """Show contextual help for a specific command."""
        
        # This will be enhanced with the guidance system
        help_text = f"Help for command: {command}"
        if context:
            help_text += f" (in context: {context})"
        
        self.console.print(Panel(help_text, title="💡 Help", border_style="yellow"))
    
    def confirm_action(self, message: str, default: bool = True) -> bool:
        """Show a user-friendly confirmation dialog."""
        
        emoji = "✅" if default else "❓"
        return Confirm.ask(f"{emoji} {message}", default=default)
    
    def get_user_choice(self, choices: List[str], prompt: str = "Choose an option") -> str:
        """Get user choice from a list of options."""
        
        # Display choices in a nice format
        choice_table = Table(show_header=False, box=None)
        choice_table.add_column("Option", style="bold cyan")
        choice_table.add_column("Description", style="white")
        
        for i, choice in enumerate(choices, 1):
            choice_table.add_row(f"{i}.", choice)
        
        self.console.print(choice_table)
        
        while True:
            try:
                selection = Prompt.ask(f"🎯 {prompt} (1-{len(choices)})")
                if selection.isdigit():
                    index = int(selection) - 1
                    if 0 <= index < len(choices):
                        return choices[index]
                self.console.print("[red]Please enter a valid number.[/red]")
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Operation cancelled.[/yellow]")
                sys.exit(0)


def with_modern_cli(func: Callable) -> Callable:
    """Decorator to enhance click commands with modern CLI features."""
    
    def wrapper(*args, **kwargs):
        # Get or create CLI context
        ctx = click.get_current_context()
        if not hasattr(ctx, 'obj') or not isinstance(ctx.obj, CLIContext):
            ctx.obj = create_modern_cli_context()
        
        # Enhance the context with modern CLI
        modern_cli = ModernCLI(ctx.obj)
        ctx.obj.modern_cli = modern_cli
        
        return func(*args, **kwargs)
    
    return wrapper


def create_modern_cli_context(
    config_file: Optional[Path] = None,
    verbose: bool = False,
    quiet: bool = False,
    working_directory: Optional[Path] = None
) -> CLIContext:
    """
    Create a modern CLI context with intelligent defaults.
    
    Args:
        config_file: Optional configuration file path
        verbose: Enable verbose output
        quiet: Suppress non-error output  
        working_directory: Working directory for operations
        
    Returns:
        Configured CLIContext instance
    """
    if working_directory is None:
        working_directory = Path.cwd()
    
    # Create base context
    context = CLIContext(
        config_file=config_file,
        verbose=verbose,
        quiet=quiet,
        working_directory=working_directory
    )
    
    # Create session-only user profile
    try:
        from .user_profiles import create_default_profile
        context.user_profile = create_default_profile()
    except Exception:
        # Fallback to None if profile creation fails
        context.user_profile = None
    
    # Initialize guidance system
    try:
        from .guidance_system import GuidanceEngine
        context.guidance_engine = GuidanceEngine(context)
    except Exception:
        # Fallback without guidance
        pass
    
    # Initialize progress manager
    try:
        from .progress_display import ProgressManager
        context.progress_manager = ProgressManager(context)
    except Exception:
        # Fallback without progress
        pass
    
    return context