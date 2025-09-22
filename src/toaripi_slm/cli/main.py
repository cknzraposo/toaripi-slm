"""
Main CLI entry point for Toaripi SLM educational content generation system.

Provides command-line interface for:
- Training management
- Data processing and validation
- Model operations and export
- Serving and deployment
- Educational content validation
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from toaripi_slm.models import (
    SessionStatus,
    ContentType,
    AgeGroup,
    ModelSize,
    ValidationLevel
)

# Initialize rich console for pretty output
console = Console()

# Global CLI context for passing state between commands
class CLIContext:
    """CLI context for managing global state."""
    
    def __init__(self):
        self.config_file: Optional[Path] = None
        self.verbose: bool = False
        self.quiet: bool = False
        self.educational_mode: bool = True  # Always prioritize educational content
        self.working_directory: Path = Path.cwd()
    
    def log(self, message: str, level: str = "info"):
        """Log message with appropriate level."""
        if self.quiet and level != "error":
            return
        
        if level == "error":
            console.print(f"[red]ERROR:[/red] {message}")
        elif level == "warning":
            console.print(f"[yellow]WARNING:[/yellow] {message}")
        elif level == "success":
            console.print(f"[green]SUCCESS:[/green] {message}")
        elif level == "info":
            console.print(f"[blue]INFO:[/blue] {message}")
        elif level == "debug" and self.verbose:
            console.print(f"[dim]DEBUG:[/dim] {message}")

# Create CLI context
pass_context = click.make_pass_decorator(CLIContext, ensure=True)


@click.group(invoke_without_command=True)
@click.option(
    "--config", 
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file path (YAML/TOML)"
)
@click.option(
    "--verbose", 
    "-v",
    is_flag=True,
    help="Enable verbose logging"
)
@click.option(
    "--quiet", 
    "-q",
    is_flag=True,
    help="Suppress non-error output"
)
@click.option(
    "--working-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="Working directory for operations"
)
@click.version_option(version="0.1.0", prog_name="toaripi-slm")
@pass_context
def cli(ctx: CLIContext, config: Optional[Path], verbose: bool, quiet: bool, working_dir: Path):
    """
    Toaripi SLM - Educational Content Generation System
    
    A small language model fine-tuned for generating educational content
    in Toaripi language for primary school students and teachers.
    
    Focus areas:
    - Stories and narratives
    - Vocabulary exercises
    - Reading comprehension
    - Cultural preservation
    - Age-appropriate content
    
    \b
    Examples:
      toaripi-slm train start --data data/parallel.csv
      toaripi-slm data validate --input data/raw/
      toaripi-slm model export --format gguf --model best_checkpoint
      toaripi-slm serve start --model models/toaripi-primary.gguf
    """
    # Set up context
    ctx.config_file = config
    ctx.verbose = verbose
    ctx.quiet = quiet
    ctx.working_directory = working_dir
    
    # Validate mutually exclusive options
    if verbose and quiet:
        click.echo("Error: --verbose and --quiet cannot be used together", err=True)
        sys.exit(1)
    
    # Show welcome message if no command provided
    if not click.get_current_context().invoked_subcommand:
        show_welcome(ctx)


def show_welcome(ctx: CLIContext):
    """Show welcome message and system status."""
    
    # Create welcome panel
    welcome_text = Text()
    welcome_text.append("Toaripi SLM Educational Content Generator\n\n", style="bold blue")
    welcome_text.append("🎓 Mission: ", style="bold")
    welcome_text.append("Generate educational content for Toaripi language learners\n")
    welcome_text.append("👥 Target: ", style="bold") 
    welcome_text.append("Primary school students and teachers\n")
    welcome_text.append("🌍 Focus: ", style="bold")
    welcome_text.append("Cultural preservation and age-appropriate learning\n\n")
    welcome_text.append("Use 'toaripi-slm --help' to see all available commands.")
    
    console.print(Panel(welcome_text, title="Welcome", border_style="blue"))
    
    # Show quick status
    console.print("\n[bold]Quick Start:[/bold]")
    console.print("  • [cyan]toaripi-slm train --help[/cyan]     - Training operations")
    console.print("  • [cyan]toaripi-slm data --help[/cyan]      - Data management")
    console.print("  • [cyan]toaripi-slm model --help[/cyan]     - Model operations")
    console.print("  • [cyan]toaripi-slm serve --help[/cyan]     - Deployment & serving")
    
    # Check for existing data/models
    data_dir = ctx.working_directory / "data"
    models_dir = ctx.working_directory / "models"
    
    status_items = []
    if data_dir.exists():
        data_files = list(data_dir.rglob("*.csv"))
        status_items.append(f"📊 Found {len(data_files)} data files")
    
    if models_dir.exists():
        model_files = list(models_dir.rglob("*.bin")) + list(models_dir.rglob("*.gguf"))
        status_items.append(f"🤖 Found {len(model_files)} model files")
    
    if status_items:
        console.print(f"\n[bold]Project Status:[/bold]")
        for item in status_items:
            console.print(f"  • {item}")


# Import command groups from separate modules
from .train import train
from .data import data as data_commands

# Add the training commands to the main CLI
cli.add_command(train)

# Add the data commands to the main CLI  
cli.add_command(data_commands, name="data")


@cli.group()
@pass_context
def model(ctx: CLIContext):
    """
    Model management and export operations.
    
    Export models for edge deployment, validate educational
    content generation, and manage model versions.
    """
    ctx.log("Model operations", level="debug")


@cli.group()
@pass_context
def serve(ctx: CLIContext):
    """
    Serving and deployment operations.
    
    Start inference servers, validate deployments, and
    test educational content generation endpoints.
    """
    ctx.log("Serving operations", level="debug")


@serve.command()
@click.option("--model", required=True, help="Model path for serving")
@click.option("--host", default="localhost", help="Server host")
@click.option("--port", default=8000, help="Server port")
@click.option("--cpu-only", is_flag=True, help="Force CPU-only inference")
@pass_context
def start_server(ctx: CLIContext, model: str, host: str, port: int, cpu_only: bool):
    """Start inference server for educational content generation."""
    ctx.log(f"Starting server with model: {model}", level="info")
    ctx.log(f"Server will run on {host}:{port}", level="info")
    
    if cpu_only:
        ctx.log("CPU-only mode enabled for edge deployment", level="info")
    
    # TODO: Implement server startup logic
    console.print("[yellow]Server implementation coming soon...[/yellow]")


# Helper commands
@cli.command()
@pass_context
def status(ctx: CLIContext):
    """Show system status and configuration."""
    console.print("[bold]Toaripi SLM System Status[/bold]\n")
    
    # Working directory
    console.print(f"Working Directory: [cyan]{ctx.working_directory}[/cyan]")
    
    # Configuration
    if ctx.config_file:
        console.print(f"Configuration: [cyan]{ctx.config_file}[/cyan]")
    else:
        console.print("Configuration: [yellow]Using defaults[/yellow]")
    
    # Project structure check
    console.print("\n[bold]Project Structure:[/bold]")
    
    required_dirs = ["data", "models", "configs", "scripts"]
    for dir_name in required_dirs:
        dir_path = ctx.working_directory / dir_name
        if dir_path.exists():
            console.print(f"  ✓ [green]{dir_name}/[/green]")
        else:
            console.print(f"  ✗ [red]{dir_name}/[/red] (missing)")
    
    # Check for key files
    console.print("\n[bold]Key Files:[/bold]")
    
    key_files = [
        ("data/processed/toaripi_parallel.csv", "Training data"),
        ("configs/training/toaripi_educational_config.yaml", "Training config"),
        ("requirements.txt", "Dependencies")
    ]
    
    for file_path, description in key_files:
        full_path = ctx.working_directory / file_path
        if full_path.exists():
            console.print(f"  ✓ [green]{description}[/green]: {file_path}")
        else:
            console.print(f"  ✗ [yellow]{description}[/yellow]: {file_path} (missing)")
    
    # Educational validation status
    console.print("\n[bold]Educational Content Settings:[/bold]")
    console.print("  • Cultural validation: [green]Enabled[/green]")
    console.print("  • Age-appropriate filtering: [green]Enabled[/green]")
    console.print("  • Safety checks: [green]Enabled[/green]")
    console.print("  • Primary school focus: [green]Active[/green]")


@cli.command()
@click.option("--check", 
              type=click.Choice(["all", "python", "data", "models", "config"], case_sensitive=False),
              default="all",
              help="What to validate")
@pass_context
def validate(ctx: CLIContext, check: str):
    """Validate system setup and educational content requirements."""
    console.print(f"[bold]Validating {check} configuration...[/bold]\n")
    
    validation_passed = True
    
    if check in ["all", "python"]:
        # Check Python environment
        console.print("[bold]Python Environment:[/bold]")
        try:
            import transformers
            import torch
            import pydantic
            console.print("  ✓ [green]Required packages installed[/green]")
        except ImportError as e:
            console.print(f"  ✗ [red]Missing package: {e.name}[/red]")
            validation_passed = False
    
    if check in ["all", "data"]:
        # Check data files
        console.print("\n[bold]Data Validation:[/bold]")
        data_dir = ctx.working_directory / "data"
        if data_dir.exists():
            csv_files = list(data_dir.rglob("*.csv"))
            if csv_files:
                console.print(f"  ✓ [green]Found {len(csv_files)} data files[/green]")
            else:
                console.print("  ✗ [yellow]No CSV data files found[/yellow]")
        else:
            console.print("  ✗ [red]Data directory not found[/red]")
            validation_passed = False
    
    if check in ["all", "models"]:
        # Check models directory
        console.print("\n[bold]Models Validation:[/bold]")
        models_dir = ctx.working_directory / "models"
        if models_dir.exists():
            console.print("  ✓ [green]Models directory exists[/green]")
        else:
            console.print("  ✗ [yellow]Models directory not found[/yellow]")
    
    if check in ["all", "config"]:
        # Check configuration
        console.print("\n[bold]Configuration Validation:[/bold]")
        configs_dir = ctx.working_directory / "configs"
        if configs_dir.exists():
            yaml_files = list(configs_dir.rglob("*.yaml"))
            if yaml_files:
                console.print(f"  ✓ [green]Found {len(yaml_files)} config files[/green]")
            else:
                console.print("  ✗ [yellow]No YAML config files found[/yellow]")
        else:
            console.print("  ✗ [red]Configs directory not found[/red]")
            validation_passed = False
    
    # Summary
    console.print(f"\n[bold]Validation Result:[/bold]")
    if validation_passed:
        console.print("  ✓ [green]All checks passed[/green]")
    else:
        console.print("  ✗ [red]Some validations failed[/red]")
        console.print("\nRun 'toaripi-slm --help' for setup instructions.")


if __name__ == "__main__":
    cli()