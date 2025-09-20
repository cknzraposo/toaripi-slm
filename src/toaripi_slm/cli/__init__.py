"""
Toaripi SLM - Command Line Interface

A comprehensive CLI tool for training, testing, and interacting with 
the Toaripi Small Language Model for educational content generation.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt, IntPrompt, FloatPrompt
from rich.text import Text
from rich import print as rprint

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

console = Console()

# CLI Configuration
CLI_VERSION = "0.1.0"
DEFAULT_CONFIG_DIR = Path("./configs")
DEFAULT_DATA_DIR = Path("./data")
DEFAULT_MODELS_DIR = Path("./models")

class CLIContext:
    """Shared context for CLI commands."""
    
    def __init__(self):
        self.config_dir = DEFAULT_CONFIG_DIR
        self.data_dir = DEFAULT_DATA_DIR
        self.models_dir = DEFAULT_MODELS_DIR
        self.verbose = False
        self.debug = False
    
    def log(self, message: str, level: str = "info"):
        """Log messages with appropriate styling."""
        if level == "error":
            console.print(f"❌ {message}", style="red")
        elif level == "warning":
            console.print(f"⚠️  {message}", style="yellow")
        elif level == "success":
            console.print(f"✅ {message}", style="green")
        elif level == "info":
            console.print(f"ℹ️  {message}", style="blue")
        elif level == "debug" and self.debug:
            console.print(f"🐛 {message}", style="dim")

# Global CLI context
cli_context = CLIContext()

def show_banner():
    """Display the Toaripi SLM banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                         Toaripi SLM CLI                          ║
    ║                                                                  ║
    ║        Educational Content Generation for Toaripi Language       ║
    ║                           Version {version}                           ║
    ╚══════════════════════════════════════════════════════════════════╝
    """.format(version=CLI_VERSION)
    
    console.print(Panel(banner.strip(), style="bold blue"))

def check_environment() -> Dict[str, Any]:
    """Check the current environment and dependencies."""
    status = {
        "python_version": sys.version_info[:2],
        "platform": sys.platform,
        "working_directory": Path.cwd(),
        "config_dir_exists": DEFAULT_CONFIG_DIR.exists(),
        "data_dir_exists": DEFAULT_DATA_DIR.exists(),
        "models_dir_exists": DEFAULT_MODELS_DIR.exists(),
        "dependencies": {}
    }
    
    # Check for required dependencies
    dependencies = [
        "torch", "transformers", "datasets", "accelerate", 
        "peft", "yaml", "pandas", "numpy"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            status["dependencies"][dep] = "✅ Available"
        except ImportError:
            status["dependencies"][dep] = "❌ Missing"
    
    return status

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", "-d", is_flag=True, help="Enable debug output")
@click.option("--config-dir", type=click.Path(exists=True), help="Configuration directory")
@click.pass_context
def cli(ctx, verbose, debug, config_dir):
    """
    Toaripi SLM CLI - Train, test, and interact with language models 
    for Toaripi educational content generation.
    
    This tool guides you through the complete workflow of creating 
    and using a small language model for generating educational 
    content in the Toaripi language.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Update global context
    cli_context.verbose = verbose
    cli_context.debug = debug
    
    if config_dir:
        cli_context.config_dir = Path(config_dir)
    
    # Show banner for interactive commands
    if ctx.invoked_subcommand != "status" and not verbose:
        show_banner()

@cli.command()
@click.option("--detailed", "-d", is_flag=True, help="Show detailed system information")
def status(detailed):
    """Check system status and environment setup."""
    
    env_status = check_environment()
    
    # Create status table
    table = Table(title="🔍 System Status", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")
    
    # Basic system info
    table.add_row(
        "Python Version", 
        f"✅ {env_status['python_version'][0]}.{env_status['python_version'][1]}", 
        "Compatible" if env_status['python_version'] >= (3, 10) else "⚠️ Requires Python 3.10+"
    )
    
    table.add_row("Platform", f"✅ {env_status['platform']}", "Supported")
    
    # Directory structure
    table.add_row(
        "Config Directory", 
        "✅ Found" if env_status['config_dir_exists'] else "❌ Missing",
        str(cli_context.config_dir)
    )
    
    table.add_row(
        "Data Directory", 
        "✅ Found" if env_status['data_dir_exists'] else "❌ Missing",
        str(cli_context.data_dir)
    )
    
    table.add_row(
        "Models Directory", 
        "✅ Found" if env_status['models_dir_exists'] else "❌ Missing",
        str(cli_context.models_dir)
    )
    
    console.print(table)
    
    if detailed:
        # Dependencies table
        deps_table = Table(title="📦 Dependencies", show_header=True, header_style="bold magenta")
        deps_table.add_column("Package", style="cyan")
        deps_table.add_column("Status", style="green")
        
        for dep, status in env_status['dependencies'].items():
            deps_table.add_row(dep, status)
        
        console.print("\n")
        console.print(deps_table)
    
    # Summary
    missing_deps = [dep for dep, status in env_status['dependencies'].items() if "❌" in status]
    
    if missing_deps:
        console.print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        console.print("Run: [bold cyan]pip install -r requirements.txt[/bold cyan]")
    else:
        console.print("\n✅ All dependencies are available!")

# Import command modules
from .commands.train import train
from .commands.test import test  
from .commands.interact import interact
from .commands.doctor import doctor

# Register commands
cli.add_command(train)
cli.add_command(test)
cli.add_command(interact)
cli.add_command(doctor)

def main():
    """Entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n❌ An unexpected error occurred: {e}", style="red")
        if cli_context.debug:
            import traceback
            console.print("\n🐛 Debug traceback:")
            console.print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()