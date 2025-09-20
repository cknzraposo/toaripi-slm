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

# Import the unified context system
from .context import get_context, CLI_VERSION

console = Console()

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

# Import new command groups
from .commands.workflow import workflow
from .commands.model import model  
from .commands.data import data
from .commands.chat import chat
from .commands.system import system

# Import smart help system
from .enhanced import smart_help_command, EnhancedGroup, EnhancedCommand
from .smart_help import smart_help, show_command_suggestions

# Create enhanced CLI with smart help
@click.group(cls=EnhancedGroup)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", "-d", is_flag=True, help="Enable debug output")
@click.option("--config-dir", type=click.Path(exists=True), help="Configuration directory")
@click.option("--profile", type=click.Choice(["beginner", "intermediate", "advanced", "expert"]),
              help="Set user profile (overrides saved preference)")
@click.pass_context
def cli(ctx, verbose, debug, config_dir, profile):
    """
    Toaripi SLM CLI - Train, test, and interact with language models 
    for Toaripi educational content generation.
    
    This tool guides you through the complete workflow of creating 
    and using a small language model for generating educational 
    content in the Toaripi language.
    
    \b
    Command Groups:
      workflow    Guided workflows for common tasks
      model       Train, test, and manage models  
      data        Validate and process training data
      chat        Interactive chat sessions
      system      System diagnostics and configuration
      help        Smart help and command discovery
    
    Smart Help Features:
      • Command suggestions based on typos or partial matches
      • Contextual help based on your current setup
      • Progressive guidance based on experience level
      • Related command recommendations
    
    Examples:
      toaripi workflow quickstart     # Get started quickly
      toaripi help suggest train      # Find training commands
      toaripi help examples           # See contextual examples
      toaripi help guide              # Get level-appropriate help
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Get unified context
    context = get_context()
    
    # Update context with CLI options
    if verbose:
        context.verbose = verbose
    if debug:
        context.debug = debug
    if config_dir:
        context.config_dir = Path(config_dir)
    if profile:
        context.profile = profile
        context.save_preferences()
    
    # Store context for subcommands
    ctx.obj = context
    
    # Show banner and smart help for main command (but not for subcommands)
    if ctx.invoked_subcommand is None:
        show_banner()
        
        # Show smart contextual help
        console.print("\n🧠 [bold blue]Smart Help:[/bold blue]")
        help_data = smart_help.get_contextual_help()
        
        if help_data["recommendations"]:
            console.print("💡 Recommendations:")
            for rec in help_data["recommendations"][:2]:
                priority_icon = "🚨" if rec["priority"] == "high" else "ℹ️"
                console.print(f"   {priority_icon} {rec['message']}")
        
        if help_data["next_steps"]:
            console.print("🚀 Quick Start:")
            for step in help_data["next_steps"][:2]:
                console.print(f"   • {step}")
        
        console.print("\n💬 Try: [cyan]toaripi help suggest <what you want to do>[/cyan]")
        console.print("📖 Or: [cyan]toaripi help guide[/cyan] for your experience level\n")

# Register command groups
cli.add_command(workflow)
cli.add_command(model)
cli.add_command(data)
cli.add_command(chat)
cli.add_command(system)

# Add smart help commands
cli.add_command(smart_help_command())

# Legacy individual commands for backward compatibility (deprecated)
@cli.group(hidden=True)
def legacy():
    """Legacy individual commands (deprecated - use command groups instead)."""
    pass

# Import legacy commands if they exist
try:
    from .commands.train import train
    from .commands.test import test  
    from .commands.interact import interact
    from .commands.doctor import doctor
    from .commands.models import models
    from .commands.export import export
    from .commands.sessions import sessions
    
    # Add to legacy group
    legacy.add_command(train)
    legacy.add_command(test)
    legacy.add_command(interact)
    legacy.add_command(doctor)
    legacy.add_command(models)
    legacy.add_command(export)
    legacy.add_command(sessions)
    
    # Also add to main CLI for backward compatibility
    cli.add_command(train)
    cli.add_command(test)
    cli.add_command(interact)
    
except ImportError:
    # Legacy commands don't exist yet
    pass

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", "-d", is_flag=True, help="Enable debug output")
@click.option("--config-dir", type=click.Path(exists=True), help="Configuration directory")
@click.option("--profile", type=click.Choice(["beginner", "intermediate", "advanced", "expert"]),
              help="Set user profile (overrides saved preference)")
@click.pass_context
def cli(ctx, verbose, debug, config_dir, profile):
    """
    Toaripi SLM CLI - Train, test, and interact with language models 
    for Toaripi educational content generation.
    
    This tool guides you through the complete workflow of creating 
    and using a small language model for generating educational 
    content in the Toaripi language.
    
    \b
    Command Groups:
      workflow    Guided workflows for common tasks
      model       Train, test, and manage models  
      data        Validate and process training data
      chat        Interactive chat sessions
      system      System diagnostics and configuration
    
    Examples:
      toaripi workflow quickstart     # Get started quickly
      toaripi model train --help      # See training options
      toaripi chat                    # Start interactive session
      toaripi system status           # Check system health
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Get unified context
    context = get_context()
    
    # Update context with CLI options
    if verbose:
        context.verbose = verbose
    if debug:
        context.debug = debug
    if config_dir:
        context.config_dir = Path(config_dir)
    if profile:
        context.profile = profile
        context.save_preferences()
    
    # Store context for subcommands
    ctx.obj = context
    
    # Show banner for main command (but not for subcommands)
    if ctx.invoked_subcommand is None:
        show_banner()
        
        # Show quick help
        console.print("\n� [bold blue]Quick Start:[/bold blue]")
        console.print("  • Run [cyan]toaripi workflow quickstart[/cyan] for guided setup")
        console.print("  • Run [cyan]toaripi system status[/cyan] to check system health") 
        console.print("  • Run [cyan]toaripi --help[/cyan] for complete command reference")
        console.print("  • Run [cyan]toaripi COMMAND --help[/cyan] for command-specific help\n")

# Import new command groups
from .commands.workflow import workflow
from .commands.model import model  
from .commands.data import data
from .commands.chat import chat
from .commands.system import system

# Register command groups
cli.add_command(workflow)
cli.add_command(model)
cli.add_command(data)
cli.add_command(chat)
cli.add_command(system)

# Legacy individual commands for backward compatibility (deprecated)
@cli.group(hidden=True)
def legacy():
    """Legacy individual commands (deprecated - use command groups instead)."""
    pass

# Import legacy commands if they exist
try:
    from .commands.train import train
    from .commands.test import test  
    from .commands.interact import interact
    from .commands.doctor import doctor
    from .commands.models import models
    from .commands.export import export
    from .commands.sessions import sessions
    
    # Add to legacy group
    legacy.add_command(train)
    legacy.add_command(test)
    legacy.add_command(interact)
    legacy.add_command(doctor)
    legacy.add_command(models)
    legacy.add_command(export)
    legacy.add_command(sessions)
    
    # Also add to main CLI for backward compatibility
    cli.add_command(train)
    cli.add_command(test)
    cli.add_command(interact)
    
except ImportError:
    # Legacy commands don't exist yet
    pass

def main():
    """Entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n❌ An unexpected error occurred: {e}", style="red")
        
        # Get context for debug flag
        try:
            context = get_context()
            if context.debug:
                import traceback
                console.print("\n🐛 Debug traceback:")
                console.print(traceback.format_exc())
        except:
            pass
            
        sys.exit(1)

if __name__ == "__main__":
    main()