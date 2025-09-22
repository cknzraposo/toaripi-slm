"""
Modern CLI Main Entry Point

Integrates all modern CLI components to provide a sleek, user-friendly
command-line experience for the Toaripi SLM educational content system.
"""

import sys
import os
from pathlib import Path
from typing import Optional

import click

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import modern CLI components
from .modern import (
    ModernCLI, CLIContext, 
    UserProfileManager
)
from .modern.framework import create_modern_cli_context, with_modern_cli
from .modern.workflows import SmartWelcome
from .modern.error_handling import SmartErrorRecovery

# Import existing command groups
from .train import train
from .data import data as data_commands
from .model import model as model_commands
from .serve import serve as serve_commands


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
@click.option(
    "--profile",
    type=click.Choice(["beginner", "intermediate", "advanced", "expert"]),
    help="Override user experience profile"
)
@click.option(
    "--interactive",
    "-i", 
    is_flag=True,
    help="Start in interactive mode"
)
@click.version_option(version="0.1.0", prog_name="toaripi")
@click.pass_context
def cli(ctx, config: Optional[Path], verbose: bool, quiet: bool, working_dir: Path, 
        profile: Optional[str], interactive: bool):
    """
    🌟 Toaripi SLM - Educational Content Generation System
    
    A modern, user-friendly AI system for creating educational content
    in Toaripi language for primary school students and teachers.
    
    🎓 Focus areas:
    • Stories and narratives for young learners
    • Vocabulary exercises with cultural context  
    • Reading comprehension materials
    • Interactive dialogues for classroom practice
    • Cultural preservation through education
    
    ✨ Key features:
    • Offline-compatible for classroom use
    • Age-appropriate content validation
    • Cultural sensitivity built-in
    • Beginner-friendly guided workflows
    • Modern, beautiful command-line interface
    
    \b
    🚀 Quick start examples:
      toaripi                    # Interactive welcome and guidance
      toaripi train start        # Train your first model (guided)
      toaripi data validate      # Check your training data
      toaripi interact           # Generate educational content
      toaripi status             # Check system health
    
    💡 New to command-line tools? Try: toaripi --interactive
    """
    
    # Validate mutually exclusive options
    if verbose and quiet:
        click.echo("❌ Error: --verbose and --quiet cannot be used together", err=True)
        sys.exit(1)
    
    # Create modern CLI context
    cli_context = create_modern_cli_context(
        config_file=config,
        verbose=verbose,
        quiet=quiet,
        working_directory=working_dir
    )
    
    # Override profile if specified
    if profile:
        profile_manager = UserProfileManager()
        current_profile = profile_manager.load_profile()
        current_profile.experience_level = profile
        profile_manager.save_profile(current_profile)
        cli_context.user_profile = current_profile
    
    # Store context for subcommands
    ctx.obj = cli_context
    
    # Create modern CLI instance
    modern_cli = ModernCLI(cli_context)
    cli_context.modern_cli = modern_cli
    
    # Install error handling
    error_recovery = SmartErrorRecovery(cli_context)
    error_recovery.install_global_handler()
    
    # If no subcommand provided, show smart welcome
    if not ctx.invoked_subcommand:
        if interactive:
            smart_welcome = SmartWelcome(cli_context)
            smart_welcome.interactive_mode()
        else:
            smart_welcome = SmartWelcome(cli_context)
            smart_welcome.show_welcome()


# Enhanced command groups with modern CLI integration
@cli.group()
@click.pass_context
def train(ctx):
    """🎓 Train educational content models with guided workflows."""
    pass


@cli.group()
@click.pass_context  
def data(ctx):
    """📚 Manage and validate training data for educational content."""
    pass


@cli.group()
@click.pass_context
def model(ctx):
    """🤖 Test, optimize, and export trained models."""
    pass


@cli.group()
@click.pass_context
def serve(ctx):
    """🚀 Deploy and serve models for classroom use."""
    pass


# Modern system commands
@cli.command()
@click.pass_context
@with_modern_cli
def status(ctx):
    """📊 Show comprehensive system status and health check."""
    
    modern_cli = ctx.obj.modern_cli
    
    # Analyze system components
    components = modern_cli._check_system_components()
    
    # Show status with modern formatting
    modern_cli.progress_manager.show_status(components)
    
    # Show next steps based on status
    if ctx.obj.guidance_engine:
        suggestions = ctx.obj.guidance_engine.suggest_next_actions(max_suggestions=3)
        if suggestions:
            steps = [
                {
                    "title": s.title,
                    "description": s.description,
                    "command": s.command,
                    "emoji": s.emoji
                }
                for s in suggestions
            ]
            modern_cli.progress_manager.show_next_steps(steps)


@cli.command()
@click.option("--interactive", "-i", is_flag=True, help="Interactive help mode")
@click.option("--topic", "-t", help="Get help on specific topic")
@click.argument("command", required=False)
@click.pass_context
@with_modern_cli
def help(ctx, interactive: bool, topic: Optional[str], command: Optional[str]):
    """❓ Get intelligent help and guidance for any task."""
    
    modern_cli = ctx.obj.modern_cli
    
    if interactive:
        smart_welcome = SmartWelcome(ctx.obj)
        smart_welcome.interactive_mode()
    elif command:
        modern_cli.show_command_help(command, topic or "")
    elif topic:
        # Show topic-specific help
        help_content = ctx.obj.guidance_engine.get_help(topic)
        if help_content:
            modern_cli.console.print(f"📖 Help for: {help_content.title}")
            modern_cli.console.print(help_content.content)
        else:
            modern_cli.console.print(f"❌ No help found for topic: {topic}")
    else:
        # Show general help
        ctx.get_help()


@cli.command()
@click.option("--reset", is_flag=True, help="Reset profile to defaults")
@click.option("--show", is_flag=True, help="Show current profile")
@click.pass_context
@with_modern_cli
def config(ctx, reset: bool, show: bool):
    """⚙️ Manage user profile and system configuration."""
    
    modern_cli = ctx.obj.modern_cli
    profile_manager = UserProfileManager()
    
    if reset:
        if modern_cli.confirm_action("Reset your profile to defaults?", default=False):
            profile_manager.reset_profile()
            modern_cli.console.print("✅ Profile reset successfully!")
    elif show:
        profile = profile_manager.get_current_profile()
        
        # Display profile information
        try:
            from rich.table import Table
            
            profile_table = Table(title="👤 Your Profile", show_header=True, header_style="bold magenta")
            profile_table.add_column("Setting", style="bold")
            profile_table.add_column("Value", style="cyan")
            
            profile_table.add_row("Name", profile.display_name)
            profile_table.add_row("Role", profile.user_type.replace("_", " ").title())
            profile_table.add_row("Experience Level", profile.experience_level.title())
            profile_table.add_row("Target Age Group", profile.target_age_group.replace("_", " ").title())
            profile_table.add_row("Preferred Workflow", profile.preferred_workflow.title())
            profile_table.add_row("Show Tips", "Yes" if profile.show_tips else "No")
            
            modern_cli.console.print(profile_table)
            
        except ImportError:
            print(f"Name: {profile.display_name}")
            print(f"Role: {profile.user_type}")
            print(f"Experience: {profile.experience_level}")
    else:
        # Interactive configuration
        profile = profile_manager.create_profile_interactively()
        modern_cli.console.print("✅ Profile updated successfully!")


@cli.command()
@click.argument("query", required=False)
@click.pass_context
@with_modern_cli
def ask(ctx, query: Optional[str]):
    """💬 Ask in natural language what you want to do."""
    
    modern_cli = ctx.obj.modern_cli
    
    if not query:
        # Interactive mode
        try:
            from rich.prompt import Prompt
            query = Prompt.ask("🤔 [bold blue]What would you like to do?[/bold blue]")
        except ImportError:
            query = input("🤔 What would you like to do? ")
    
    if query:
        suggested_command = modern_cli.handle_natural_language_input(query)
        if suggested_command:
            if modern_cli.confirm_action(f"Run 'toaripi {suggested_command}'?"):
                # This would execute the suggested command
                modern_cli.console.print(f"🚀 Running: toaripi {suggested_command}")
                # Here you would actually invoke the command
            else:
                modern_cli.console.print("💡 Try 'toaripi help' to see all available commands")
        else:
            modern_cli.console.print("🤔 I'm not sure what you'd like to do.")
            modern_cli.console.print("💡 Try 'toaripi help' to see what's available")


# Add enhanced subcommands from existing modules
# These would be enhanced versions that use the modern CLI framework

@train.command(name="start")
@click.option("--beginner", is_flag=True, help="Beginner-friendly guided training")
@click.option("--quick", is_flag=True, help="Quick start with sensible defaults")
@click.option("--teacher", is_flag=True, help="Teacher-focused training mode")
@click.pass_context
@with_modern_cli
def train_start(ctx, beginner: bool, quick: bool, teacher: bool):
    """🎓 Start training an educational content model."""
    
    modern_cli = ctx.obj.modern_cli
    
    # Determine appropriate training mode based on user profile
    profile = ctx.obj.user_profile
    if profile:
        if profile.experience_level == "beginner" or beginner:
            beginner = True
        if profile.user_type == "teacher" or teacher:
            teacher = True
    
    # Show training intro
    if beginner or teacher:
        modern_cli.console.print("🎓 [bold blue]Starting guided educational model training...[/bold blue]")
        modern_cli.console.print("This will create an AI model specifically for generating")
        modern_cli.console.print("age-appropriate educational content in Toaripi language.\n")
    
    # This would integrate with the existing training system
    # For now, show what would happen
    modern_cli.console.print("✨ Training system integration coming soon...")
    modern_cli.console.print("This will connect to the existing training pipeline with modern UI")


@data.command(name="validate")
@click.option("--detailed", is_flag=True, help="Show detailed validation report")
@click.pass_context
@with_modern_cli
def data_validate(ctx, detailed: bool):
    """📊 Validate training data for educational content."""
    
    modern_cli = ctx.obj.modern_cli
    modern_cli.console.print("📊 [bold blue]Validating educational training data...[/bold blue]")
    
    # This would integrate with existing data validation
    modern_cli.console.print("✨ Data validation integration coming soon...")


if __name__ == "__main__":
    cli()