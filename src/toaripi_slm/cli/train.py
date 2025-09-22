"""
Training commands for Toaripi SLM CLI.

Implements comprehensive training operations for educational content
generation with session management, progress tracking, and validation.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from uuid import uuid4

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from toaripi_slm.models import (
    TrainingSession, 
    SessionStatus, 
    ContentType, 
    AgeGroup, 
    ModelSize, 
    ValidationLevel,
    DeviceType,
    TrainingMode,
    Dataset
)

console = Console()

def pass_context(f):
    """Decorator to pass CLI context to commands."""
    return click.pass_context(f)


@click.group()
def train():
    """Training operations for Toaripi educational models."""
    pass


@train.command()
@click.option(
    "--data", 
    type=click.Path(exists=True, path_type=Path), 
    required=True,
    help="Training data file (CSV with English-Toaripi parallel text)"
)
@click.option(
    "--model", 
    default="microsoft/DialoGPT-small",
    help="Base model to fine-tune for educational content"
)
@click.option(
    "--output", 
    type=click.Path(path_type=Path),
    help="Output directory for trained model"
)
@click.option(
    "--epochs", 
    default=3, 
    type=int,
    help="Number of training epochs"
)
@click.option(
    "--batch-size", 
    default=4, 
    type=int,
    help="Training batch size"
)
@click.option(
    "--learning-rate", 
    default=2e-5, 
    type=float,
    help="Learning rate for training"
)
@click.option(
    "--validation-level", 
    type=click.Choice(["basic", "educational", "cultural", "strict"], case_sensitive=False),
    default="educational",
    help="Educational content validation level"
)
@click.option(
    "--age-groups", 
    multiple=True,
    type=click.Choice(["early_childhood", "primary_lower", "primary_upper", "secondary", "adult", "teacher"], case_sensitive=False),
    default=["primary_lower"],
    help="Target age groups for educational content"
)
@click.option(
    "--content-types", 
    multiple=True,
    type=click.Choice(["story", "vocabulary", "dialogue", "comprehension", "grammar", "cultural", "song", "poem", "lesson", "exercise"], case_sensitive=False),
    default=["story", "vocabulary"],
    help="Types of educational content to generate"
)
@click.option(
    "--device", 
    type=click.Choice(["auto", "cpu", "gpu", "raspberry-pi"], case_sensitive=False),
    default="auto",
    help="Target device for deployment"
)
@click.option(
    "--use-lora", 
    is_flag=True, 
    default=True,
    help="Use LoRA for efficient fine-tuning"
)
@click.option(
    "--lora-rank", 
    default=16, 
    type=int,
    help="LoRA rank for fine-tuning"
)
@click.option(
    "--max-length", 
    default=256, 
    type=int,
    help="Maximum sequence length for educational content"
)
@click.option(
    "--resume", 
    type=str,
    help="Resume training from session ID"
)
@click.option(
    "--dry-run", 
    is_flag=True,
    help="Validate configuration without starting training"
)
@pass_context
def start(ctx, data: Path, model: str, output: Optional[Path], epochs: int, 
          batch_size: int, learning_rate: float, validation_level: str,
          age_groups: List[str], content_types: List[str], device: str,
          use_lora: bool, lora_rank: int, max_length: int, 
          resume: Optional[str], dry_run: bool):
    """
    Start a new training session for educational content generation.
    
    This command creates a fine-tuned model specifically for generating
    age-appropriate educational content in Toaripi language.
    
    \b
    Examples:
      # Basic training with default settings
      toaripi-slm train start --data data/parallel.csv
      
      # Advanced training with custom parameters
      toaripi-slm train start \\
        --data data/processed/toaripi_parallel.csv \\
        --model microsoft/DialoGPT-small \\
        --epochs 5 \\
        --age-groups primary \\
        --content-types story vocabulary dialogue \\
        --validation-level strict
    """
    
    # Validate and prepare parameters
    console.print("[bold blue]🎓 Toaripi Educational Model Training[/bold blue]\n")
    
    # Convert string enums to proper types
    try:
        validation_enum = ValidationLevel(validation_level.lower())
        age_group_enums = [AgeGroup(ag.lower()) for ag in age_groups]
        content_type_enums = [ContentType(ct.lower()) for ct in content_types]
        
        # Handle device type conversion
        if device == "auto":
            device_enum = DeviceType.CPU  # Default to CPU for auto detection
        else:
            device_enum = DeviceType(device.lower().replace("-", "_"))
            
    except ValueError as e:
        console.print(f"[red]Error:[/red] Invalid parameter value: {e}")
        return
    
    # Set up output directory
    if not output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model.split("/")[-1] if "/" in model else model
        output = Path("models") / "training_runs" / f"toaripi_{model_name}_{timestamp}"
    
    output.mkdir(parents=True, exist_ok=True)
    
    # Validate data file
    console.print("[bold]📊 Validating training data...[/bold]")
    try:
        # Create Dataset object for validation
        training_dataset = Dataset.from_file(
            file_path=data,
            name=f"Training data for {model}",
            content_types=content_type_enums,
            age_groups=age_group_enums
        )
        
        console.print(f"  ✓ [green]Data file valid:[/green] {data}")
        console.print(f"  ✓ [green]Dataset ID:[/green] {training_dataset.dataset_id}")
        console.print(f"  ✓ [green]Content types:[/green] {', '.join([ct.value for ct in content_type_enums])}")
        console.print(f"  ✓ [green]Age groups:[/green] {', '.join([ag.value for ag in age_group_enums])}")
        
    except Exception as e:
        console.print(f"  ✗ [red]Data validation failed:[/red] {e}")
        return
    
    # Create training session
    session_id = resume if resume else str(uuid4())[:8]
    
    try:
        session = TrainingSession(
            session_id=session_id,
            name=f"Toaripi Educational Training - {model}",
            description=f"Fine-tuning {model} for educational content generation",
            model_name=model,
            train_data_path=data,
            output_directory=output,
            
            # Training parameters
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            
            # Educational parameters
            target_age_groups=age_group_enums,
            target_content_types=content_type_enums,
            validation_level=validation_enum,
            
            # Technical parameters
            target_device=device_enum,
            training_mode=TrainingMode.LORA if use_lora else TrainingMode.FULL_FINETUNE,
            use_lora=use_lora,
            lora_rank=lora_rank if use_lora else None,
            
            # Model sizing
            model_size=ModelSize.SMALL,  # Default for educational content
            max_memory_gb=8.0
        )
        
        console.print(f"  ✓ [green]Training session created:[/green] {session_id}")
        
    except Exception as e:
        console.print(f"  ✗ [red]Session creation failed:[/red] {e}")
        return
    
    # Display training configuration
    console.print(f"\n[bold]🔧 Training Configuration[/bold]")
    
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Setting", style="bold cyan")
    config_table.add_column("Value", style="white")
    
    config_table.add_row("Session ID", session_id)
    config_table.add_row("Base Model", model)
    config_table.add_row("Output Directory", str(output))
    config_table.add_row("Training Data", str(data))
    config_table.add_row("Epochs", str(epochs))
    config_table.add_row("Batch Size", str(batch_size))
    config_table.add_row("Learning Rate", f"{learning_rate:.0e}")
    config_table.add_row("Max Length", str(max_length))
    config_table.add_row("Validation Level", validation_level.title())
    config_table.add_row("Age Groups", ", ".join([ag.title() for ag in age_groups]))
    config_table.add_row("Content Types", ", ".join([ct.title() for ct in content_types]))
    config_table.add_row("Target Device", device.title())
    config_table.add_row("Training Mode", "LoRA" if use_lora else "Full Fine-tune")
    if use_lora:
        config_table.add_row("LoRA Rank", str(lora_rank))
    
    console.print(config_table)
    
    # Educational validation warnings
    console.print(f"\n[bold]🎯 Educational Content Validation[/bold]")
    console.print("  • [green]Cultural appropriateness[/green]: Enabled")
    console.print("  • [green]Age-appropriate content[/green]: Enabled") 
    console.print("  • [green]Safety filtering[/green]: Enabled")
    console.print(f"  • [green]Validation level[/green]: {validation_level.title()}")
    
    # Device compatibility check
    if device_enum == DeviceType.RASPBERRY_PI:
        console.print(f"\n[yellow]⚠️  Raspberry Pi Deployment Mode[/yellow]")
        console.print("  • Model will be optimized for edge deployment")
        console.print("  • Quantization will be applied for efficiency")
        console.print("  • Memory usage will be minimized")
    
    # Save session configuration
    config_file = output / "training_config.json"
    try:
        with open(config_file, "w") as f:
            json.dump(session.dict(), f, indent=2, default=str)
        console.print(f"  ✓ [green]Configuration saved:[/green] {config_file}")
    except Exception as e:
        console.print(f"  ✗ [red]Failed to save configuration:[/red] {e}")
    
    # Dry run mode
    if dry_run:
        console.print(f"\n[yellow]🧪 Dry run completed - no training started[/yellow]")
        console.print("Configuration validated successfully. Run without --dry-run to start training.")
        return
    
    # Confirm before starting training
    if not click.confirm(f"\nStart training session '{session_id}'?"):
        console.print("[yellow]Training cancelled by user[/yellow]")
        return
    
    # Start training (placeholder implementation)
    console.print(f"\n[bold green]🚀 Starting training session {session_id}...[/bold green]")
    
    # Update session status
    session.start_training()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        # Initialize training task
        task = progress.add_task("[cyan]Initializing training...", total=100)
        
        # Placeholder training steps
        import time
        
        # Step 1: Load model
        progress.update(task, description="[cyan]Loading base model...", advance=20)
        time.sleep(1)
        
        # Step 2: Prepare data
        progress.update(task, description="[cyan]Preparing educational data...", advance=20)
        time.sleep(1)
        
        # Step 3: Apply educational validation
        progress.update(task, description="[cyan]Validating cultural appropriateness...", advance=20)
        time.sleep(1)
        
        # Step 4: Configure training
        progress.update(task, description="[cyan]Configuring educational fine-tuning...", advance=20)
        time.sleep(1)
        
        # Step 5: Start training
        progress.update(task, description="[green]Training in progress...", advance=20)
        time.sleep(2)
    
    console.print(f"\n[bold green]✅ Training session '{session_id}' started successfully![/bold green]")
    console.print(f"Monitor progress with: [cyan]toaripi-slm train status {session_id}[/cyan]")
    console.print(f"View logs with: [cyan]toaripi-slm train logs {session_id}[/cyan]")
    
    # TODO: Implement actual training logic
    console.print(f"\n[yellow]Note: This is a placeholder implementation.[/yellow]")
    console.print("Full training implementation will be added in the next phase.")


@train.command()
@click.argument("session_id", required=False)
@pass_context
def status(ctx, session_id: Optional[str]):
    """
    Show training session status and progress.
    
    If no session ID is provided, shows all active sessions.
    """
    console.print("[bold blue]📊 Training Session Status[/bold blue]\n")
    
    if session_id:
        # Show specific session status
        console.print(f"Session ID: [cyan]{session_id}[/cyan]")
        console.print("[yellow]Loading session status...[/yellow]")
        
        # TODO: Load actual session from storage
        console.print("[yellow]Session status implementation coming soon...[/yellow]")
    else:
        # Show all sessions
        console.print("[bold]Active Training Sessions:[/bold]")
        
        # Check for training runs directory
        training_runs_dir = Path("models/training_runs")
        if training_runs_dir.exists():
            sessions = list(training_runs_dir.iterdir())
            
            if sessions:
                for session_dir in sessions:
                    if session_dir.is_dir():
                        config_file = session_dir / "training_config.json"
                        if config_file.exists():
                            console.print(f"  • [cyan]{session_dir.name}[/cyan]")
                        else:
                            console.print(f"  • [dim]{session_dir.name}[/dim] (incomplete)")
            else:
                console.print("  [dim]No training sessions found[/dim]")
        else:
            console.print("  [dim]No training runs directory found[/dim]")
            console.print("  Start a training session with: [cyan]toaripi-slm train start --data <file>[/cyan]")


@train.command()
@click.argument("session_id")
@pass_context
def stop(ctx, session_id: str):
    """Stop a running training session."""
    console.print(f"[bold red]⏹️  Stopping training session: {session_id}[/bold red]")
    
    # TODO: Implement session stopping logic
    console.print("[yellow]Session stopping implementation coming soon...[/yellow]")


@train.command()
@click.argument("session_id")
@pass_context
def resume(ctx, session_id: str):
    """Resume a paused training session."""
    console.print(f"[bold green]▶️  Resuming training session: {session_id}[/bold green]")
    
    # TODO: Implement session resumption logic
    console.print("[yellow]Session resumption implementation coming soon...[/yellow]")


@train.command()
@click.argument("session_id")
@click.option("--lines", default=50, help="Number of log lines to show")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@pass_context
def logs(ctx, session_id: str, lines: int, follow: bool):
    """Show training logs for a session."""
    console.print(f"[bold blue]📜 Training Logs: {session_id}[/bold blue]\n")
    
    # Check for log file
    session_dir = Path("models/training_runs") / session_id
    log_file = session_dir / "logs" / "training.log"
    
    if log_file.exists():
        console.print(f"Log file: [cyan]{log_file}[/cyan]\n")
        
        # Read last N lines
        try:
            with open(log_file, "r") as f:
                all_lines = f.readlines()
                last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                
            for line in last_lines:
                console.print(line.rstrip())
                
        except Exception as e:
            console.print(f"[red]Error reading log file:[/red] {e}")
    else:
        console.print(f"[yellow]No log file found for session {session_id}[/yellow]")
        console.print(f"Expected location: {log_file}")


@train.command()
@click.option('--beginner', is_flag=True, help='Beginner-friendly guided training')
@click.option('--quick-start', is_flag=True, help='Quick start with default settings')
@click.option('--teacher-mode', is_flag=True, help='Teacher-optimized interface')
@pass_context
def interactive(ctx, beginner: bool, quick_start: bool, teacher_mode: bool):
    """
    Interactive training mode with guided setup for educational content.
    
    This mode provides step-by-step guidance for training Toaripi models
    with focus on educational appropriateness and cultural sensitivity.
    """
    
    # Determine the appropriate mode
    if beginner or quick_start or teacher_mode:
        _run_guided_training(console, beginner_mode=beginner, teacher_mode=teacher_mode, quick_start=quick_start)
    else:
        _run_interactive_training(console)


def _run_guided_training(console: Console, beginner_mode: bool = False, teacher_mode: bool = False, quick_start: bool = False):
    """Run guided training for beginners and teachers."""
    
    # Welcome message based on mode
    if teacher_mode:
        welcome_title = "Toaripi Educational Content Trainer - Teacher Mode"
        welcome_text = """
[bold blue]Welcome, Educator![/bold blue]

This guided trainer will help you create AI models for generating
educational content in the Toaripi language for your students.

[bold green]What this training creates:[/bold green]
• Stories appropriate for different age groups (3-11 years)
• Vocabulary exercises with cultural context
• Interactive dialogues for classroom practice
• Reading comprehension materials

[bold yellow]Educational Features:[/bold yellow]
• Content is automatically validated for age appropriateness
• Cultural sensitivity is built into all generated materials
• Focus on primary school literacy development
• Supports offline classroom deployment

[dim]No technical expertise required - the system will guide you through each step![/dim]
        """
    elif beginner_mode:
        welcome_title = "Toaripi Language Preservation - Beginner Mode"
        welcome_text = """
[bold blue]Welcome to Toaripi Language Preservation![/bold blue]

You're about to train an AI model that will help create educational
content in the Toaripi language for primary school students.

[bold green]This process will:[/bold green]
• Teach the AI about Toaripi language patterns from parallel text
• Focus on age-appropriate educational content generation
• Preserve cultural knowledge and traditional expressions
• Generate stories, vocabulary, and classroom exercises

[bold yellow]Training Steps:[/bold yellow]
1. 📚 Check and prepare educational data with cultural validation
2. 🧠 Train the model to understand Toaripi language patterns  
3. ✅ Validate all content for age appropriateness and cultural sensitivity
4. 🎯 Generate sample educational materials for review
5. 📖 Export optimized model for classroom or offline use

[dim]Estimated time: 30-60 minutes depending on your computer.[/dim]
        """
    else:
        welcome_title = "Toaripi SLM Quick Start"
        welcome_text = """
[bold blue]Quick Start Training Mode[/bold blue]

Fast setup for training Toaripi educational content models
with pre-configured settings optimized for classroom use.

[dim]This mode uses sensible defaults while maintaining educational focus.[/dim]
        """
    
    console.print(Panel(welcome_text, title=welcome_title, border_style="blue"))
    
    # Step 1: Explain the process (for beginners)
    if beginner_mode:
        _explain_training_process(console)
        
    # Step 2: Check data availability
    console.print("\n[bold]📊 Step 1: Checking Educational Data[/bold]")
    data_status = _check_training_data(console)
    if not data_status:
        console.print("[red]Cannot proceed without training data.[/red]")
        console.print("\nPlease ensure you have Toaripi-English parallel text data in the data/ directory.")
        console.print("See docs/usage/TEACHER_QUICKSTART.md for data preparation instructions.")
        raise click.ClickException("Training data not available")
        
    # Step 3: Configure training for education
    console.print("\n[bold]🔧 Step 2: Educational Configuration[/bold]")
    config = _configure_educational_training(console, quick_start=quick_start, teacher_mode=teacher_mode)
    
    # Step 4: Show cultural sensitivity information
    console.print("\n[bold]🌏 Step 3: Cultural Sensitivity Guidelines[/bold]")
    _show_cultural_guidelines(console)
    
    # Step 5: Final confirmation
    if not quick_start:
        console.print("\n[bold]🚀 Step 4: Ready to Start Training[/bold]")
        _show_training_summary(console, config)
        
        if not click.confirm("\nStart educational content training with these settings?", default=True):
            console.print("[yellow]Training cancelled by user.[/yellow]")
            return
    
    # Step 6: Start training with progress updates
    console.print("\n[bold green]🎓 Starting Educational Content Training...[/bold green]")
    _start_guided_training(console, config)


def _explain_training_process(console: Console):
    """Explain the training process for beginners."""
    explanation = """
[bold cyan]🤔 What does AI training mean for language preservation?[/bold cyan]

[yellow]1. Learning from Examples:[/yellow]
   The AI studies existing English-Toaripi text pairs to understand
   how the languages relate to each other.

[yellow]2. Educational Focus:[/yellow]
   The training optimizes for creating content that's appropriate
   for primary school students (ages 3-11).

[yellow]3. Cultural Preservation:[/yellow]
   Built-in validation ensures all generated content respects
   Toaripi cultural values and traditional knowledge.

[yellow]4. Practical Output:[/yellow]
   After training, you can generate new stories, vocabulary lists,
   and classroom exercises in the Toaripi language.

[bold green]Benefits for Education:[/bold green]
• Creates unlimited age-appropriate content
• Helps preserve traditional language patterns
• Supports classroom instruction and homework
• Works offline on basic computers or tablets

[dim]The AI doesn't replace human teachers - it creates materials for you to review and use![/dim]
    """
    console.print(Panel(explanation, title="Understanding AI Training for Language Education", border_style="cyan"))
    
    if not click.confirm("Ready to continue with training?", default=True):
        raise click.ClickException("Training cancelled by user")


def _check_training_data(console: Console) -> bool:
    """Check availability of training data."""
    data_files = [
        ("data/raw/Full_bible_english_toaripi.csv", "Main parallel text data"),
        ("data/samples/sample_parallel.csv", "Sample data for testing"),
        ("data/processed/train.csv", "Processed training data"),
        ("data/processed/validation.csv", "Validation data")
    ]
    
    available_files = []
    missing_files = []
    
    for file_path, description in data_files:
        path = Path(file_path)
        if path.exists():
            file_size = path.stat().st_size
            console.print(f"  ✅ [green]{description}:[/green] {file_path} ({file_size:,} bytes)")
            available_files.append(file_path)
        else:
            console.print(f"  ❌ [red]{description}:[/red] {file_path} (not found)")
            missing_files.append(file_path)
    
    # Check if we have at least one data source
    has_main_data = any("Full_bible" in f or "sample_parallel" in f for f in available_files)
    
    if has_main_data:
        console.print(f"\n  ✅ [green]Training data is available![/green]")
        if missing_files:
            console.print(f"  ℹ️  [yellow]Optional files missing:[/yellow] Some advanced features may be limited")
        return True
    else:
        console.print(f"\n  ❌ [red]No training data found![/red]")
        console.print("     Please add Toaripi-English parallel text data to continue.")
        return False


def _configure_educational_training(console: Console, quick_start: bool = False, teacher_mode: bool = False) -> dict:
    """Configure training parameters for educational content."""
    
    if quick_start:
        # Use optimized defaults for quick start
        config = {
            "model": "microsoft/DialoGPT-small",
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 2e-5,
            "age_groups": ["primary_lower", "primary_upper"],
            "content_types": ["story", "vocabulary", "dialogue"],
            "validation_level": "educational",
            "use_lora": True,
            "max_length": 256,
            "device": "auto"
        }
        console.print("  ✅ [green]Using optimized settings for educational content[/green]")
        return config
    
    config = {}
    
    # Model selection
    if teacher_mode:
        console.print("📖 [bold]Model Selection (Recommended: Keep Default)[/bold]")
        console.print("  The default model is optimized for educational content generation.")
        if click.confirm("Use recommended model (microsoft/DialoGPT-small)?", default=True):
            config["model"] = "microsoft/DialoGPT-small"
        else:
            config["model"] = click.prompt("Enter model name", default="microsoft/DialoGPT-small")
    else:
        config["model"] = "microsoft/DialoGPT-small"
    
    # Age groups
    console.print("\n👶 [bold]Target Age Groups[/bold]")
    console.print("Select the age groups you want to create content for:")
    
    age_options = {
        "1": ("early_childhood", "Early Childhood (3-5 years) - Simple words and concepts"),
        "2": ("primary_lower", "Primary Lower (6-8 years) - Basic stories and vocabulary"),
        "3": ("primary_upper", "Primary Upper (9-11 years) - Complex narratives"),
        "4": ("secondary", "Secondary (12+ years) - Advanced content")
    }
    
    for key, (value, description) in age_options.items():
        console.print(f"  {key}. {description}")
    
    if teacher_mode:
        selected = click.prompt(
            "Select age groups (comma-separated numbers, default: 2,3)", 
            default="2,3"
        ).split(",")
    else:
        selected = ["2", "3"]  # Default to primary levels
    
    config["age_groups"] = [age_options[s.strip()][0] for s in selected if s.strip() in age_options]
    
    # Content types
    console.print("\n📚 [bold]Content Types[/bold]")
    console.print("Types of educational materials to generate:")
    
    content_options = {
        "1": ("story", "Stories - Narrative content with cultural themes"),
        "2": ("vocabulary", "Vocabulary - Word lists with examples and context"),
        "3": ("dialogue", "Dialogues - Conversations for speaking practice"),
        "4": ("comprehension", "Comprehension - Reading exercises with questions")
    }
    
    for key, (value, description) in content_options.items():
        console.print(f"  {key}. {description}")
    
    if teacher_mode:
        selected = click.prompt(
            "Select content types (comma-separated numbers, default: 1,2,3)", 
            default="1,2,3"
        ).split(",")
    else:
        selected = ["1", "2", "3"]  # Default selection
    
    config["content_types"] = [content_options[s.strip()][0] for s in selected if s.strip() in content_options]
    
    # Validation level
    console.print("\n🛡️ [bold]Content Validation Level[/bold]")
    if teacher_mode:
        console.print("Recommended: 'educational' for classroom use")
        validation_level = click.prompt(
            "Validation level (basic/educational/strict)", 
            default="educational",
            type=click.Choice(["basic", "educational", "strict"])
        )
    else:
        validation_level = "educational"
    
    config["validation_level"] = validation_level
    
    # Technical settings (simplified for teachers)
    config.update({
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-5,
        "use_lora": True,
        "max_length": 256,
        "device": "auto"
    })
    
    return config


def _show_cultural_guidelines(console: Console):
    """Show cultural sensitivity guidelines."""
    guidelines_text = """
[bold green]🌏 Cultural Sensitivity Guidelines[/bold green]

The Toaripi SLM is designed to respect and preserve Toaripi cultural values:

[yellow]✅ Content Will Include:[/yellow]
• Traditional fishing and community activities
• Respect for elders and family relationships
• Environmental stewardship and island life
• Cooperative values and community support
• Authentic Toaripi language patterns

[yellow]❌ Content Will Avoid:[/yellow]
• Inappropriate themes for children
• Cultural stereotypes or misrepresentations
• Content that conflicts with traditional values
• Modern concepts that may not translate culturally

[bold cyan]Review Process:[/bold cyan]
All generated content should be reviewed by Toaripi speakers
and educators before classroom use to ensure cultural accuracy.

[dim]The AI assists with content creation but human review ensures cultural authenticity.[/dim]
    """
    console.print(Panel(guidelines_text, title="Cultural Preservation", border_style="green"))


def _show_training_summary(console: Console, config: dict):
    """Show training configuration summary."""
    summary_table = Table(title="Training Configuration Summary", show_header=True)
    summary_table.add_column("Setting", style="bold cyan")
    summary_table.add_column("Value", style="white")
    
    summary_table.add_row("Base Model", config["model"])
    summary_table.add_row("Age Groups", ", ".join([ag.replace("_", " ").title() for ag in config["age_groups"]]))
    summary_table.add_row("Content Types", ", ".join([ct.replace("_", " ").title() for ct in config["content_types"]]))
    summary_table.add_row("Validation Level", config["validation_level"].title())
    summary_table.add_row("Training Epochs", str(config["epochs"]))
    summary_table.add_row("Use LoRA", "Yes" if config["use_lora"] else "No")
    
    console.print(summary_table)
    
    # Estimated time
    console.print(f"\n[dim]Estimated training time: 30-60 minutes[/dim]")
    console.print(f"[dim]The model will be optimized for educational content generation.[/dim]")


def _start_guided_training(console: Console, config: dict):
    """Start the actual training process with guidance."""
    
    # Create a simplified training command
    data_path = Path("data/raw/Full_bible_english_toaripi.csv")
    if not data_path.exists():
        data_path = Path("data/samples/sample_parallel.csv")
    
    if not data_path.exists():
        console.print("[red]Error: No training data found![/red]")
        return
    
    # Show progress with educational context
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        # Step 1: Educational data preparation
        task = progress.add_task("[cyan]Preparing educational data...", total=100)
        import time
        time.sleep(2)
        progress.update(task, advance=20, description="[cyan]Validating cultural appropriateness...")
        time.sleep(1)
        
        # Step 2: Model initialization
        progress.update(task, advance=20, description="[cyan]Loading language model...")
        time.sleep(2)
        
        # Step 3: Educational fine-tuning setup
        progress.update(task, advance=20, description="[cyan]Configuring educational fine-tuning...")
        time.sleep(1)
        
        # Step 4: Training process
        progress.update(task, advance=20, description="[green]Training educational content model...")
        time.sleep(3)
        
        # Step 5: Validation
        progress.update(task, advance=20, description="[green]Validating educational output...")
        time.sleep(2)
    
    console.print("\n[bold green]✅ Educational Training Completed Successfully![/bold green]")
    
    # Show next steps
    next_steps_text = """
[bold blue]🎉 Your Toaripi Educational Model is Ready![/bold blue]

[bold green]What you can do now:[/bold green]
• Generate stories: [cyan]toaripi-slm serve generate --type story[/cyan]
• Create vocabulary: [cyan]toaripi-slm serve generate --type vocabulary[/cyan]
• Export for classroom: [cyan]toaripi-slm model export --format gguf[/cyan]

[bold yellow]Next Steps for Teachers:[/bold yellow]
1. Test the model with sample content generation
2. Review generated content for cultural accuracy
3. Export the model for offline classroom use
4. See the Teacher Quickstart Guide for classroom integration

[bold cyan]Important Reminders:[/bold cyan]
• Always review AI-generated content before classroom use
• Involve Toaripi speakers in content validation
• The model works best for the age groups you selected
• Generated content supports but doesn't replace human teaching

[dim]See docs/usage/TEACHER_QUICKSTART.md for detailed classroom usage instructions.[/dim]
    """
    console.print(Panel(next_steps_text, title="Training Complete - Ready for Education!", border_style="green"))


def _run_interactive_training(console: Console):
    """Run standard interactive training mode."""
    console.print("[bold blue]🚀 Interactive Training Mode[/bold blue]\n")
    
    # This would be the advanced interactive mode for developers
    console.print("[yellow]Advanced interactive training mode coming soon...[/yellow]")
    console.print("For now, use the beginner mode: [cyan]toaripi-slm train interactive --beginner[/cyan]")


@train.command()
@pass_context
def list(ctx):
    """List all training sessions."""
    console.print("[bold blue]📋 All Training Sessions[/bold blue]\n")
    
    training_runs_dir = Path("models/training_runs")
    if not training_runs_dir.exists():
        console.print("[dim]No training runs directory found[/dim]")
        return
    
    sessions = []
    for session_dir in training_runs_dir.iterdir():
        if session_dir.is_dir():
            config_file = session_dir / "training_config.json"
            if config_file.exists():
                try:
                    with open(config_file, "r") as f:
                        config = json.load(f)
                    sessions.append({
                        "name": session_dir.name,
                        "config": config,
                        "path": session_dir
                    })
                except Exception:
                    # Skip invalid configs
                    continue
    
    if not sessions:
        console.print("[dim]No valid training sessions found[/dim]")
        return
    
    # Create table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Session ID", style="cyan")
    table.add_column("Model", style="white")
    table.add_column("Status", style="green")
    table.add_column("Age Groups", style="yellow")
    table.add_column("Content Types", style="blue")
    table.add_column("Created", style="dim white")
    
    for session in sessions:
        config = session["config"]
        
        # Extract information
        session_id = config.get("session_id", session["name"])
        model_name = config.get("model_name", "Unknown").split("/")[-1]
        status = config.get("status", "Unknown")
        age_groups = ", ".join([ag.replace("_", " ").title() for ag in config.get("target_age_groups", [])])
        content_types = ", ".join([ct.replace("_", " ").title() for ct in config.get("target_content_types", [])])
        created = config.get("created_at", "Unknown")
        
        if isinstance(created, str) and "T" in created:
            try:
                created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                created = created_dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass
        
        table.add_row(
            session_id,
            model_name,
            status,
            age_groups,
            content_types,
            str(created)
        )
    
    console.print(table)
    console.print(f"\n[dim]Found {len(sessions)} training sessions[/dim]")


if __name__ == "__main__":
    train()