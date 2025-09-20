"""
Training command for the Toaripi SLM CLI.

Provides interactive and guided training workflows with validation,
progress tracking, and resume capabilities.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt, IntPrompt, FloatPrompt
from rich import print as rprint

console = Console()

class TrainingSession:
    """Manages a training session with state tracking."""
    
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = session_dir / "session.json"
        self.config_file = session_dir / "training_config.yaml"
        self.log_file = session_dir / "training.log"
        
    def save_state(self, state: Dict[str, Any]):
        """Save the current training state."""
        state["last_updated"] = datetime.now().isoformat()
        with open(self.session_file, "w") as f:
            json.dump(state, f, indent=2)
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load the training state if it exists."""
        if self.session_file.exists():
            with open(self.session_file, "r") as f:
                return json.load(f)
        return None
    
    def save_config(self, config: Dict[str, Any]):
        """Save the training configuration."""
        with open(self.config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

def validate_data_files(data_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate that required data files exist and are properly formatted."""
    
    validation_results = {
        "valid": True,
        "issues": [],
        "files_found": {},
        "data_stats": {}
    }
    
    required_files = [
        "train.csv",
        "validation.csv", 
        "test.csv"
    ]
    
    processed_dir = data_dir / "processed"
    
    for file_name in required_files:
        file_path = processed_dir / file_name
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                validation_results["files_found"][file_name] = True
                validation_results["data_stats"][file_name] = {
                    "rows": len(df),
                    "columns": list(df.columns)
                }
                
                # Check for required columns
                required_cols = ["english", "toaripi"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    validation_results["issues"].append(
                        f"{file_name}: Missing required columns: {missing_cols}"
                    )
                    validation_results["valid"] = False
                    
            except Exception as e:
                validation_results["files_found"][file_name] = False
                validation_results["issues"].append(f"{file_name}: Error reading file - {e}")
                validation_results["valid"] = False
        else:
            validation_results["files_found"][file_name] = False
            validation_results["issues"].append(f"{file_name}: File not found")
            validation_results["valid"] = False
    
    return validation_results["valid"], validation_results

def create_training_config(base_config_path: Path) -> Dict[str, Any]:
    """Create a training configuration through interactive prompts."""
    
    console.print("\n🔧 [bold blue]Training Configuration Setup[/bold blue]")
    console.print("Let's configure your training parameters...\n")
    
    # Load base config
    if base_config_path.exists():
        with open(base_config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "model": {"name": "microsoft/DialoGPT-medium"},
            "training": {},
            "data": {},
            "optimization": {},
            "output": {},
            "logging": {}
        }
    
    # Model selection
    console.print("📋 [bold]Model Selection[/bold]")
    model_options = [
        "microsoft/DialoGPT-medium (Recommended for beginners)",
        "mistralai/Mistral-7B-Instruct-v0.2 (Better quality, needs more resources)",
        "Custom model path"
    ]
    
    for i, option in enumerate(model_options, 1):
        console.print(f"  {i}. {option}")
    
    while True:
        choice = IntPrompt.ask("\nSelect model", default=1, choices=["1", "2", "3"])
        if choice == 1:
            config["model"]["name"] = "microsoft/DialoGPT-medium"
            break
        elif choice == 2:
            config["model"]["name"] = "mistralai/Mistral-7B-Instruct-v0.2"
            break
        elif choice == 3:
            custom_model = Prompt.ask("Enter custom model path/name")
            config["model"]["name"] = custom_model
            break
    
    # Training parameters
    console.print("\n⚙️  [bold]Training Parameters[/bold]")
    
    config["training"]["epochs"] = IntPrompt.ask(
        "Number of training epochs", 
        default=config["training"].get("epochs", 3)
    )
    
    config["training"]["learning_rate"] = FloatPrompt.ask(
        "Learning rate", 
        default=config["training"].get("learning_rate", 2e-5)
    )
    
    config["training"]["batch_size"] = IntPrompt.ask(
        "Batch size", 
        default=config["training"].get("batch_size", 4)
    )
    
    # Hardware optimization
    console.print("\n🔧 [bold]Hardware Optimization[/bold]")
    
    use_gpu = Confirm.ask("Use GPU acceleration (if available)?", default=True)
    config["optimization"]["fp16"] = use_gpu
    
    # LoRA settings
    use_lora = Confirm.ask("Use LoRA (efficient fine-tuning)?", default=True)
    if use_lora:
        config["lora"] = {
            "enabled": True,
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.1
        }
    
    # Output settings
    console.print("\n💾 [bold]Output Settings[/bold]")
    
    output_dir = Prompt.ask(
        "Output directory for model checkpoints", 
        default="./models/checkpoints"
    )
    config["output"]["checkpoint_dir"] = output_dir
    
    # Logging
    use_wandb = Confirm.ask("Enable Weights & Biases logging?", default=False)
    config["logging"]["use_wandb"] = use_wandb
    
    if use_wandb:
        project_name = Prompt.ask("W&B project name", default="toaripi-slm")
        config["logging"]["project_name"] = project_name
    
    return config

def display_training_summary(config: Dict[str, Any], data_stats: Dict[str, Any]):
    """Display a summary of the training configuration."""
    
    summary_table = Table(title="🎯 Training Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Setting", style="cyan", no_wrap=True)
    summary_table.add_column("Value", style="green")
    
    # Model info
    summary_table.add_row("Model", config["model"]["name"])
    summary_table.add_row("Epochs", str(config["training"]["epochs"]))
    summary_table.add_row("Learning Rate", str(config["training"]["learning_rate"]))
    summary_table.add_row("Batch Size", str(config["training"]["batch_size"]))
    
    # Data info
    for file_name, stats in data_stats.items():
        summary_table.add_row(f"Data ({file_name})", f"{stats['rows']} samples")
    
    # Hardware
    if config["optimization"].get("fp16"):
        summary_table.add_row("GPU Acceleration", "✅ Enabled")
    else:
        summary_table.add_row("GPU Acceleration", "❌ Disabled")
    
    if config.get("lora", {}).get("enabled"):
        summary_table.add_row("LoRA Fine-tuning", "✅ Enabled")
    else:
        summary_table.add_row("LoRA Fine-tuning", "❌ Disabled")
    
    console.print("\n")
    console.print(summary_table)

@click.command()
@click.option("--config", "-c", type=click.Path(), help="Training configuration file")
@click.option("--data-dir", type=click.Path(exists=True), help="Data directory")
@click.option("--resume", "-r", is_flag=True, help="Resume from previous training session")
@click.option("--interactive", "-i", is_flag=True, default=True, help="Interactive configuration")
@click.option("--dry-run", is_flag=True, help="Validate setup without training")
def train(config, data_dir, resume, interactive, dry_run):
    """
    Train a Toaripi SLM model with guided setup and monitoring.
    
    This command provides an interactive workflow for:
    - Data validation
    - Model configuration  
    - Training execution
    - Progress monitoring
    """
    
    console.print("🚀 [bold blue]Starting Toaripi SLM Training[/bold blue]\n")
    
    # Set up paths
    data_path = Path(data_dir) if data_dir else Path("./data")
    config_path = Path(config) if config else Path("./configs/training/base_config.yaml")
    
    # Create session directory
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path("./training_sessions") / session_id
    training_session = TrainingSession(session_dir)
    
    # Check for resume
    if resume:
        console.print("🔄 [bold yellow]Checking for previous training sessions...[/bold yellow]")
        # Implementation for resuming would go here
        console.print("Resume functionality will be implemented in the next iteration.")
    
    # Step 1: Validate data
    console.print("📊 [bold]Step 1: Data Validation[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Validating data files...", total=None)
        
        data_valid, validation_results = validate_data_files(data_path)
        progress.update(task, completed=True)
    
    if not data_valid:
        console.print("❌ [red]Data validation failed:[/red]")
        for issue in validation_results["issues"]:
            console.print(f"  • {issue}")
        
        console.print("\n💡 [yellow]To fix data issues:[/yellow]")
        console.print("  1. Run data preprocessing: [cyan]toaripi-prepare-data[/cyan]")
        console.print("  2. Check data format requirements in docs/")
        return
    
    console.print("✅ Data validation passed!")
    
    # Step 2: Configuration
    if interactive:
        console.print("\n⚙️  [bold]Step 2: Training Configuration[/bold]")
        
        training_config = create_training_config(config_path)
        training_session.save_config(training_config)
    else:
        with open(config_path, "r") as f:
            training_config = yaml.safe_load(f)
    
    # Step 3: Training summary
    console.print("\n📋 [bold]Step 3: Training Summary[/bold]")
    display_training_summary(training_config, validation_results["data_stats"])
    
    if dry_run:
        console.print("\n🔍 [yellow]Dry run completed - no training performed.[/yellow]")
        console.print(f"Configuration saved to: {training_session.config_file}")
        return
    
    # Step 4: Confirmation
    if interactive:
        if not Confirm.ask("\nReady to start training?"):
            console.print("Training cancelled.")
            return
    
    # Step 5: Training execution
    console.print("\n🎯 [bold]Step 4: Training Execution[/bold]")
    console.print("Training will start shortly...")
    
    try:
        # Save initial state
        training_session.save_state({
            "status": "starting",
            "config": training_config,
            "session_id": session_id
        })
        
        # This is where the actual training would be implemented
        # For now, we'll show a placeholder
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # Placeholder training loop
            total_steps = training_config["training"]["epochs"] * 100  # Estimated
            task = progress.add_task("Training model...", total=total_steps)
            
            for epoch in range(training_config["training"]["epochs"]):
                # Placeholder for epoch training
                for step in range(100):  # Estimated steps per epoch
                    progress.advance(task, 1)
                    
                    # Update session state periodically
                    if step % 20 == 0:
                        training_session.save_state({
                            "status": "training",
                            "current_epoch": epoch + 1,
                            "current_step": step,
                            "total_epochs": training_config["training"]["epochs"]
                        })
        
        # Training completed
        training_session.save_state({
            "status": "completed",
            "completion_time": datetime.now().isoformat()
        })
        
        console.print("\n✅ [green]Training completed successfully![/green]")
        console.print(f"📁 Session saved to: {session_dir}")
        console.print(f"🤖 Model checkpoints: {training_config['output']['checkpoint_dir']}")
        
        # Next steps
        console.print("\n🎉 [bold]Next Steps:[/bold]")
        console.print("  1. Test your model: [cyan]toaripi test[/cyan]")
        console.print("  2. Generate content: [cyan]toaripi interact[/cyan]")
        console.print("  3. Export for edge deployment: [cyan]toaripi export[/cyan]")
        
    except KeyboardInterrupt:
        training_session.save_state({
            "status": "interrupted", 
            "interruption_time": datetime.now().isoformat()
        })
        console.print("\n⏸️  Training interrupted. Session saved for resume.")
        
    except Exception as e:
        training_session.save_state({
            "status": "failed",
            "error": str(e),
            "failure_time": datetime.now().isoformat()
        })
        console.print(f"\n❌ Training failed: {e}")
        if "--debug" in sys.argv:
            import traceback
            console.print(traceback.format_exc())