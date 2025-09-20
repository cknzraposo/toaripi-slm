#!/usr/bin/env python3
"""
Training command for Toaripi SLM CLI.

Provides guided training experience with comprehensive validation and monitoring.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import pandas as pd
from loguru import logger


def estimate_training_time(
    model_name: str, 
    dataset_size: int, 
    batch_size: int, 
    epochs: int,
    use_lora: bool = True
) -> Dict[str, str]:
    """Estimate training time and resources"""
    
    # Base estimates (very rough, for guidance only)
    samples_per_second = {
        "microsoft/DialoGPT-medium": 10 if not use_lora else 20,
        "mistralai/Mistral-7B-Instruct-v0.2": 2 if not use_lora else 8,
        "default": 5 if not use_lora else 12
    }
    
    rate = samples_per_second.get(model_name, samples_per_second["default"])
    total_samples = dataset_size * epochs
    estimated_seconds = total_samples / rate
    
    # Convert to human readable
    if estimated_seconds < 60:
        time_str = f"{estimated_seconds:.0f} seconds"
    elif estimated_seconds < 3600:
        time_str = f"{estimated_seconds/60:.1f} minutes"
    else:
        time_str = f"{estimated_seconds/3600:.1f} hours"
    
    # Memory estimates (GB)
    memory_estimates = {
        "microsoft/DialoGPT-medium": 4 if not use_lora else 2,
        "mistralai/Mistral-7B-Instruct-v0.2": 16 if not use_lora else 8,
        "default": 8 if not use_lora else 4
    }
    
    memory_gb = memory_estimates.get(model_name, memory_estimates["default"])
    
    return {
        "time": time_str,
        "memory_gb": memory_gb,
        "samples_per_second": rate,
        "total_samples": total_samples
    }


def validate_training_data(data_dir: Path) -> Tuple[bool, List[str], Dict[str, int]]:
    """Validate training data and return status, issues, and stats"""
    issues = []
    stats = {}
    
    # Check required files
    required_files = ["train.csv", "validation.csv"]
    for file in required_files:
        file_path = data_dir / file
        if not file_path.exists():
            issues.append(f"Missing required file: {file}")
            return False, issues, stats
    
    try:
        # Load and validate training data
        train_df = pd.read_csv(data_dir / "train.csv")
        val_df = pd.read_csv(data_dir / "validation.csv")
        
        # Check required columns
        required_cols = ["english", "toaripi"]
        for df_name, df in [("train", train_df), ("validation", val_df)]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                issues.append(f"Missing columns in {df_name}.csv: {missing_cols}")
        
        # Check for empty data
        if len(train_df) == 0:
            issues.append("Training data is empty")
        if len(val_df) == 0:
            issues.append("Validation data is empty")
        
        # Calculate statistics
        stats = {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "total_samples": len(train_df) + len(val_df),
            "avg_english_length": train_df["english"].str.len().mean() if "english" in train_df.columns else 0,
            "avg_toaripi_length": train_df["toaripi"].str.len().mean() if "toaripi" in train_df.columns else 0,
        }
        
        # Check data quality
        if stats["train_samples"] < 100:
            issues.append(f"Very small training set ({stats['train_samples']} samples). Consider adding more data.")
        
        if stats["val_samples"] < 20:
            issues.append(f"Very small validation set ({stats['val_samples']} samples).")
        
    except Exception as e:
        issues.append(f"Error reading data files: {e}")
        return False, issues, stats
    
    return len(issues) == 0, issues, stats


def check_gpu_memory() -> Optional[float]:
    """Check available GPU memory in GB"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            return device.total_memory / (1024**3)  # Convert to GB
    except ImportError:
        pass
    return None


def print_training_summary(
    model_name: str,
    data_dir: Path,
    config_path: Path,
    use_lora: bool,
    epochs: int,
    stats: Dict[str, int],
    estimates: Dict[str, str]
):
    """Print a comprehensive training summary"""
    
    click.echo("\n" + "="*60)
    click.echo("📊 TRAINING SUMMARY")
    click.echo("="*60)
    
    click.echo(f"🤖 Model: {model_name}")
    click.echo(f"📁 Data: {data_dir}")
    click.echo(f"⚙️  Config: {config_path}")
    click.echo(f"🔧 Method: {'LoRA Fine-tuning' if use_lora else 'Full Fine-tuning'}")
    click.echo(f"🔄 Epochs: {epochs}")
    
    click.echo(f"\n📈 Dataset:")
    click.echo(f"   Training samples: {stats.get('train_samples', 0):,}")
    click.echo(f"   Validation samples: {stats.get('val_samples', 0):,}")
    click.echo(f"   Total samples: {stats.get('total_samples', 0):,}")
    
    click.echo(f"\n⏱️  Estimates:")
    click.echo(f"   Training time: {estimates['time']}")
    click.echo(f"   Memory needed: ~{estimates['memory_gb']}GB")
    click.echo(f"   Processing rate: ~{estimates['samples_per_second']} samples/sec")
    
    # GPU info
    gpu_memory = check_gpu_memory()
    if gpu_memory:
        status = "✅" if gpu_memory >= estimates['memory_gb'] else "⚠️"
        click.echo(f"   GPU memory: {gpu_memory:.1f}GB available {status}")
    else:
        click.echo(f"   GPU memory: Not detected (CPU training)")


@click.command()
@click.option(
    '--data-dir',
    type=click.Path(exists=True, path_type=Path),
    help='Directory containing training data (train.csv, validation.csv)'
)
@click.option(
    '--model-name',
    type=str,
    help='Name or path of pre-trained model to fine-tune'
)
@click.option(
    '--config',
    type=click.Path(path_type=Path),
    help='Path to training configuration file'
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    help='Output directory for fine-tuned model'
)
@click.option(
    '--use-lora/--no-lora',
    default=None,
    help='Use LoRA for parameter-efficient fine-tuning'
)
@click.option(
    '--epochs',
    type=int,
    help='Number of training epochs'
)
@click.option(
    '--batch-size',
    type=int,
    help='Training batch size'
)
@click.option(
    '--learning-rate',
    type=float,
    help='Learning rate for training'
)
@click.option(
    '--resume-from',
    type=click.Path(exists=True, path_type=Path),
    help='Path to checkpoint to resume training from'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Perform a dry run without actual training'
)
@click.option(
    '--guided',
    is_flag=True,
    help='Run in guided mode with interactive prompts'
)
@click.option(
    '--wandb-project',
    type=str,
    help='Weights & Biases project name for logging'
)
def train(
    data_dir: Optional[Path],
    model_name: Optional[str],
    config: Optional[Path],
    output_dir: Optional[Path],
    use_lora: Optional[bool],
    epochs: Optional[int],
    batch_size: Optional[int],
    learning_rate: Optional[float],
    resume_from: Optional[Path],
    dry_run: bool,
    guided: bool,
    wandb_project: Optional[str]
):
    """
    Train a Toaripi language model on educational data.
    
    This command provides a comprehensive training pipeline with validation,
    progress monitoring, and helpful guidance for new users.
    
    \b
    Examples:
        toaripi train --guided                    # Interactive guided training
        toaripi train --data data/processed       # Quick training with defaults
        toaripi train --config configs/training/lora_config.yaml
        toaripi train --dry-run                   # Validate setup without training
    """
    
    if guided or not all([data_dir, model_name]):
        click.echo("🚀 Welcome to Toaripi SLM Training!\n")
        
        # Data directory selection
        if not data_dir:
            click.echo("📁 First, let's locate your training data.")
            
            # Check common locations
            common_locations = [
                Path("data/processed"),
                Path("data/samples"),
                Path("data"),
            ]
            
            for loc in common_locations:
                if loc.exists() and (loc / "train.csv").exists():
                    if click.confirm(f"Use training data from {loc}?"):
                        data_dir = loc
                        break
            
            if not data_dir:
                data_dir_str = click.prompt(
                    "Enter path to your training data directory",
                    default="data/processed"
                )
                data_dir = Path(data_dir_str)
                
                if not data_dir.exists():
                    click.echo(f"❌ Directory {data_dir} does not exist")
                    click.echo("💡 Tip: Run 'toaripi setup' to create project structure")
                    return
        
        # Validate data
        click.echo(f"\n🔍 Validating training data in {data_dir}...")
        valid, issues, stats = validate_training_data(data_dir)
        
        if not valid:
            click.echo("❌ Data validation failed:")
            for issue in issues:
                click.echo(f"   • {issue}")
            click.echo("\n💡 Tips:")
            click.echo("   • Ensure train.csv and validation.csv exist")
            click.echo("   • Check that files have 'english' and 'toaripi' columns")
            click.echo("   • Run 'toaripi prepare-data' to process raw data")
            return
        else:
            click.echo("✅ Data validation passed")
            if issues:  # Warnings
                click.echo("⚠️  Warnings:")
                for issue in issues:
                    click.echo(f"   • {issue}")
        
        # Model selection
        if not model_name:
            click.echo("\n🤖 Now let's choose a model to fine-tune.")
            
            model_options = [
                ("microsoft/DialoGPT-medium", "Small, fast, good for development (355M params)"),
                ("mistralai/Mistral-7B-Instruct-v0.2", "Larger, better quality (7B params)"),
                ("custom", "Enter custom model name")
            ]
            
            click.echo("Available models:")
            for i, (name, desc) in enumerate(model_options, 1):
                click.echo(f"  {i}. {name}")
                click.echo(f"     {desc}")
            
            choice = click.prompt(
                "\nSelect model (1-3)",
                type=click.IntRange(1, len(model_options))
            )
            
            if choice == len(model_options):  # Custom
                model_name = click.prompt("Enter model name")
            else:
                model_name = model_options[choice - 1][0]
        
        # LoRA decision
        if use_lora is None:
            click.echo(f"\n⚡ Training method for {model_name}:")
            click.echo("LoRA (Low-Rank Adaptation) is recommended for:")
            click.echo("  • Faster training and less memory usage")
            click.echo("  • Better performance on small datasets")
            click.echo("  • Easier deployment and sharing")
            
            use_lora = click.confirm("Use LoRA fine-tuning?", default=True)
        
        # Configuration
        if not config:
            if use_lora:
                default_config = Path("configs/training/lora_config.yaml")
            else:
                default_config = Path("configs/training/base_config.yaml")
            
            if default_config.exists():
                if click.confirm(f"Use configuration from {default_config}?", default=True):
                    config = default_config
            
            if not config:
                config_str = click.prompt(
                    "Enter path to configuration file",
                    default=str(default_config)
                )
                config = Path(config_str)
        
        # Output directory
        if not output_dir:
            suggested_output = Path("checkpoints") / f"toaripi-{model_name.split('/')[-1]}"
            if use_lora:
                suggested_output = suggested_output.with_name(suggested_output.name + "-lora")
            
            output_dir_str = click.prompt(
                "Enter output directory for trained model",
                default=str(suggested_output)
            )
            output_dir = Path(output_dir_str)
        
        # Training parameters
        if not epochs:
            epochs = click.prompt("Number of training epochs", default=3, type=int)
        
        # Show estimates
        estimates = estimate_training_time(
            model_name, stats.get("total_samples", 1000), 
            batch_size or 4, epochs, use_lora
        )
        
        print_training_summary(
            model_name, data_dir, config, use_lora, epochs, stats, estimates
        )
        
        if not click.confirm("\n🚀 Start training with these settings?"):
            click.echo("Training cancelled.")
            return
    
    else:
        # Non-guided mode: validate required parameters
        if not data_dir:
            click.echo("❌ Error: --data-dir is required")
            return
        
        if not model_name:
            click.echo("❌ Error: --model-name is required")
            return
        
        # Quick validation
        valid, issues, stats = validate_training_data(data_dir)
        if not valid:
            click.echo("❌ Data validation failed:")
            for issue in issues:
                click.echo(f"   • {issue}")
            return
    
    # Set defaults
    if config is None:
        config = Path("configs/training/lora_config.yaml" if use_lora else "configs/training/base_config.yaml")
    
    if output_dir is None:
        output_dir = Path("checkpoints") / f"toaripi-{model_name.split('/')[-1]}"
        if use_lora:
            output_dir = output_dir.with_name(output_dir.name + "-lora")
    
    if epochs is None:
        epochs = 3
    
    # Dry run mode
    if dry_run:
        click.echo("\n🧪 DRY RUN MODE - No actual training will be performed")
        click.echo("✅ All validations passed. Training would proceed with these settings.")
        return
    
    # Import and run the actual training
    click.echo(f"\n🚀 Starting training...")
    click.echo(f"📊 Progress will be logged to: {output_dir / 'logs'}")
    
    try:
        # Import the training script functionality
        from ....scripts.finetune import main as finetune_main
        
        # Call the existing training function with our parameters
        # This integrates with the existing finetune.py script
        import sys
        sys.argv = [
            "finetune",
            "--data-dir", str(data_dir),
            "--model-name", model_name,
            "--output-dir", str(output_dir),
            "--config", str(config),
            "--epochs", str(epochs),
        ]
        
        if use_lora:
            sys.argv.append("--use-lora")
        
        if wandb_project:
            sys.argv.extend(["--wandb-project", wandb_project])
        
        # Run training
        finetune_main()
        
    except ImportError:
        click.echo("❌ Training module not found. Please ensure all dependencies are installed.")
        click.echo("💡 Try: pip install -e .")
    except KeyboardInterrupt:
        click.echo("\n⏹️  Training interrupted by user")
    except Exception as e:
        click.echo(f"❌ Training failed: {e}")
        click.echo("💡 Check logs for more details")
        logger.exception("Training error")


if __name__ == '__main__':
    train()