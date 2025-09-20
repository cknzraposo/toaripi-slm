#!/usr/bin/env python3
"""
Setup command for Toaripi SLM CLI.

Provides guided setup and initialization for new users.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import click
from loguru import logger


def create_directory_structure():
    """Create required directory structure"""
    dirs = [
        "data/raw",
        "data/processed", 
        "data/samples",
        "configs/data",
        "configs/training",
        "models/cache",
        "models/hf",
        "models/gguf",
        "logs",
        "checkpoints",
    ]
    
    created = []
    for dir_path in dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(str(path))
    
    return created


def create_config_files():
    """Create default configuration files"""
    configs_created = []
    
    # Data processing config
    data_config = Path("configs/data/preprocessing_config.yaml")
    if not data_config.exists():
        data_config_content = """# Toaripi SLM Data Processing Configuration
input:
  data_dir: "data/raw"
  file_pattern: "*.txt"
  encoding: "utf-8"
  
processing:
  min_length: 10
  max_length: 512
  remove_duplicates: true
  normalize_unicode: true
  remove_non_text: true
  language_detection: true

output:
  format: "csv"
  encoding: "utf-8"
  columns: ["english", "toaripi", "verse_id", "book", "chapter"]
  
validation:
  test_split: 0.1
  dev_split: 0.05
  random_seed: 42
  stratify_by: "book"
"""
        data_config.write_text(data_config_content)
        configs_created.append(str(data_config))
    
    # Basic training config
    train_config = Path("configs/training/base_config.yaml")
    if not train_config.exists():
        train_config_content = """# Basic Toaripi SLM Training Configuration
model:
  name: "microsoft/DialoGPT-medium"
  cache_dir: "./models/cache"
  trust_remote_code: false

training:
  epochs: 3
  learning_rate: 2e-5
  batch_size: 4
  gradient_accumulation_steps: 4
  warmup_ratio: 0.1
  weight_decay: 0.001
  
  eval_strategy: "steps"
  eval_steps: 250
  save_strategy: "steps"
  save_steps: 250
  logging_steps: 50

data:
  max_length: 512
  padding: true
  truncation: true
  return_tensors: "pt"

optimization:
  optimizer: "adamw"
  lr_scheduler_type: "cosine"
  
output:
  output_dir: "./checkpoints"
  hub_model_id: ""

logging:
  use_wandb: false
  project_name: "toaripi-slm"
  run_name: "base-training"
  log_level: "INFO"
"""
        train_config.write_text(train_config_content)
        configs_created.append(str(train_config))
    
    # LoRA config for efficient training
    lora_config = Path("configs/training/lora_config.yaml")
    if not lora_config.exists():
        lora_config_content = """# LoRA Fine-tuning Configuration for Efficient Training
model:
  name: "mistralai/Mistral-7B-Instruct-v0.2"
  cache_dir: "./models/cache"
  trust_remote_code: false
  device_map: "auto"

lora:
  enabled: true
  rank: 16
  alpha: 32
  dropout: 0.1
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  bias: "none"
  task_type: "CAUSAL_LM"

training:
  epochs: 2
  learning_rate: 1e-4
  batch_size: 1
  gradient_accumulation_steps: 16
  warmup_ratio: 0.1
  weight_decay: 0.001
  
  eval_strategy: "steps"
  eval_steps: 250
  save_strategy: "steps"
  save_steps: 250
  logging_steps: 50
  
  gradient_checkpointing: true
  dataloader_pin_memory: false

data:
  max_length: 1024
  padding: true
  truncation: true
  return_tensors: "pt"

optimization:
  optimizer: "adamw"
  lr_scheduler_type: "cosine"
  
output:
  output_dir: "./checkpoints/lora"
  hub_model_id: ""

logging:
  use_wandb: false
  project_name: "toaripi-slm"
  run_name: "lora-training"
  log_level: "INFO"
"""
        lora_config.write_text(lora_config_content)
        configs_created.append(str(lora_config))
    
    return configs_created


def create_sample_data():
    """Create sample data files if they don't exist"""
    samples_created = []
    
    # Sample parallel data
    sample_data = Path("data/samples/sample_parallel.csv")
    if not sample_data.exists():
        sample_content = """english,toaripi,verse_id,book,chapter
"In the beginning God created the heavens and the earth.","Aio mao Gado aia kavo ma damuia aia.",Gen.1.1,Genesis,1
"The earth was formless and empty.","Damuia ai sioa ma hurua aia.",Gen.1.2,Genesis,1
"God said let there be light.","Gado aia: Mau maea aia.",Gen.1.3,Genesis,1
"""
        sample_data.write_text(sample_content)
        samples_created.append(str(sample_data))
    
    return samples_created


def check_dependencies() -> Dict[str, bool]:
    """Check for required dependencies"""
    deps = {}
    
    # Core ML libraries
    try:
        import torch
        deps['torch'] = True
    except ImportError:
        deps['torch'] = False
    
    try:
        import transformers
        deps['transformers'] = True
    except ImportError:
        deps['transformers'] = False
    
    try:
        import datasets
        deps['datasets'] = True
    except ImportError:
        deps['datasets'] = False
    
    try:
        import peft
        deps['peft'] = True
    except ImportError:
        deps['peft'] = False
    
    return deps


def install_missing_dependencies():
    """Install missing dependencies"""
    click.echo("🔄 Installing missing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "transformers", "datasets", "peft", "accelerate"
        ])
        click.echo("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Failed to install dependencies: {e}")
        return False


@click.command()
@click.option(
    '--guided', 
    is_flag=True, 
    help='Run guided setup with interactive prompts'
)
@click.option(
    '--force', 
    is_flag=True, 
    help='Force recreation of existing files'
)
@click.option(
    '--minimal', 
    is_flag=True, 
    help='Create minimal setup without sample data'
)
def setup(guided: bool, force: bool, minimal: bool):
    """
    Initialize Toaripi SLM project structure and configuration.
    
    This command sets up the required directories, configuration files,
    and sample data needed to get started with Toaripi SLM.
    
    \b
    Examples:
        toaripi setup --guided     # Interactive setup with guidance
        toaripi setup --minimal    # Quick setup without samples
        toaripi setup --force      # Recreate all files
    """
    
    if guided:
        click.echo("🌴 Welcome to Toaripi SLM Setup!\n")
        click.echo("This guided setup will help you get started with training and using")
        click.echo("Toaripi language models for educational content generation.\n")
        
        # Check if this is a clean directory
        if any(Path(p).exists() for p in ["data", "configs", "models"]):
            if not force:
                if not click.confirm("Existing project detected. Continue with setup?"):
                    click.echo("Setup cancelled.")
                    return
        
        # System requirements check
        click.echo("🔧 Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 10):
            click.echo(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} (compatible)")
        else:
            click.echo(f"❌ Python {python_version.major}.{python_version.minor}.{python_version.micro} (requires 3.10+)")
            if not click.confirm("Continue anyway?"):
                return
        
        # Check dependencies
        click.echo("\n📦 Checking dependencies...")
        deps = check_dependencies()
        missing_deps = [name for name, available in deps.items() if not available]
        
        if missing_deps:
            click.echo(f"⚠️  Missing dependencies: {', '.join(missing_deps)}")
            if click.confirm("Install missing dependencies?"):
                if not install_missing_dependencies():
                    click.echo("❌ Setup failed due to dependency issues")
                    return
        else:
            click.echo("✅ All dependencies are available")
        
        # Project structure
        click.echo("\n📁 Creating project structure...")
    
    # Create directories
    created_dirs = create_directory_structure()
    if created_dirs:
        click.echo(f"✅ Created directories: {', '.join(created_dirs)}")
    else:
        click.echo("ℹ️  Directory structure already exists")
    
    # Create config files
    created_configs = create_config_files()
    if created_configs:
        click.echo(f"✅ Created config files: {', '.join(created_configs)}")
    else:
        click.echo("ℹ️  Configuration files already exist")
    
    # Create sample data (unless minimal mode)
    if not minimal:
        created_samples = create_sample_data()
        if created_samples:
            click.echo(f"✅ Created sample files: {', '.join(created_samples)}")
    
    if guided:
        click.echo("\n🎉 Setup complete!")
        click.echo("\nNext steps:")
        click.echo("1. Place your Toaripi text data in data/raw/")
        click.echo("2. Process your data: toaripi prepare-data")
        click.echo("3. Start training: toaripi train --help")
        click.echo("4. Check system status: toaripi --status")
        click.echo("\n💡 Run 'toaripi --help' to see all available commands")
    else:
        click.echo("✅ Toaripi SLM project setup complete")


if __name__ == '__main__':
    setup()