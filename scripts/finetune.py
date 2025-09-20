#!/usr/bin/env python3
"""
Fine-tuning script for Toaripi SLM.

This script fine-tunes a pre-trained language model on Toaripi educational data.
"""

import click
import torch
from pathlib import Path
from loguru import logger

from toaripi_slm import ToaripiTrainer, ToaripiTrainingConfig, load_config, setup_logging


@click.command()
@click.option(
    '--data-dir',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Directory containing training data (train.csv, validation.csv)'
)
@click.option(
    '--model-name',
    type=str,
    default='microsoft/DialoGPT-medium',
    help='Name or path of pre-trained model to fine-tune'
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    required=True,
    help='Output directory for fine-tuned model'
)
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    default='configs/training/base_config.yaml',
    help='Path to training configuration file'
)
@click.option(
    '--use-lora',
    is_flag=True,
    default=True,
    help='Use LoRA for parameter-efficient fine-tuning'
)
@click.option(
    '--epochs',
    type=int,
    default=3,
    help='Number of training epochs'
)
@click.option(
    '--learning-rate',
    type=float,
    default=2e-5,
    help='Learning rate for training'
)
@click.option(
    '--batch-size',
    type=int,
    default=4,
    help='Training batch size'
)
@click.option(
    '--max-length',
    type=int,
    default=512,
    help='Maximum sequence length'
)
@click.option(
    '--resume-from',
    type=click.Path(exists=True, path_type=Path),
    help='Path to checkpoint to resume training from'
)
@click.option(
    '--wandb-project',
    type=str,
    help='Weights & Biases project name for logging'
)
@click.option(
    '--log-level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
    default='INFO',
    help='Logging level'
)
@click.option(
    '--log-file',
    type=click.Path(path_type=Path),
    help='Optional log file path'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Perform a dry run without actual training'
)
def finetune(
    data_dir: Path,
    model_name: str,
    output_dir: Path,
    config: Path,
    use_lora: bool,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    max_length: int,
    resume_from: Path,
    wandb_project: str,
    log_level: str,
    log_file: Path,
    dry_run: bool
):
    """Fine-tune a language model on Toaripi educational data."""
    
    # Setup logging
    setup_logging(level=log_level, log_file=str(log_file) if log_file else None)
    
    logger.info("Starting Toaripi SLM fine-tuning")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Use LoRA: {use_lora}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        logger.info(f"Current device: {torch.cuda.current_device()}")
    else:
        logger.warning("CUDA not available, training on CPU (will be slow)")
    
    try:
        # Load configuration
        if config.exists():
            config_data = load_config(str(config))
            logger.info(f"Loaded training config from {config}")
            
            # Flatten nested config to match dataclass fields
            training_config_dict = {}
            
            # Model settings
            if 'model' in config_data:
                model_config = config_data['model']
                training_config_dict['model_name'] = model_config.get('name', model_name)
                training_config_dict['model_cache_dir'] = model_config.get('cache_dir', './models/cache')
            
            # Training settings
            if 'training' in config_data:
                training_config = config_data['training']
                training_config_dict.update({
                    'epochs': training_config.get('epochs', 3),
                    'learning_rate': training_config.get('learning_rate', 2e-5),
                    'batch_size': training_config.get('batch_size', 4),
                    'gradient_accumulation_steps': training_config.get('gradient_accumulation_steps', 4),
                    'warmup_steps': training_config.get('warmup_steps', 100),
                    'weight_decay': training_config.get('weight_decay', 0.01),
                    'save_steps': training_config.get('save_steps', 500),
                    'eval_steps': training_config.get('eval_steps', 500),
                    'logging_steps': training_config.get('logging_steps', 100)
                })
            
            # Data settings
            if 'data' in config_data:
                data_config = config_data['data']
                training_config_dict['max_length'] = data_config.get('max_length', 512)
            
            # Output settings
            if 'output' in config_data:
                output_config = config_data['output']
                training_config_dict['save_total_limit'] = output_config.get('save_total_limit', 3)
            
            # Logging settings
            if 'logging' in config_data:
                logging_config = config_data['logging']
                training_config_dict['use_wandb'] = logging_config.get('use_wandb', False)
                training_config_dict['wandb_project'] = logging_config.get('project_name', 'toaripi-slm')
                training_config_dict['wandb_run_name'] = logging_config.get('run_name', 'training')
            
        else:
            logger.warning(f"Config file {config} not found, using defaults")
            training_config_dict = {}
        
        # Override config with command line arguments
        if use_lora is not None:
            training_config_dict['use_lora'] = use_lora
        if epochs is not None:
            training_config_dict['epochs'] = epochs
        if learning_rate is not None:
            training_config_dict['learning_rate'] = learning_rate
        if batch_size is not None:
            training_config_dict['batch_size'] = batch_size
        if max_length is not None:
            training_config_dict['max_length'] = max_length
        if wandb_project:
            training_config_dict['wandb_project'] = wandb_project
        
        # Set model name
        training_config_dict['model_name'] = model_name
        training_config_dict['output_dir'] = str(output_dir)
        
        # Create training configuration
        training_config = ToaripiTrainingConfig(**training_config_dict)
        
        logger.info("Training configuration:")
        logger.info(f"  Model: {training_config.model_name}")
        logger.info(f"  Epochs: {training_config.epochs}")
        logger.info(f"  Learning rate: {training_config.learning_rate}")
        logger.info(f"  Batch size: {training_config.batch_size}")
        logger.info(f"  Max length: {training_config.max_length}")
        logger.info(f"  Use LoRA: {training_config.use_lora}")
        if training_config.use_lora:
            logger.info(f"  LoRA rank: {training_config.lora_rank}")
            logger.info(f"  LoRA alpha: {training_config.lora_alpha}")
        
        if dry_run:
            logger.info("Dry run mode - configuration validated successfully")
            click.echo("✅ Dry run completed - configuration is valid")
            return
        
        # Check training data files
        train_file = data_dir / 'train.csv'
        val_file = data_dir / 'validation.csv'
        
        if not train_file.exists():
            raise click.ClickException(f"Training file not found: {train_file}")
        
        if not val_file.exists():
            logger.warning(f"Validation file not found: {val_file}")
            logger.warning("Training without validation data")
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = ToaripiTrainer(training_config)
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        trainer.load_model(model_name)
        
        # Prepare data
        logger.info("Preparing training data...")
        trainer.prepare_data(
            train_file=str(train_file),
            validation_file=str(val_file) if val_file.exists() else None
        )
        
        # Resume from checkpoint if specified
        if resume_from:
            logger.info(f"Resuming training from: {resume_from}")
            trainer.load_checkpoint(str(resume_from))
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving model to: {output_dir}")
        trainer.save_model(str(output_dir))
        
        logger.success("Fine-tuning completed successfully!")
        
        # Print summary
        click.echo("\n" + "="*50)
        click.echo("FINE-TUNING SUMMARY")
        click.echo("="*50)
        click.echo(f"Base model: {model_name}")
        click.echo(f"Training method: {'LoRA' if use_lora else 'Full fine-tuning'}")
        click.echo(f"Epochs: {epochs}")
        click.echo(f"Learning rate: {learning_rate}")
        click.echo(f"Batch size: {batch_size}")
        click.echo(f"Output directory: {output_dir}")
        click.echo("="*50)
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        raise click.ClickException(f"Fine-tuning failed: {e}")


if __name__ == '__main__':
    finetune()
    print("Toaripi fine-tuning script - not yet implemented")
    print("This will fine-tune models using LoRA on Toaripi data")

if __name__ == "__main__":
    main()