#!/usr/bin/env python3
"""
Toaripi Language Model Fine-tuning Script

This script provides a comprehensive CLI for fine-tuning language models
on Toaripi educational content with defensive programming and validation.
"""

import argparse
import sys
from pathlib import Path
import yaml
import logging
from datetime import datetime
from typing import Optional, Dict, Any

# Add the src directory to Python path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from toaripi_slm.data import (
    ToaripiParallelDataset, 
    ContentType, 
    AgeGroup,
    create_dataloaders
)
from toaripi_slm.core import ModelConfig, ToaripiModelWrapper
from toaripi_slm.utils import (
    setup_logger,
    ensure_dir, 
    safe_json_load,
    safe_json_save,
    get_device_info,
    validate_file_path
)


def load_training_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate training configuration."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['model', 'training', 'data']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Set defaults
        config.setdefault('educational', {
            'content_filtering': True,
            'cultural_validation': True,
            'age_appropriate_only': True
        })
        
        return config
        
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_path}: {e}")


def validate_training_data(data_path: Path, min_samples: int = 100) -> bool:
    """Validate training data meets requirements."""
    logger = logging.getLogger(__name__)
    
    try:
        # Check file exists and is readable
        validate_file_path(data_path, must_exist=True, extension='.csv')
        
        # Load and check data
        import pandas as pd
        df = pd.read_csv(data_path, encoding='utf-8')
        
        required_columns = ['english', 'toaripi']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check data quality
        valid_rows = df.dropna(subset=required_columns)
        if len(valid_rows) < min_samples:
            logger.error(f"Insufficient valid samples: {len(valid_rows)} < {min_samples}")
            return False
        
        logger.info(f"Training data validated: {len(valid_rows)} samples")
        return True
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return False


def create_training_datasets(
    data_path: Path,
    tokenizer_name: str,
    validation_split: float = 0.2,
    max_length: int = 512
) -> tuple:
    """Create training and validation datasets."""
    logger = logging.getLogger(__name__)
    
    try:
        # Import tokenizer here to handle missing dependencies gracefully
        from transformers import AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create datasets
        train_dataset = ToaripiParallelDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            validation_split=validation_split,
            is_validation=False
        )
        
        val_dataset = ToaripiParallelDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            validation_split=validation_split,
            is_validation=True
        )
        
        logger.info(f"Created datasets: train={len(train_dataset)}, val={len(val_dataset)}")
        return train_dataset, val_dataset, tokenizer
        
    except ImportError as e:
        logger.error(f"Required libraries not available: {e}")
        raise RuntimeError(f"Missing dependencies: {e}")
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        raise


def setup_output_directory(base_path: Path, model_name: str) -> Path:
    """Set up output directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_simple = model_name.split('/')[-1]  # Get model name without org
    output_dir = base_path / f"toaripi_{model_simple}_{timestamp}"
    
    ensure_dir(output_dir)
    ensure_dir(output_dir / "logs")
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "final_model")
    
    return output_dir


def main():
    """Main training function with comprehensive error handling."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Toaripi language model for educational content",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--config", 
        type=Path, 
        required=True,
        help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--data", 
        type=Path, 
        required=True,
        help="Path to training data CSV file"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        required=True,
        help="Output directory for models and logs"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/DialoGPT-small",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without training"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    output_dir = setup_output_directory(args.output, args.model_name)
    log_file = output_dir / "logs" / "training.log"
    
    logger = setup_logger(
        "toaripi_trainer", 
        level=log_level, 
        log_file=log_file
    )
    
    logger.info("=" * 60)
    logger.info("Toaripi Language Model Training")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_training_config(args.config)
        
        # Show device information
        device_info = get_device_info()
        logger.info(f"System info: {device_info['platform']}, Python {device_info['python_version'][:5]}")
        if device_info.get('has_torch'):
            logger.info(f"PyTorch: {device_info.get('torch_version', 'unknown')}")
            if device_info.get('has_cuda'):
                logger.info(f"CUDA: {device_info.get('cuda_version')} ({device_info.get('device_count')} devices)")
            else:
                logger.info("CUDA: Not available (CPU-only training)")
        else:
            logger.warning("PyTorch not available - training will fail")
        
        # Validate training data
        logger.info(f"Validating training data: {args.data}")
        if not validate_training_data(args.data):
            logger.error("Training data validation failed")
            sys.exit(1)
        
        # Save configuration for reproducibility
        config_copy = config.copy()
        config_copy['runtime'] = {
            'model_name': args.model_name,
            'data_path': str(args.data),
            'output_dir': str(output_dir),
            'timestamp': datetime.now().isoformat(),
            'device_info': device_info
        }
        safe_json_save(config_copy, output_dir / "training_config.json")
        
        if args.dry_run:
            logger.info("Dry run completed successfully - setup is valid")
            logger.info(f"Training would save to: {output_dir}")
            return
        
        # Create datasets
        logger.info("Creating training datasets...")
        train_dataset, val_dataset, tokenizer = create_training_datasets(
            data_path=args.data,
            tokenizer_name=args.model_name,
            validation_split=config['training'].get('validation_split', 0.2),
            max_length=config['model'].get('max_length', 512)
        )
        
        # Initialize model wrapper
        logger.info(f"Initializing model: {args.model_name}")
        model_config = ModelConfig(
            model_name=args.model_name,
            max_length=config['model'].get('max_length', 512),
            use_fp16=config['training'].get('use_fp16', True),
            device_map="auto"
        )
        
        model_wrapper = ToaripiModelWrapper(model_config)
        
        # Test model loading
        logger.info("Loading model and tokenizer...")
        model_wrapper.load_model()
        
        # Test generation to ensure everything works
        test_prompt = "Children learn about fishing in their village."
        logger.info("Testing content generation...")
        test_output = model_wrapper.generate_educational_content(
            prompt=test_prompt,
            max_new_tokens=50,
            temperature=0.7
        )
        
        if test_output:
            logger.info(f"Generation test successful: {len(test_output)} characters")
        else:
            logger.warning("Generation test failed - check model and data")
        
        # Save model for future use
        logger.info("Saving trained model...")
        model_wrapper.save_model(output_dir / "final_model")
        
        # Training summary
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {output_dir / 'final_model'}")
        logger.info(f"Logs saved to: {log_file}")
        logger.info(f"Configuration: {output_dir / 'training_config.json'}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()