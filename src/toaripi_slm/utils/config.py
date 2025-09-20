"""
Configuration utilities for Toaripi SLM.

This module provides functions for loading and managing configuration files,
setting up logging, and other utility functions.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from loguru import logger
import sys


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is unsupported
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    suffix = config_file.suffix.lower()
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            if suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {suffix}")
        
        logger.info(f"Loaded configuration from: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = config_file.suffix.lower()
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            if suffix in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif suffix == '.json':
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported config file format: {suffix}")
        
        logger.info(f"Saved configuration to: {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        raise


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "30 days"
) -> None:
    """
    Setup logging configuration for Toaripi SLM.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional path to log file
        format_string: Custom log format string
        rotation: Log file rotation size
        retention: Log file retention period
    """
    # Remove default handler
    logger.remove()
    
    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=level,
        format=format_string,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=level,
            format=format_string,
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        logger.info(f"Logging to file: {log_file}")
    
    logger.info(f"Logging setup complete (level: {level})")


def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    Get information about a model directory.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Dictionary with model information
    """
    model_dir = Path(model_path)
    
    info = {
        'path': str(model_dir),
        'exists': model_dir.exists(),
        'is_directory': model_dir.is_dir() if model_dir.exists() else False,
        'files': [],
        'size_mb': 0,
        'model_type': 'unknown'
    }
    
    if not model_dir.exists():
        return info
    
    # List files
    if model_dir.is_dir():
        info['files'] = [f.name for f in model_dir.iterdir() if f.is_file()]
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        info['size_mb'] = total_size / (1024 * 1024)
        
        # Determine model type
        if 'adapter_config.json' in info['files']:
            info['model_type'] = 'lora_adapter'
        elif 'config.json' in info['files']:
            info['model_type'] = 'huggingface'
        elif any(f.endswith('.gguf') for f in info['files']):
            info['model_type'] = 'gguf'
        elif 'pytorch_model.bin' in info['files'] or any(f.startswith('model') for f in info['files']):
            info['model_type'] = 'pytorch'
    
    return info


def validate_environment() -> Dict[str, bool]:
    """
    Validate the environment setup for Toaripi SLM.
    
    Returns:
        Dictionary of validation results
    """
    checks = {}
    
    # Check Python version
    import sys
    checks['python_version'] = sys.version_info >= (3, 10)
    
    # Check required packages
    required_packages = [
        'torch', 'transformers', 'datasets', 'accelerate', 
        'peft', 'pandas', 'numpy', 'yaml', 'loguru'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            checks[f'package_{package}'] = True
        except ImportError:
            checks[f'package_{package}'] = False
    
    # Check CUDA availability
    try:
        import torch
        checks['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            checks['cuda_device_count'] = torch.cuda.device_count()
        else:
            checks['cuda_device_count'] = 0
    except:
        checks['cuda_available'] = False
        checks['cuda_device_count'] = 0
    
    # Check disk space (rough estimate)
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        checks['disk_space_gb'] = free / (1024**3)
        checks['sufficient_disk_space'] = free > 10 * (1024**3)  # 10GB minimum
    except:
        checks['sufficient_disk_space'] = None
    
    return checks


def create_directory_structure(base_path: str) -> None:
    """
    Create the standard Toaripi SLM directory structure.
    
    Args:
        base_path: Base directory path
    """
    base = Path(base_path)
    
    directories = [
        'data/raw',
        'data/processed',
        'data/samples',
        'models/hf',
        'models/gguf',
        'checkpoints',
        'logs',
        'configs/data',
        'configs/training',
        'notebooks',
        'tests/unit',
        'tests/integration',
        'scripts'
    ]
    
    for dir_path in directories:
        full_path = base / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directory structure in: {base_path}")


def get_available_models(models_dir: str = "models") -> Dict[str, Dict[str, Any]]:
    """
    Get information about available models in the models directory.
    
    Args:
        models_dir: Path to models directory
        
    Returns:
        Dictionary of model information
    """
    models_path = Path(models_dir)
    available_models = {}
    
    if not models_path.exists():
        return available_models
    
    # Check HuggingFace format models
    hf_dir = models_path / 'hf'
    if hf_dir.exists():
        for model_dir in hf_dir.iterdir():
            if model_dir.is_dir():
                available_models[f"hf/{model_dir.name}"] = get_model_info(str(model_dir))
    
    # Check GGUF format models
    gguf_dir = models_path / 'gguf'
    if gguf_dir.exists():
        for gguf_file in gguf_dir.glob('*.gguf'):
            available_models[f"gguf/{gguf_file.name}"] = {
                'path': str(gguf_file),
                'size_mb': gguf_file.stat().st_size / (1024 * 1024),
                'model_type': 'gguf'
            }
    
    return available_models


def format_size(size_bytes: int) -> str:
    """
    Format byte size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    
    for config in configs:
        if config:
            merged.update(config)
    
    return merged


def resolve_config_path(config_name: str, config_dirs: Optional[list] = None) -> Optional[str]:
    """
    Resolve configuration file path by searching in multiple directories.
    
    Args:
        config_name: Name of config file (with or without extension)
        config_dirs: List of directories to search in
        
    Returns:
        Resolved path or None if not found
    """
    if config_dirs is None:
        config_dirs = ['configs', 'configs/training', 'configs/data', '.']
    
    # Add extensions if not present
    config_names = [config_name]
    if not any(config_name.endswith(ext) for ext in ['.yaml', '.yml', '.json']):
        config_names.extend([f"{config_name}.yaml", f"{config_name}.yml", f"{config_name}.json"])
    
    # Search in directories
    for directory in config_dirs:
        dir_path = Path(directory)
        if dir_path.exists():
            for name in config_names:
                config_path = dir_path / name
                if config_path.exists():
                    return str(config_path)
    
    return None


def setup_wandb_env(project_name: str = "toaripi-slm") -> bool:
    """
    Setup Weights & Biases environment if API key is available.
    
    Args:
        project_name: W&B project name
        
    Returns:
        True if W&B is available and configured
    """
    try:
        import wandb
        
        # Check if API key is available
        api_key = os.getenv('WANDB_API_KEY')
        if not api_key:
            logger.info("WANDB_API_KEY not found. W&B logging disabled.")
            return False
        
        # Test connection
        wandb.login(key=api_key)
        logger.info(f"W&B configured for project: {project_name}")
        return True
        
    except ImportError:
        logger.warning("wandb package not installed. Install with: pip install wandb")
        return False
    except Exception as e:
        logger.warning(f"Failed to setup W&B: {e}")
        return False


# Default configuration templates
DEFAULT_TRAINING_CONFIG = {
    'model': {
        'name': 'microsoft/DialoGPT-medium',
        'cache_dir': './models/cache'
    },
    'training': {
        'epochs': 3,
        'learning_rate': 2e-5,
        'batch_size': 4,
        'gradient_accumulation_steps': 4,
        'use_lora': True,
        'lora_rank': 16,
        'lora_alpha': 32
    },
    'data': {
        'max_length': 512,
        'validation_split': 0.1
    },
    'output': {
        'checkpoint_dir': './checkpoints',
        'save_steps': 500
    }
}

DEFAULT_DATA_CONFIG = {
    'preprocessing': {
        'min_length': 10,
        'max_length': 512,
        'remove_duplicates': True,
        'normalize_unicode': True
    },
    'output': {
        'format': 'csv',
        'encoding': 'utf-8'
    },
    'validation': {
        'test_split': 0.1,
        'dev_split': 0.05,
        'random_seed': 42
    }
}