"""
Configuration and logging utilities for Toaripi SLM.
"""

import yaml
import toml
from pathlib import Path
from typing import Dict, Any, Union
import logging
from loguru import logger
import sys


class ConfigError(Exception):
    """Configuration-related errors."""
    pass


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or TOML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigError: If file not found or invalid format
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif config_path.suffix.lower() == '.toml':
                return toml.load(f)
            else:
                raise ConfigError(f"Unsupported config format: {config_path.suffix}")
    except Exception as e:
        raise ConfigError(f"Error loading config {config_path}: {e}")


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or TOML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif config_path.suffix.lower() == '.toml':
            toml.dump(config, f)
        else:
            raise ConfigError(f"Unsupported config format: {config_path.suffix}")


def setup_logging(level: str = "INFO", 
                 log_file: Union[str, Path, None] = None,
                 format_string: str = None) -> None:
    """
    Set up logging with loguru.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        format_string: Optional custom format string
    """
    # Remove default handler
    logger.remove()
    
    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation="10 MB",
            retention="1 week"
        )
    
    # Suppress some noisy loggers
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)
    logging.getLogger("transformers.utils.generic").setLevel(logging.WARNING)


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "setup.py").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root")


def validate_model_name(model_name: str) -> bool:
    """
    Validate that a model name is appropriate for educational content.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        True if model is suitable
    """
    # List of approved model families for educational content
    approved_patterns = [
        "microsoft/DialoGPT",
        "mistralai/Mistral", 
        "meta-llama/Llama-2",
        "google/flan-t5",
        "facebook/opt",
        "bigscience/bloom"
    ]
    
    # Check if model matches approved patterns
    for pattern in approved_patterns:
        if model_name.startswith(pattern):
            return True
    
    logger.warning(f"Model {model_name} not in approved list for educational content")
    return False


def ensure_directories(paths: Union[Path, str, list]) -> None:
    """
    Ensure directories exist, creating them if necessary.
    
    Args:
        paths: Single path or list of paths to create
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)