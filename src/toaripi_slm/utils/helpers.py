"""
Common utility functions for Toaripi SLM.

This module provides defensive utilities for file handling,
logging, and validation across the Toaripi SLM project.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
import sys


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path to ensure
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Set up logger with defensive configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        ensure_dir(log_file.parent)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_file_path(
    file_path: Union[str, Path],
    must_exist: bool = True,
    extension: Optional[str] = None
) -> Path:
    """
    Validate file path with defensive checks.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
        extension: Required file extension (e.g., '.csv')
        
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist and must_exist=True
        ValueError: If extension doesn't match
    """
    path = Path(file_path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if extension and path.suffix.lower() != extension.lower():
        raise ValueError(f"Expected {extension} file, got: {path.suffix}")
    
    return path


def safe_json_load(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Safely load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid
    """
    import json
    
    path = validate_file_path(file_path, must_exist=True, extension='.json')
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON from {path}: {e}")


def safe_json_save(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Safely save data as JSON with error handling.
    
    Args:
        data: Dictionary to save
        file_path: Output JSON file path
        
    Raises:
        ValueError: If data cannot be serialized
        RuntimeError: If save operation fails
    """
    import json
    
    path = Path(file_path)
    ensure_dir(path.parent)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot serialize data to JSON: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to save JSON to {path}: {e}")


def get_device_info() -> Dict[str, Any]:
    """
    Get system device information for model deployment.
    
    Returns:
        Dictionary with device information
    """
    device_info = {
        "platform": sys.platform,
        "python_version": sys.version,
        "has_cuda": False,
        "cuda_version": None,
        "device_count": 0,
        "recommended_device": "cpu"
    }
    
    try:
        import torch
        device_info["has_torch"] = True
        device_info["torch_version"] = torch.__version__
        
        if torch.cuda.is_available():
            device_info["has_cuda"] = True
            device_info["cuda_version"] = torch.version.cuda
            device_info["device_count"] = torch.cuda.device_count()
            device_info["recommended_device"] = "cuda"
        
    except ImportError:
        device_info["has_torch"] = False
    
    return device_info


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.1f} {units[unit_index]}"


def get_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    Calculate file hash for integrity checking.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (sha256, md5, sha1)
        
    Returns:
        Hexadecimal hash string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If algorithm is not supported
    """
    import hashlib
    
    path = validate_file_path(file_path, must_exist=True)
    
    # Validate algorithm
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    # Calculate hash
    hasher = hashlib.new(algorithm)
    
    try:
        with open(path, 'rb') as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    except Exception as e:
        raise RuntimeError(f"Failed to calculate hash for {path}: {e}")