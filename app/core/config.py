"""
Configuration settings for Toaripi SLM Web Interface
"""

import os
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ]
    
    # File upload settings
    MAX_UPLOAD_SIZE: int = 52_428_800  # 50MB
    UPLOAD_DIR: Path = Path("data/uploads")
    ALLOWED_EXTENSIONS: List[str] = [".csv"]
    
    # Training settings
    MAX_CONCURRENT_TRAINING: int = 1
    TRAINING_TIMEOUT: int = 7200  # 2 hours
    MODEL_OUTPUT_DIR: Path = Path("models")
    
    # Model settings
    MAX_MODEL_SIZE_PARAMS: int = 7_000_000_000  # 7B parameters
    DEFAULT_BASE_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.2"
    QUANTIZATION_FORMAT: str = "q4_k_m"
    
    # Safety and validation
    MIN_TRAINING_PAIRS: int = 150
    SAFETY_THRESHOLD: float = 0.7
    MAX_TEXT_LENGTH: int = 300
    MIN_TEXT_LENGTH: int = 5
    
    # System resources
    MAX_MEMORY_GB: int = 8
    WARNING_MEMORY_THRESHOLD: float = 0.8
    WARNING_DISK_THRESHOLD: float = 0.9
    
    # Database settings (for future use)
    DATABASE_URL: str = "sqlite:///./toaripi_slm.db"
    
    # Cache settings
    CACHE_TTL: int = 300  # 5 minutes
    HEALTH_CHECK_INTERVAL: int = 60  # 1 minute
    
    # Constitutional compliance
    ENABLE_SAFETY_CHECKS: bool = True
    ENABLE_AGE_FILTER: bool = True
    ENABLE_CULTURAL_FILTER: bool = True
    BLOCK_THEOLOGICAL_CONTENT: bool = True
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()