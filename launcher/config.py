"""
Configuration management for Toaripi SLM launcher.

This module handles launcher settings, educational parameters, and user preferences
with focus on educational content generation and cultural sensitivity.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum


class AgeGroup(Enum):
    """Age groups for educational content targeting."""
    EARLY_CHILDHOOD = "early_childhood"  # 3-5 years
    PRIMARY_LOWER = "primary_lower"      # 6-8 years
    PRIMARY_upper = "primary_upper"      # 9-11 years
    SECONDARY = "secondary"              # 12+ years


class ContentType(Enum):
    """Types of educational content to generate."""
    STORY = "story"
    VOCABULARY = "vocabulary"
    DIALOGUE = "dialogue"
    COMPREHENSION = "comprehension"
    EXERCISE = "exercise"


class ValidationLevel(Enum):
    """Content validation strictness levels."""
    BASIC = "basic"          # Basic language checks
    EDUCATIONAL = "educational"  # Educational appropriateness
    STRICT = "strict"        # Full cultural validation


@dataclass
class EducationalConfig:
    """Educational content configuration."""
    age_groups: List[AgeGroup] = field(default_factory=lambda: [AgeGroup.PRIMARY_LOWER, AgeGroup.PRIMARY_upper])
    content_types: List[ContentType] = field(default_factory=lambda: [ContentType.STORY, ContentType.VOCABULARY])
    validation_level: ValidationLevel = ValidationLevel.EDUCATIONAL
    cultural_sensitivity: bool = True
    max_content_length: int = 256
    language_preservation_mode: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "age_groups": [ag.value for ag in self.age_groups],
            "content_types": [ct.value for ct in self.content_types],
            "validation_level": self.validation_level.value,
            "cultural_sensitivity": self.cultural_sensitivity,
            "max_content_length": self.max_content_length,
            "language_preservation_mode": self.language_preservation_mode
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EducationalConfig':
        """Create from dictionary."""
        return cls(
            age_groups=[AgeGroup(ag) for ag in data.get("age_groups", ["primary_lower", "primary_upper"])],
            content_types=[ContentType(ct) for ct in data.get("content_types", ["story", "vocabulary"])],
            validation_level=ValidationLevel(data.get("validation_level", "educational")),
            cultural_sensitivity=data.get("cultural_sensitivity", True),
            max_content_length=data.get("max_content_length", 256),
            language_preservation_mode=data.get("language_preservation_mode", True)
        )


@dataclass
class TrainingConfig:
    """Training configuration for model fine-tuning."""
    model_name: str = "microsoft/DialoGPT-small"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    max_length: int = 256
    device: str = "auto"
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "max_length": self.max_length,
            "device": self.device,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class SystemConfig:
    """System and environment configuration."""
    python_version_min: str = "3.10"
    virtual_env_required: bool = True
    auto_install_dependencies: bool = True
    auto_create_venv: bool = True
    check_gpu_availability: bool = True
    memory_limit_gb: Optional[int] = None
    disk_space_required_gb: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "python_version_min": self.python_version_min,
            "virtual_env_required": self.virtual_env_required,
            "auto_install_dependencies": self.auto_install_dependencies,
            "auto_create_venv": self.auto_create_venv,
            "check_gpu_availability": self.check_gpu_availability,
            "memory_limit_gb": self.memory_limit_gb,
            "disk_space_required_gb": self.disk_space_required_gb
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class UIConfig:
    """User interface configuration."""
    show_welcome_message: bool = True
    use_rich_formatting: bool = True
    show_progress_bars: bool = True
    auto_scroll: bool = True
    theme: str = "default"
    beginner_mode: bool = False
    teacher_mode: bool = False
    verbose_logging: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "show_welcome_message": self.show_welcome_message,
            "use_rich_formatting": self.use_rich_formatting,
            "show_progress_bars": self.show_progress_bars,
            "auto_scroll": self.auto_scroll,
            "theme": self.theme,
            "beginner_mode": self.beginner_mode,
            "teacher_mode": self.teacher_mode,
            "verbose_logging": self.verbose_logging
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UIConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class LauncherConfig:
    """Complete launcher configuration."""
    educational: EducationalConfig = field(default_factory=EducationalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "educational": self.educational.to_dict(),
            "training": self.training.to_dict(),
            "system": self.system.to_dict(),
            "ui": self.ui.to_dict(),
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LauncherConfig':
        """Create from dictionary."""
        return cls(
            educational=EducationalConfig.from_dict(data.get("educational", {})),
            training=TrainingConfig.from_dict(data.get("training", {})),
            system=SystemConfig.from_dict(data.get("system", {})),
            ui=UIConfig.from_dict(data.get("ui", {})),
            version=data.get("version", "1.0.0")
        )


class ConfigManager:
    """Manages launcher configuration with educational focus."""
    
    DEFAULT_CONFIG_PATHS = [
        "launcher/config.yaml",
        "configs/launcher.yaml",
        ".toaripi-launcher.yaml",
        "~/.toaripi/launcher.yaml"
    ]
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager."""
        self.config_path = self._resolve_config_path(config_path)
        self._config: Optional[LauncherConfig] = None
    
    def _resolve_config_path(self, config_path: Optional[Union[str, Path]]) -> Path:
        """Resolve configuration file path."""
        if config_path:
            return Path(config_path)
        
        # Try default paths
        for path_str in self.DEFAULT_CONFIG_PATHS:
            path = Path(path_str).expanduser()
            if path.exists():
                return path
        
        # Return first default path for creation
        return Path(self.DEFAULT_CONFIG_PATHS[0])
    
    def load_config(self) -> LauncherConfig:
        """Load configuration from file or create default."""
        if self._config is not None:
            return self._config
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                self._config = LauncherConfig.from_dict(data)
            except Exception as e:
                # If config file is corrupted, use defaults and warn
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                print("Using default configuration.")
                self._config = LauncherConfig()
        else:
            # Create default configuration
            self._config = LauncherConfig()
        
        return self._config
    
    def save_config(self, config: Optional[LauncherConfig] = None) -> None:
        """Save configuration to file."""
        if config is None:
            config = self._config
        
        if config is None:
            raise ValueError("No configuration to save")
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with educational-focused comments
        config_data = config.to_dict()
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                # Write header comment
                f.write("# Toaripi SLM Launcher Configuration\n")
                f.write("# Configuration for educational content generation and cultural preservation\n")
                f.write("# See docs/usage/CONFIGURATION_GUIDE.md for detailed explanations\n\n")
                
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise ValueError(f"Could not save configuration to {self.config_path}: {e}")
        
        self._config = config
    
    def get_config(self) -> LauncherConfig:
        """Get current configuration."""
        return self.load_config()
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values."""
        config = self.load_config()
        
        # Update nested configurations
        for section, values in kwargs.items():
            if hasattr(config, section) and isinstance(values, dict):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
        
        self.save_config(config)
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self._config = LauncherConfig()
        self.save_config()
    
    def get_educational_config(self) -> EducationalConfig:
        """Get educational configuration section."""
        return self.load_config().educational
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration section."""
        return self.load_config().training
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration section."""
        return self.load_config().system
    
    def get_ui_config(self) -> UIConfig:
        """Get UI configuration section."""
        return self.load_config().ui
    
    def create_teacher_config(self) -> LauncherConfig:
        """Create teacher-optimized configuration."""
        config = LauncherConfig()
        
        # Teacher-friendly settings
        config.ui.teacher_mode = True
        config.ui.show_welcome_message = True
        config.ui.beginner_mode = False
        config.educational.age_groups = [AgeGroup.PRIMARY_LOWER, AgeGroup.PRIMARY_upper]
        config.educational.content_types = [ContentType.STORY, ContentType.VOCABULARY, ContentType.DIALOGUE]
        config.educational.validation_level = ValidationLevel.EDUCATIONAL
        config.training.epochs = 3
        config.training.use_lora = True
        config.system.auto_install_dependencies = True
        config.system.auto_create_venv = True
        
        return config
    
    def create_beginner_config(self) -> LauncherConfig:
        """Create beginner-friendly configuration."""
        config = LauncherConfig()
        
        # Beginner-friendly settings
        config.ui.beginner_mode = True
        config.ui.teacher_mode = False
        config.ui.show_welcome_message = True
        config.ui.verbose_logging = True
        config.educational.age_groups = [AgeGroup.PRIMARY_LOWER]
        config.educational.content_types = [ContentType.STORY]
        config.educational.validation_level = ValidationLevel.EDUCATIONAL
        config.training.epochs = 2  # Shorter for beginners
        config.training.use_lora = True
        config.system.auto_install_dependencies = True
        config.system.auto_create_venv = True
        
        return config
    
    def create_developer_config(self) -> LauncherConfig:
        """Create developer-optimized configuration."""
        config = LauncherConfig()
        
        # Developer settings
        config.ui.beginner_mode = False
        config.ui.teacher_mode = False
        config.ui.verbose_logging = True
        config.educational.validation_level = ValidationLevel.BASIC
        config.training.epochs = 5
        config.training.save_steps = 250
        config.training.eval_steps = 50
        config.system.auto_install_dependencies = False
        config.system.auto_create_venv = False
        
        return config


def get_default_config() -> LauncherConfig:
    """Get default launcher configuration."""
    return LauncherConfig()


def load_config_from_file(config_path: Union[str, Path]) -> LauncherConfig:
    """Load configuration from specific file."""
    manager = ConfigManager(config_path)
    return manager.load_config()


def save_config_to_file(config: LauncherConfig, config_path: Union[str, Path]) -> None:
    """Save configuration to specific file."""
    manager = ConfigManager(config_path)
    manager.save_config(config)


# Educational content validation templates
EDUCATIONAL_TEMPLATES = {
    "story_prompts": {
        AgeGroup.EARLY_CHILDHOOD: [
            "Simple actions with family",
            "Daily activities in village",
            "Playing with friends",
            "Helping parents"
        ],
        AgeGroup.PRIMARY_LOWER: [
            "Adventure in the forest",
            "Fishing with grandfather",
            "Community celebration",
            "Learning from elders"
        ],
        AgeGroup.PRIMARY_upper: [
            "Traditional stories retold",
            "Solving problems together",
            "Cultural ceremonies",
            "Environmental stewardship"
        ],
        AgeGroup.SECONDARY: [
            "Complex narratives",
            "Historical events",
            "Cultural preservation",
            "Leadership stories"
        ]
    },
    "vocabulary_themes": {
        AgeGroup.EARLY_CHILDHOOD: [
            "Family members",
            "Body parts",
            "Colors",
            "Animals"
        ],
        AgeGroup.PRIMARY_LOWER: [
            "Fishing equipment",
            "Daily activities",
            "Food and cooking",
            "Nature and weather"
        ],
        AgeGroup.PRIMARY_upper: [
            "Community roles",
            "Traditional crafts",
            "Cultural practices",
            "Environmental terms"
        ],
        AgeGroup.SECONDARY: [
            "Abstract concepts",
            "Cultural values",
            "Historical terms",
            "Complex descriptions"
        ]
    }
}


def get_educational_templates() -> Dict[str, Dict[AgeGroup, List[str]]]:
    """Get educational content templates by age group."""
    return EDUCATIONAL_TEMPLATES


def validate_educational_config(config: EducationalConfig) -> List[str]:
    """Validate educational configuration for cultural appropriateness."""
    issues = []
    
    # Check age group coverage
    if not config.age_groups:
        issues.append("No age groups selected - educational content needs target audience")
    
    # Check content type variety
    if not config.content_types:
        issues.append("No content types selected - need at least one type of educational material")
    
    # Validate cultural sensitivity
    if not config.cultural_sensitivity:
        issues.append("Cultural sensitivity should be enabled for Toaripi content")
    
    # Check content length appropriateness
    if config.max_content_length > 512:
        issues.append("Content length too long for primary education (recommend ≤256 tokens)")
    
    # Ensure language preservation mode
    if not config.language_preservation_mode:
        issues.append("Language preservation mode should be enabled for Toaripi SLM")
    
    return issues