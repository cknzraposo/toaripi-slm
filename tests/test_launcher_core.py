"""
Simplified test suite for Toaripi SLM launcher components.

Tests core functionality with mocked dependencies.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import launcher components
from launcher.config import (
    AgeGroup, ContentType, ValidationLevel,
    EducationalConfig, TrainingConfig, SystemConfig, UIConfig,
    LauncherConfig, ConfigManager
)
from launcher.validator import ValidationIssue, ValidationResult


class TestLauncherBasicFunctionality:
    """Test basic launcher functionality with minimal dependencies."""
    
    def test_enum_values(self):
        """Test enum value access."""
        assert AgeGroup.PRIMARY_upper.value == "primary_upper"
        assert ContentType.STORY.value == "story"
        assert ValidationLevel.EDUCATIONAL.value == "educational"
    
    def test_educational_config_creation(self):
        """Test educational configuration creation."""
        config = EducationalConfig()
        
        # Test default values
        assert AgeGroup.PRIMARY_LOWER in config.age_groups
        assert AgeGroup.PRIMARY_upper in config.age_groups
        assert ContentType.STORY in config.content_types
        assert ContentType.VOCABULARY in config.content_types
        assert config.validation_level == ValidationLevel.EDUCATIONAL
        assert config.cultural_sensitivity is True
        assert config.max_content_length == 256
        assert config.language_preservation_mode is True
    
    def test_educational_config_serialization(self):
        """Test configuration serialization to dictionary."""
        config = EducationalConfig(
            age_groups=[AgeGroup.PRIMARY_LOWER],
            content_types=[ContentType.STORY],
            validation_level=ValidationLevel.STRICT
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["age_groups"] == ["primary_lower"]
        assert config_dict["content_types"] == ["story"]
        assert config_dict["validation_level"] == "strict"
        assert config_dict["cultural_sensitivity"] is True
    
    def test_educational_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "age_groups": ["primary_lower", "primary_upper"],
            "content_types": ["story", "vocabulary"],
            "validation_level": "educational",
            "cultural_sensitivity": True,
            "max_content_length": 512,
            "language_preservation_mode": True
        }
        
        config = EducationalConfig.from_dict(config_dict)
        
        assert AgeGroup.PRIMARY_LOWER in config.age_groups
        assert AgeGroup.PRIMARY_upper in config.age_groups
        assert config.max_content_length == 512
    
    def test_training_config_creation(self):
        """Test training configuration creation."""
        config = TrainingConfig()
        
        assert config.model_name == "microsoft/DialoGPT-small"
        assert config.learning_rate == 2e-5
        assert config.batch_size == 4
        assert config.epochs == 3
        assert config.use_lora is True
        assert config.max_length == 256
    
    def test_training_config_serialization(self):
        """Test training configuration serialization."""
        config = TrainingConfig(
            model_name="test-model",
            learning_rate=1e-4,
            batch_size=8
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["model_name"] == "test-model"
        assert config_dict["learning_rate"] == 1e-4
        assert config_dict["batch_size"] == 8
    
    def test_system_config_creation(self):
        """Test system configuration creation."""
        config = SystemConfig()
        
        # Test basic creation works
        assert isinstance(config, SystemConfig)
    
    def test_ui_config_creation(self):
        """Test UI configuration creation."""
        config = UIConfig()
        
        # Test basic creation works
        assert isinstance(config, UIConfig)
    
    def test_launcher_config_creation(self):
        """Test launcher configuration creation."""
        config = LauncherConfig()
        
        assert isinstance(config.educational, EducationalConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.system, SystemConfig)
        assert isinstance(config.ui, UIConfig)
    
    def test_config_manager_basic(self):
        """Test basic config manager functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            manager = ConfigManager(config_path)
            
            # Test basic creation
            assert manager.config_path == config_path
            
            # Test config loading
            config = manager.load_config()
            assert isinstance(config, LauncherConfig)
    
    def test_validation_issue_creation(self):
        """Test validation issue creation."""
        issue = ValidationIssue(
            component="python",
            issue="Version too old",
            severity="error",
            fix_suggestion="Upgrade to Python 3.10+",
            auto_fixable=False
        )
        
        assert issue.component == "python"
        assert issue.severity == "error"
        assert issue.auto_fixable is False
    
    def test_validation_result_creation(self):
        """Test validation result creation."""
        issues = [
            ValidationIssue(
                component="python",
                issue="Version check",
                severity="info",
                fix_suggestion="No action needed"
            )
        ]
        
        result = ValidationResult(
            is_valid=True,
            issues=issues,
            python_version="3.11",
            venv_exists=True
        )
        
        assert result.is_valid is True
        assert len(result.issues) == 1
        assert result.python_version == "3.11"
        assert result.venv_exists is True


class TestConfigurationIntegration:
    """Test configuration integration functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
    
    def test_config_save_and_load(self):
        """Test configuration save and load functionality."""
        manager = ConfigManager(self.config_path)
        
        # Load default config
        config = manager.load_config()
        
        # Modify configuration
        config.educational.max_content_length = 512
        config.training.batch_size = 8
        
        # Save configuration
        manager.save_config(config)
        assert self.config_path.exists()
        
        # Load configuration in new manager
        new_manager = ConfigManager(self.config_path)
        new_config = new_manager.load_config()
        
        assert new_config.educational.max_content_length == 512
        assert new_config.training.batch_size == 8
    
    def test_educational_settings_integration(self):
        """Test educational settings integration."""
        manager = ConfigManager(self.config_path)
        config = manager.load_config()
        
        # Test educational settings
        assert AgeGroup.PRIMARY_upper in config.educational.age_groups
        assert config.educational.cultural_sensitivity is True
    
    def test_yaml_serialization(self):
        """Test YAML serialization of configuration."""
        manager = ConfigManager(self.config_path)
        config = manager.load_config()
        
        # Update and save
        config.educational.max_content_length = 1024
        manager.save_config(config)
        
        # Read raw YAML and verify
        with open(self.config_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        assert yaml_data["educational"]["max_content_length"] == 1024


class TestEducationalContentSettings:
    """Test educational content specific settings."""
    
    def test_age_group_configuration(self):
        """Test age group configuration."""
        config = EducationalConfig(
            age_groups=[AgeGroup.PRIMARY_LOWER, AgeGroup.EARLY_CHILDHOOD]
        )
        
        assert AgeGroup.PRIMARY_LOWER in config.age_groups
        assert AgeGroup.EARLY_CHILDHOOD in config.age_groups
        assert AgeGroup.SECONDARY not in config.age_groups
    
    def test_content_type_configuration(self):
        """Test content type configuration."""
        config = EducationalConfig(
            content_types=[ContentType.STORY, ContentType.DIALOGUE]
        )
        
        assert ContentType.STORY in config.content_types
        assert ContentType.DIALOGUE in config.content_types
        assert ContentType.EXERCISE not in config.content_types
    
    def test_validation_level_configuration(self):
        """Test validation level configuration."""
        # Basic validation
        config = EducationalConfig(validation_level=ValidationLevel.BASIC)
        assert config.validation_level == ValidationLevel.BASIC
        
        # Strict validation
        config = EducationalConfig(validation_level=ValidationLevel.STRICT)
        assert config.validation_level == ValidationLevel.STRICT
    
    def test_cultural_sensitivity_settings(self):
        """Test cultural sensitivity settings."""
        # Enabled
        config = EducationalConfig(cultural_sensitivity=True)
        assert config.cultural_sensitivity is True
        
        # Disabled
        config = EducationalConfig(cultural_sensitivity=False)
        assert config.cultural_sensitivity is False
    
    def test_language_preservation_settings(self):
        """Test language preservation settings."""
        # Enabled (default)
        config = EducationalConfig()
        assert config.language_preservation_mode is True
        
        # Disabled
        config = EducationalConfig(language_preservation_mode=False)
        assert config.language_preservation_mode is False


class TestTrainingSettings:
    """Test training configuration settings."""
    
    def test_model_selection(self):
        """Test model selection configuration."""
        config = TrainingConfig(model_name="test/model")
        assert config.model_name == "test/model"
    
    def test_training_parameters(self):
        """Test training parameter configuration."""
        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=8,
            epochs=5
        )
        
        assert config.learning_rate == 1e-4
        assert config.batch_size == 8
        assert config.epochs == 5
    
    def test_lora_configuration(self):
        """Test LoRA configuration."""
        # LoRA enabled
        config = TrainingConfig(use_lora=True)
        assert config.use_lora is True
        
        # LoRA disabled
        config = TrainingConfig(use_lora=False)
        assert config.use_lora is False
    
    def test_max_length_configuration(self):
        """Test max length configuration."""
        config = TrainingConfig(max_length=512)
        assert config.max_length == 512


if __name__ == "__main__":
    pytest.main([__file__, "-v"])