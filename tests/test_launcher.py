"""
Comprehensive test suite for Toaripi SLM launcher components.

This module tests launcher functionality, system validation, configuration management,
and educational content workflows with focus on cultural sensitivity.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import launcher components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from launcher.config import (
    ConfigManager, LauncherConfig, EducationalConfig, TrainingConfig,
    SystemConfig, UIConfig, AgeGroup, ContentType, ValidationLevel,
    get_educational_templates, validate_educational_config
)
from launcher.validator import SystemValidator, ValidationResult, ValidationIssue
from launcher.guidance import UserGuidance
from launcher.launcher import ToaripiLauncher


class TestEducationalConfig:
    """Test educational configuration management."""
    
    def test_default_educational_config(self):
        """Test default educational configuration values."""
        config = EducationalConfig()
        
        assert AgeGroup.PRIMARY_LOWER in config.age_groups
        assert AgeGroup.PRIMARY_upper in config.age_groups
        assert ContentType.STORY in config.content_types
        assert ContentType.VOCABULARY in config.content_types
        assert config.validation_level == ValidationLevel.EDUCATIONAL
        assert config.cultural_sensitivity is True
        assert config.language_preservation_mode is True
    
    def test_educational_config_serialization(self):
        """Test educational config to/from dict conversion."""
        config = EducationalConfig(
            age_groups=[AgeGroup.PRIMARY_LOWER],
            content_types=[ContentType.STORY],
            validation_level=ValidationLevel.STRICT,
            cultural_sensitivity=True,
            max_content_length=128
        )
        
        # Test to_dict
        data = config.to_dict()
        assert data["age_groups"] == ["primary_lower"]
        assert data["content_types"] == ["story"]
        assert data["validation_level"] == "strict"
        assert data["cultural_sensitivity"] is True
        assert data["max_content_length"] == 128
        
        # Test from_dict
        restored_config = EducationalConfig.from_dict(data)
        assert restored_config.age_groups == [AgeGroup.PRIMARY_LOWER]
        assert restored_config.content_types == [ContentType.STORY]
        assert restored_config.validation_level == ValidationLevel.STRICT
        assert restored_config.cultural_sensitivity is True
        assert restored_config.max_content_length == 128
    
    def test_educational_config_validation(self):
        """Test educational configuration validation."""
        # Valid configuration
        valid_config = EducationalConfig(
            age_groups=[AgeGroup.PRIMARY_LOWER],
            content_types=[ContentType.STORY],
            cultural_sensitivity=True,
            language_preservation_mode=True
        )
        
        issues = validate_educational_config(valid_config)
        assert len(issues) == 0
        
        # Invalid configuration - no age groups
        invalid_config = EducationalConfig(
            age_groups=[],
            content_types=[ContentType.STORY],
            cultural_sensitivity=False,
            language_preservation_mode=False,
            max_content_length=1000
        )
        
        issues = validate_educational_config(invalid_config)
        assert len(issues) > 0
        assert any("age groups" in issue for issue in issues)
        assert any("cultural sensitivity" in issue for issue in issues)
        assert any("language preservation" in issue for issue in issues)
        assert any("content length" in issue for issue in issues)


class TestTrainingConfig:
    """Test training configuration management."""
    
    def test_default_training_config(self):
        """Test default training configuration values."""
        config = TrainingConfig()
        
        assert config.model_name == "microsoft/DialoGPT-small"
        assert config.epochs == 3
        assert config.batch_size == 4
        assert config.learning_rate == 2e-5
        assert config.use_lora is True
        assert config.max_length == 256
        assert config.device == "auto"
    
    def test_training_config_serialization(self):
        """Test training config to/from dict conversion."""
        config = TrainingConfig(
            model_name="test/model",
            epochs=5,
            batch_size=8,
            use_lora=False
        )
        
        # Test to_dict
        data = config.to_dict()
        assert data["model_name"] == "test/model"
        assert data["epochs"] == 5
        assert data["batch_size"] == 8
        assert data["use_lora"] is False
        
        # Test from_dict
        restored_config = TrainingConfig.from_dict(data)
        assert restored_config.model_name == "test/model"
        assert restored_config.epochs == 5
        assert restored_config.batch_size == 8
        assert restored_config.use_lora is False


class TestSystemConfig:
    """Test system configuration management."""
    
    def test_default_system_config(self):
        """Test default system configuration values."""
        config = SystemConfig()
        
        assert config.python_version_min == "3.10"
        assert config.virtual_env_required is True
        assert config.auto_install_dependencies is True
        assert config.auto_create_venv is True
        assert config.check_gpu_availability is True
        assert config.disk_space_required_gb == 10


class TestConfigManager:
    """Test configuration manager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_config_manager_initialization(self):
        """Test config manager initialization."""
        manager = ConfigManager(self.config_path)
        assert manager.config_path == self.config_path
    
    def test_load_default_config(self):
        """Test loading default configuration when file doesn't exist."""
        manager = ConfigManager(self.config_path)
        config = manager.load_config()
        
        assert isinstance(config, LauncherConfig)
        assert config.version == "1.0.0"
        assert isinstance(config.educational, EducationalConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.system, SystemConfig)
        assert isinstance(config.ui, UIConfig)
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        manager = ConfigManager(self.config_path)
        
        # Create custom config
        config = LauncherConfig()
        config.training.epochs = 5
        config.educational.age_groups = [AgeGroup.PRIMARY_LOWER]
        
        # Save config
        manager.save_config(config)
        assert self.config_path.exists()
        
        # Load config
        manager._config = None  # Reset cache
        loaded_config = manager.load_config()
        
        assert loaded_config.training.epochs == 5
        assert loaded_config.educational.age_groups == [AgeGroup.PRIMARY_LOWER]
    
    def test_create_teacher_config(self):
        """Test teacher-optimized configuration creation."""
        manager = ConfigManager(self.config_path)
        config = manager.create_teacher_config()
        
        assert config.ui.teacher_mode is True
        assert config.ui.beginner_mode is False
        assert config.educational.validation_level == ValidationLevel.EDUCATIONAL
        assert AgeGroup.PRIMARY_LOWER in config.educational.age_groups
        assert AgeGroup.PRIMARY_UPPER in config.educational.age_groups
    
    def test_create_beginner_config(self):
        """Test beginner-friendly configuration creation."""
        manager = ConfigManager(self.config_path)
        config = manager.create_beginner_config()
        
        assert config.ui.beginner_mode is True
        assert config.ui.teacher_mode is False
        assert config.ui.verbose_logging is True
        assert config.educational.age_groups == [AgeGroup.PRIMARY_LOWER]
        assert config.educational.content_types == [ContentType.STORY]
        assert config.training.epochs == 2  # Shorter for beginners
    
    def test_create_developer_config(self):
        """Test developer-optimized configuration creation."""
        manager = ConfigManager(self.config_path)
        config = manager.create_developer_config()
        
        assert config.ui.beginner_mode is False
        assert config.ui.teacher_mode is False
        assert config.ui.verbose_logging is True
        assert config.educational.validation_level == ValidationLevel.BASIC
        assert config.system.auto_install_dependencies is False
        assert config.system.auto_create_venv is False


class TestValidationIssue:
    """Test validation issue representation."""
    
    def test_validation_issue_creation(self):
        """Test validation issue creation."""
        issue = ValidationIssue(
            component="Python Version",
            message="Python 3.11+ required",
            severity="error",
            auto_fix="python -m pip install --upgrade python"
        )
        
        assert issue.component == "Python Version"
        assert issue.message == "Python 3.11+ required"
        assert issue.severity == "error"
        assert issue.auto_fix == "python -m pip install --upgrade python"
    
    def test_validation_issue_defaults(self):
        """Test validation issue default values."""
        issue = ValidationIssue(
            component="Test Component",
            message="Test message"
        )
        
        assert issue.severity == "warning"
        assert issue.auto_fix is None


class TestValidationResult:
    """Test validation result aggregation."""
    
    def test_validation_result_valid(self):
        """Test validation result with no errors."""
        result = ValidationResult(
            is_valid=True,
            issues=[],
            python_version="3.11.0",
            system_info={"platform": "Windows"}
        )
        
        assert result.is_valid is True
        assert len(result.issues) == 0
        assert result.python_version == "3.11.0"
    
    def test_validation_result_invalid(self):
        """Test validation result with errors."""
        issues = [
            ValidationIssue("Python", "Version too old", "error"),
            ValidationIssue("Dependencies", "Missing packages", "warning")
        ]
        
        result = ValidationResult(
            is_valid=False,
            issues=issues,
            python_version="3.8.0",
            system_info={"platform": "Windows"}
        )
        
        assert result.is_valid is False
        assert len(result.issues) == 2
        assert result.issues[0].severity == "error"
        assert result.issues[1].severity == "warning"


class TestSystemValidator:
    """Test system validation functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.validator = SystemValidator()
    
    @patch('sys.version_info', (3, 11, 0))
    def test_validate_python_version_success(self):
        """Test successful Python version validation."""
        # This test would require mocking the actual validation logic
        pass
    
    @patch('sys.version_info', (3, 8, 0))
    def test_validate_python_version_failure(self):
        """Test failed Python version validation."""
        # This test would require mocking the actual validation logic
        pass
    
    def test_validate_educational_data(self):
        """Test educational data validation."""
        # This test would check for Toaripi educational data availability
        pass


class TestUserGuidance:
    """Test user guidance functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.console = Mock()
        self.guidance = UserGuidance(self.console)
    
    def test_guidance_initialization(self):
        """Test guidance system initialization."""
        assert self.guidance.console == self.console
    
    def test_show_beginner_welcome(self):
        """Test beginner welcome message display."""
        # This test would verify the welcome message is shown
        # self.guidance.show_beginner_welcome()
        # assert self.console.print.called
        pass
    
    def test_show_system_error_resolution(self):
        """Test system error resolution guidance."""
        issues = [
            ValidationIssue("Python", "Version too old", "error"),
            ValidationIssue("Dependencies", "Missing packages", "warning")
        ]
        
        # This test would verify error resolution guidance is shown
        # self.guidance.show_error_resolution(issues)
        # assert self.console.print.called
        pass


class TestToaripiLauncher:
    """Test main launcher functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        
        # Mock the console to avoid rich dependency issues in tests
        with patch('launcher.launcher.RICH_AVAILABLE', False):
            self.launcher = ToaripiLauncher(self.config_path)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_launcher_initialization(self):
        """Test launcher initialization."""
        assert self.launcher.config_manager is not None
        assert self.launcher.config is not None
        assert self.launcher.validator is not None
        assert self.launcher.guidance is not None
    
    def test_configure_for_beginner_mode(self):
        """Test launcher configuration for beginner mode."""
        self.launcher._configure_for_mode("beginner", True, False)
        
        assert self.launcher.config.ui.beginner_mode is True
        assert self.launcher.config.ui.teacher_mode is False
        assert self.launcher.config.educational.age_groups == [AgeGroup.PRIMARY_LOWER]
    
    def test_configure_for_teacher_mode(self):
        """Test launcher configuration for teacher mode."""
        self.launcher._configure_for_mode("teacher", False, True)
        
        assert self.launcher.config.ui.teacher_mode is True
        assert self.launcher.config.ui.beginner_mode is False
        assert AgeGroup.PRIMARY_LOWER in self.launcher.config.educational.age_groups
        assert AgeGroup.PRIMARY_UPPER in self.launcher.config.educational.age_groups
    
    def test_configure_for_developer_mode(self):
        """Test launcher configuration for developer mode."""
        self.launcher._configure_for_mode("developer", False, False)
        
        assert self.launcher.config.ui.beginner_mode is False
        assert self.launcher.config.ui.teacher_mode is False
        assert self.launcher.config.educational.validation_level == ValidationLevel.BASIC
    
    @patch('subprocess.run')
    def test_launch_training_success(self, mock_subprocess):
        """Test successful training launch."""
        # Mock successful subprocess execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        result = self.launcher._launch_training()
        assert result is True
        
        # Verify subprocess was called with correct command
        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        command = args[0]
        assert "python" in command
        assert "toaripi_slm.cli" in command
        assert "train" in command
    
    @patch('subprocess.run')
    def test_launch_training_failure(self, mock_subprocess):
        """Test failed training launch."""
        # Mock failed subprocess execution
        mock_subprocess.side_effect = FileNotFoundError()
        
        result = self.launcher._launch_training()
        assert result is False


class TestEducationalTemplates:
    """Test educational content templates."""
    
    def test_get_educational_templates(self):
        """Test educational template retrieval."""
        templates = get_educational_templates()
        
        assert "story_prompts" in templates
        assert "vocabulary_themes" in templates
        
        # Check age group coverage
        story_prompts = templates["story_prompts"]
        assert AgeGroup.EARLY_CHILDHOOD in story_prompts
        assert AgeGroup.PRIMARY_LOWER in story_prompts
        assert AgeGroup.PRIMARY_UPPER in story_prompts
        assert AgeGroup.SECONDARY in story_prompts
        
        # Check vocabulary themes
        vocab_themes = templates["vocabulary_themes"]
        assert AgeGroup.EARLY_CHILDHOOD in vocab_themes
        assert AgeGroup.PRIMARY_LOWER in vocab_themes
        
        # Verify content appropriateness
        early_childhood_stories = story_prompts[AgeGroup.EARLY_CHILDHOOD]
        assert any("family" in prompt.lower() for prompt in early_childhood_stories)
        
        primary_lower_vocab = vocab_themes[AgeGroup.PRIMARY_LOWER]
        assert any("fishing" in theme.lower() for theme in primary_lower_vocab)
    
    def test_educational_content_cultural_focus(self):
        """Test that educational templates maintain cultural focus."""
        templates = get_educational_templates()
        
        # Check for cultural themes in story prompts
        story_prompts = templates["story_prompts"]
        primary_stories = story_prompts[AgeGroup.PRIMARY_UPPER]
        
        cultural_themes = ["traditional", "cultural", "community", "elders"]
        assert any(any(theme in prompt.lower() for theme in cultural_themes) 
                  for prompt in primary_stories)
        
        # Check for environmental themes
        environmental_themes = ["environment", "fishing", "nature"]
        vocab_themes = templates["vocabulary_themes"]
        primary_vocab = vocab_themes[AgeGroup.PRIMARY_LOWER]
        
        assert any(any(theme in vocab.lower() for theme in environmental_themes)
                  for vocab in primary_vocab)


class TestLauncherIntegration:
    """Integration tests for complete launcher workflow."""
    
    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "integration_config.yaml"
    
    def teardown_method(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.temp_dir)
    
    @patch('launcher.launcher.RICH_AVAILABLE', False)
    @patch('subprocess.run')
    def test_full_beginner_workflow(self, mock_subprocess):
        """Test complete beginner workflow from config to launch."""
        # Mock successful subprocess execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create launcher in beginner mode
        launcher = ToaripiLauncher(self.config_path)
        
        # Mock validation to always pass
        with patch.object(launcher, '_validate_system') as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=True,
                issues=[],
                python_version="3.11.0",
                system_info={"platform": "Windows"}
            )
            
            # Launch in beginner mode
            result = launcher.launch(mode="beginner", force_beginner=True)
            
            # Verify successful launch
            assert result is True
            
            # Verify configuration was set correctly
            assert launcher.config.ui.beginner_mode is True
            assert launcher.config.educational.age_groups == [AgeGroup.PRIMARY_LOWER]
            assert launcher.config.educational.content_types == [ContentType.STORY]
            
            # Verify training command was called
            mock_subprocess.assert_called_once()
    
    @patch('launcher.launcher.RICH_AVAILABLE', False)
    @patch('subprocess.run')
    def test_full_teacher_workflow(self, mock_subprocess):
        """Test complete teacher workflow from config to launch."""
        # Mock successful subprocess execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create launcher in teacher mode
        launcher = ToaripiLauncher(self.config_path)
        
        # Mock validation to always pass
        with patch.object(launcher, '_validate_system') as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=True,
                issues=[],
                python_version="3.11.0",
                system_info={"platform": "Windows"}
            )
            
            # Launch in teacher mode
            result = launcher.launch(mode="teacher", teacher_mode=True)
            
            # Verify successful launch
            assert result is True
            
            # Verify teacher configuration
            assert launcher.config.ui.teacher_mode is True
            assert AgeGroup.PRIMARY_LOWER in launcher.config.educational.age_groups
            assert AgeGroup.PRIMARY_UPPER in launcher.config.educational.age_groups
            assert ContentType.STORY in launcher.config.educational.content_types
            assert ContentType.VOCABULARY in launcher.config.educational.content_types
            
            # Verify training command was called
            mock_subprocess.assert_called_once()


# Test fixtures and utilities
@pytest.fixture
def sample_educational_config():
    """Sample educational configuration for testing."""
    return EducationalConfig(
        age_groups=[AgeGroup.PRIMARY_LOWER, AgeGroup.PRIMARY_UPPER],
        content_types=[ContentType.STORY, ContentType.VOCABULARY],
        validation_level=ValidationLevel.EDUCATIONAL,
        cultural_sensitivity=True,
        max_content_length=256,
        language_preservation_mode=True
    )


@pytest.fixture
def sample_training_config():
    """Sample training configuration for testing."""
    return TrainingConfig(
        model_name="microsoft/DialoGPT-small",
        epochs=3,
        batch_size=4,
        learning_rate=2e-5,
        use_lora=True,
        max_length=256
    )


@pytest.fixture
def sample_validation_issues():
    """Sample validation issues for testing."""
    return [
        ValidationIssue(
            component="Python Version",
            message="Python 3.11+ required, found 3.8.0",
            severity="error",
            auto_fix="python -m pip install --upgrade python"
        ),
        ValidationIssue(
            component="Virtual Environment",
            message="No virtual environment detected",
            severity="warning",
            auto_fix="python -m venv .venv"
        ),
        ValidationIssue(
            component="Educational Data",
            message="Toaripi parallel data not found",
            severity="error",
            auto_fix=None
        )
    ]


# Performance and educational content tests
class TestEducationalContentValidation:
    """Test educational content validation specifically."""
    
    def test_age_appropriate_content_detection(self):
        """Test detection of age-appropriate content markers."""
        # This would test content analysis for age appropriateness
        # Including vocabulary complexity, sentence structure, themes
        pass
    
    def test_cultural_sensitivity_validation(self):
        """Test cultural sensitivity validation."""
        # This would test validation of content for cultural appropriateness
        # Including respect for traditions, avoidance of stereotypes
        pass
    
    def test_toaripi_language_preservation(self):
        """Test language preservation features."""
        # This would test features specific to Toaripi language preservation
        # Including traditional knowledge, environmental themes, community values
        pass


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])