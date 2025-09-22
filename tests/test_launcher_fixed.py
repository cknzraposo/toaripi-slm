"""
Comprehensive test suite for Toaripi SLM launcher components.

Tests cover configuration management, system validation, user guidance,
and educational content generation with focus on cultural sensitivity.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from rich.console import Console

# Import launcher components
from launcher.config import (
    AgeGroup, ContentType, ValidationLevel,
    EducationalConfig, TrainingConfig, SystemConfig, UIConfig,
    LauncherConfig, ConfigManager
)
from launcher.validator import ValidationIssue, ValidationResult, SystemValidator
from launcher.guidance import UserGuidance
from launcher.launcher import ToaripiLauncher


class TestEducationalConfig:
    """Test educational configuration management."""
    
    def test_default_educational_config(self):
        """Test default educational configuration."""
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
    
    def test_age_group_guidelines(self):
        """Test age group specific guidelines."""
        config = EducationalConfig()
        guidelines = config.get_age_group_guidelines(AgeGroup.PRIMARY_LOWER)
        
        assert "simple vocabulary" in guidelines
        assert "short sentences" in guidelines
        assert "familiar concepts" in guidelines
    
    def test_cultural_validation(self):
        """Test cultural sensitivity validation."""
        config = EducationalConfig(cultural_sensitivity=True)
        
        # Valid content
        assert config.validate_cultural_content("Children helping with fishing") is True
        
        # Invalid content (should be filtered)
        assert config.validate_cultural_content("Religious doctrine") is False
        assert config.validate_cultural_content("Adult themes") is False


class TestTrainingConfig:
    """Test training configuration management."""
    
    def test_default_training_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        
        assert config.model_name == "microsoft/DialoGPT-small"
        assert config.learning_rate == 2e-5
        assert config.batch_size == 4
        assert config.epochs == 3
        assert config.use_lora is True
        assert config.max_length == 256
    
    def test_training_config_validation(self):
        """Test training configuration validation."""
        # Valid config
        config = TrainingConfig(
            model_name="microsoft/DialoGPT-small",
            learning_rate=1e-4,
            batch_size=8
        )
        assert config.validate() is True
        
        # Invalid config - learning rate too high
        config.learning_rate = 1.0
        assert config.validate() is False
    
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


class TestSystemConfig:
    """Test system configuration management."""
    
    def test_default_system_config(self):
        """Test default system configuration."""
        config = SystemConfig()
        
        assert config.python_min_version == "3.10"
        assert config.require_venv is True
        assert config.auto_install_deps is True
        assert config.data_dir == Path("data")
        assert config.models_dir == Path("models")
    
    def test_system_config_validation(self):
        """Test system configuration path validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            config = SystemConfig(
                data_dir=temp_path / "data",
                models_dir=temp_path / "models"
            )
            
            # Create directories
            config.data_dir.mkdir(exist_ok=True)
            config.models_dir.mkdir(exist_ok=True)
            
            validation_result = config.validate_paths()
            assert validation_result["data_dir_exists"] is True
            assert validation_result["models_dir_exists"] is True


class TestConfigManager:
    """Test configuration manager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
    
    def test_config_manager_creation(self):
        """Test configuration manager creation."""
        manager = ConfigManager(self.config_path)
        
        assert manager.config_path == self.config_path
        assert isinstance(manager.config, LauncherConfig)
    
    def test_config_save_and_load(self):
        """Test configuration save and load functionality."""
        manager = ConfigManager(self.config_path)
        
        # Modify configuration
        manager.config.educational.max_content_length = 512
        manager.config.training.batch_size = 8
        
        # Save configuration
        manager.save_config()
        assert self.config_path.exists()
        
        # Load configuration
        new_manager = ConfigManager(self.config_path)
        new_manager.load_config()
        
        assert new_manager.config.educational.max_content_length == 512
        assert new_manager.config.training.batch_size == 8
    
    def test_config_validation(self):
        """Test configuration validation."""
        manager = ConfigManager(self.config_path)
        
        # Valid configuration
        validation_result = manager.validate_config()
        assert validation_result["is_valid"] is True
        
        # Invalid configuration
        manager.config.educational.age_groups = []  # Empty age groups
        validation_result = manager.validate_config()
        assert validation_result["is_valid"] is False
        assert "educational" in validation_result["issues"]
    
    def test_educational_config_integration(self):
        """Test educational configuration integration."""
        manager = ConfigManager(self.config_path)
        
        # Test educational settings
        assert AgeGroup.PRIMARY_upper in manager.config.educational.age_groups
        assert manager.config.educational.cultural_sensitivity is True
        
        # Test educational content guidelines
        guidelines = manager.get_educational_guidelines()
        assert "cultural_sensitivity" in guidelines
        assert "age_appropriate" in guidelines


class TestValidationIssue:
    """Test validation issue representation."""
    
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
    
    def test_validation_issue_defaults(self):
        """Test validation issue default values."""
        issue = ValidationIssue(
            component="test",
            issue="test issue",
            severity="warning",
            fix_suggestion="test fix"
        )
        
        assert issue.auto_fixable is False


class TestValidationResult:
    """Test validation result handling."""
    
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
    
    def test_validation_result_defaults(self):
        """Test validation result default values."""
        result = ValidationResult(is_valid=False, issues=[])
        
        assert result.python_version is None
        assert result.venv_exists is False
        assert result.dependencies_installed is False
        assert result.system_info == {}


class TestSystemValidator:
    """Test system validation functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.console = Console()
        self.validator = SystemValidator(self.console)
    
    def test_validator_creation(self):
        """Test validator creation."""
        assert isinstance(self.validator.console, Console)
        assert isinstance(self.validator.project_root, Path)
    
    @patch('sys.version_info', (3, 11, 0))
    def test_python_version_validation(self):
        """Test Python version validation."""
        result = self.validator.validate_python_version()
        
        assert result.component == "python"
        assert result.severity == "info"
        assert "3.11" in result.issue
    
    @patch('sys.version_info', (3, 8, 0))
    def test_python_version_validation_old(self):
        """Test Python version validation with old version."""
        result = self.validator.validate_python_version()
        
        assert result.component == "python"
        assert result.severity == "error"
        assert "upgrade" in result.fix_suggestion.lower()
    
    def test_validate_all(self):
        """Test complete validation run."""
        with patch.object(self.validator, 'validate_python_version') as mock_python, \
             patch.object(self.validator, 'validate_virtual_environment') as mock_venv, \
             patch.object(self.validator, 'validate_dependencies') as mock_deps:
            
            mock_python.return_value = ValidationIssue(
                component="python", issue="OK", severity="info", fix_suggestion="None"
            )
            mock_venv.return_value = ValidationIssue(
                component="venv", issue="OK", severity="info", fix_suggestion="None"
            )
            mock_deps.return_value = ValidationIssue(
                component="deps", issue="OK", severity="info", fix_suggestion="None"
            )
            
            result = self.validator.validate_all()
            
            assert isinstance(result, ValidationResult)
            assert len(result.issues) == 3
    
    def test_educational_content_validation(self):
        """Test educational content validation."""
        # Valid educational content
        is_valid = self.validator.validate_educational_content(
            "Children learning about fishing",
            AgeGroup.PRIMARY_LOWER
        )
        assert is_valid is True
        
        # Invalid content for age group
        is_valid = self.validator.validate_educational_content(
            "Complex philosophical concepts",
            AgeGroup.PRIMARY_LOWER
        )
        assert is_valid is False


class TestUserGuidance:
    """Test user guidance system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.console = Console()
        self.guidance = UserGuidance(self.console)
    
    def test_guidance_creation(self):
        """Test guidance system creation."""
        assert isinstance(self.guidance.console, Console)
    
    def test_welcome_message(self):
        """Test welcome message display."""
        # Mock console output
        with patch.object(self.console, 'print') as mock_print:
            self.guidance.show_welcome()
            
            # Verify welcome message was displayed
            mock_print.assert_called()
            call_args = str(mock_print.call_args_list)
            assert "Toaripi" in call_args
    
    def test_educational_guidelines(self):
        """Test educational guidelines display."""
        with patch.object(self.console, 'print') as mock_print:
            self.guidance.show_educational_guidelines()
            
            mock_print.assert_called()
            call_args = str(mock_print.call_args_list)
            assert "educational" in call_args.lower()
    
    def test_training_guidance(self):
        """Test training guidance for beginners."""
        with patch.object(self.console, 'print') as mock_print:
            self.guidance.show_training_guidance(beginner=True)
            
            mock_print.assert_called()
            call_args = str(mock_print.call_args_list)
            assert "beginner" in call_args.lower() or "guide" in call_args.lower()
    
    def test_cultural_guidelines(self):
        """Test cultural sensitivity guidelines."""
        with patch.object(self.console, 'print') as mock_print:
            self.guidance.show_cultural_guidelines()
            
            mock_print.assert_called()
            call_args = str(mock_print.call_args_list)
            assert "cultural" in call_args.lower()


class TestToaripiLauncher:
    """Test main launcher functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        self.launcher = ToaripiLauncher(self.config_path)
    
    def test_launcher_creation(self):
        """Test launcher creation."""
        assert isinstance(self.launcher.console, Console)
        assert isinstance(self.launcher.config_manager, ConfigManager)
        assert isinstance(self.launcher.validator, SystemValidator)
        assert isinstance(self.launcher.guidance, UserGuidance)
    
    def test_launcher_config_access(self):
        """Test launcher configuration access."""
        config = self.launcher.config
        
        assert isinstance(config, LauncherConfig)
        assert AgeGroup.PRIMARY_upper in config.educational.age_groups
        assert config.educational.cultural_sensitivity is True
    
    def test_validation_run(self):
        """Test launcher validation."""
        with patch.object(self.launcher.validator, 'validate_all') as mock_validate:
            mock_result = ValidationResult(
                is_valid=True,
                issues=[],
                python_version="3.11",
                venv_exists=True
            )
            mock_validate.return_value = mock_result
            
            result = self.launcher.validate_system()
            
            assert result.is_valid is True
            assert result.python_version == "3.11"
    
    def test_educational_mode_setup(self):
        """Test educational mode configuration."""
        self.launcher.setup_educational_mode(
            age_group=AgeGroup.PRIMARY_LOWER,
            content_types=[ContentType.STORY, ContentType.VOCABULARY]
        )
        
        config = self.launcher.config.educational
        assert AgeGroup.PRIMARY_LOWER in config.age_groups
        assert ContentType.STORY in config.content_types
        assert ContentType.VOCABULARY in config.content_types
    
    @patch('subprocess.run')
    def test_launch_training(self, mock_subprocess):
        """Test training launch functionality."""
        mock_subprocess.return_value = Mock(returncode=0)
        
        success = self.launcher.launch_training(mode="interactive")
        
        assert success is True
        mock_subprocess.assert_called_once()
        
        # Verify training command includes educational parameters
        call_args = mock_subprocess.call_args[0][0]
        assert "train" in " ".join(call_args)


class TestEducationalTemplates:
    """Test educational content template system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        self.launcher = ToaripiLauncher(self.config_path)
    
    def test_story_template_generation(self):
        """Test story template generation."""
        templates = self.launcher.get_educational_templates(ContentType.STORY)
        
        assert len(templates) > 0
        assert AgeGroup.PRIMARY_LOWER in templates
        assert AgeGroup.PRIMARY_upper in templates
        
        # Test age-appropriate content
        primary_template = templates[AgeGroup.PRIMARY_LOWER]
        assert "simple" in primary_template.lower()
    
    def test_vocabulary_template_generation(self):
        """Test vocabulary template generation."""
        templates = self.launcher.get_educational_templates(ContentType.VOCABULARY)
        
        assert len(templates) > 0
        for age_group, template in templates.items():
            assert "vocabulary" in template.lower()
            assert "toaripi" in template.lower()
    
    def test_cultural_content_templates(self):
        """Test cultural content in templates."""
        story_prompts = self.launcher.get_story_prompts()
        
        assert AgeGroup.PRIMARY_upper in story_prompts
        
        # Check for cultural relevance
        primary_stories = story_prompts[AgeGroup.PRIMARY_upper]
        cultural_themes = ["fishing", "family", "village", "tradition"]
        
        story_text = " ".join(primary_stories).lower()
        cultural_found = any(theme in story_text for theme in cultural_themes)
        assert cultural_found is True
    
    def test_age_appropriate_content(self):
        """Test age-appropriate content generation."""
        # Early childhood content
        early_templates = self.launcher.get_educational_templates(
            ContentType.STORY, 
            age_group=AgeGroup.EARLY_CHILDHOOD
        )
        
        early_content = " ".join(early_templates.values()).lower()
        assert "simple" in early_content
        assert "short" in early_content
        
        # Secondary content
        secondary_templates = self.launcher.get_educational_templates(
            ContentType.STORY,
            age_group=AgeGroup.SECONDARY
        )
        
        secondary_content = " ".join(secondary_templates.values()).lower()
        assert "complex" in secondary_content or "advanced" in secondary_content


class TestLauncherIntegration:
    """Test launcher integration workflows."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "integration_config.yaml"
    
    def test_full_launcher_workflow(self):
        """Test complete launcher workflow."""
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value = Mock(returncode=0)
            
            launcher = ToaripiLauncher(self.config_path)
            
            # 1. System validation
            with patch.object(launcher.validator, 'validate_all') as mock_validate:
                mock_validate.return_value = ValidationResult(
                    is_valid=True, 
                    issues=[],
                    python_version="3.11"
                )
                
                validation_result = launcher.validate_system()
                assert validation_result.is_valid is True
            
            # 2. Educational setup
            launcher.setup_educational_mode(
                age_group=AgeGroup.PRIMARY_LOWER,
                content_types=[ContentType.STORY]
            )
            
            assert AgeGroup.PRIMARY_upper in launcher.config.educational.age_groups
            
            # 3. Launch training
            success = launcher.launch_training(mode="beginner")
            assert success is True
    
    def test_error_handling_workflow(self):
        """Test error handling in launcher workflow."""
        launcher = ToaripiLauncher(self.config_path)
        
        # Simulate validation failure
        with patch.object(launcher.validator, 'validate_all') as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        component="python",
                        issue="Version too old",
                        severity="error",
                        fix_suggestion="Upgrade Python"
                    )
                ]
            )
            
            validation_result = launcher.validate_system()
            assert validation_result.is_valid is False
            assert len(validation_result.issues) == 1
    
    def test_configuration_persistence(self):
        """Test configuration persistence across sessions."""
        # Create launcher and modify config
        launcher1 = ToaripiLauncher(self.config_path)
        launcher1.config.educational.max_content_length = 512
        launcher1.config_manager.save_config()
        
        # Create new launcher instance
        launcher2 = ToaripiLauncher(self.config_path)
        
        assert launcher2.config.educational.max_content_length == 512


class TestEducationalContentValidation:
    """Test educational content validation system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "validation_config.yaml"
        self.launcher = ToaripiLauncher(self.config_path)
    
    def test_age_appropriate_validation(self):
        """Test age-appropriate content validation."""
        # Valid content for primary students
        content = "Children help their parents with daily tasks"
        is_valid = self.launcher.validate_content_for_age(
            content, 
            AgeGroup.PRIMARY_LOWER
        )
        assert is_valid is True
        
        # Invalid content for young children
        complex_content = "Philosophical discourse on existential matters"
        is_valid = self.launcher.validate_content_for_age(
            complex_content,
            AgeGroup.EARLY_CHILDHOOD
        )
        assert is_valid is False
    
    def test_cultural_sensitivity_validation(self):
        """Test cultural sensitivity validation."""
        # Culturally appropriate content
        cultural_content = "Families gather for traditional fishing"
        is_valid = self.launcher.validate_cultural_sensitivity(cultural_content)
        assert is_valid is True
        
        # Potentially inappropriate content
        inappropriate_content = "Religious conversion stories"
        is_valid = self.launcher.validate_cultural_sensitivity(inappropriate_content)
        assert is_valid is False
    
    def test_educational_value_validation(self):
        """Test educational value validation."""
        # High educational value
        educational_content = "Learn new Toaripi words for family members"
        score = self.launcher.assess_educational_value(educational_content)
        assert score > 0.7  # High educational value
        
        # Low educational value
        non_educational = "Random meaningless text content"
        score = self.launcher.assess_educational_value(non_educational)
        assert score < 0.3  # Low educational value
    
    def test_language_preservation_validation(self):
        """Test language preservation focus validation."""
        # Content that supports language preservation
        preservation_content = "Traditional Toaripi stories passed down"
        is_supportive = self.launcher.supports_language_preservation(
            preservation_content
        )
        assert is_supportive is True
        
        # Content that doesn't support preservation
        non_supportive = "Modern technology concepts only"
        is_supportive = self.launcher.supports_language_preservation(
            non_supportive
        )
        assert is_supportive is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])