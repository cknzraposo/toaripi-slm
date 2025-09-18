"""Test suite for verifying Toaripi SLM installation and setup."""

import pytest
import pandas as pd
import yaml
import json
from pathlib import Path
import sys
import importlib


class TestInstallation:
    """Test basic installation and setup."""
    
    def test_python_version(self):
        """Test that Python version is 3.10+."""
        assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version_info}"
    
    def test_package_import(self):
        """Test that main package can be imported."""
        try:
            import src.toaripi_slm
            assert hasattr(src.toaripi_slm, '__version__')
            assert hasattr(src.toaripi_slm, 'PACKAGE_INFO')
        except ImportError as e:
            pytest.fail(f"Failed to import toaripi_slm package: {e}")
    
    def test_core_dependencies(self):
        """Test that core ML dependencies are available."""
        required_packages = [
            'torch',
            'transformers', 
            'datasets',
            'accelerate',
            'peft',
            'pandas',
            'numpy',
            'fastapi',
            'uvicorn',
            'yaml',
            'tqdm'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
        
        assert not missing_packages, f"Missing required packages: {missing_packages}"


class TestProjectStructure:
    """Test project directory structure and files."""
    
    def test_required_directories(self):
        """Test that all required directories exist."""
        required_dirs = [
            'src/toaripi_slm',
            'src/toaripi_slm/core',
            'src/toaripi_slm/data', 
            'src/toaripi_slm/inference',
            'src/toaripi_slm/utils',
            'configs/data',
            'configs/training',
            'data/samples',
            'tests/unit',
            'tests/integration'
        ]
        
        for dir_path in required_dirs:
            assert Path(dir_path).exists(), f"Required directory missing: {dir_path}"
    
    def test_config_files(self):
        """Test that configuration files exist and are valid."""
        config_files = [
            'configs/data/preprocessing_config.yaml',
            'configs/training/base_config.yaml',
            'configs/training/lora_config.yaml'
        ]
        
        for config_file in config_files:
            config_path = Path(config_file)
            assert config_path.exists(), f"Config file missing: {config_file}"
            
            # Test YAML validity
            with open(config_path) as f:
                try:
                    yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {config_file}: {e}")
    
    def test_requirements_files(self):
        """Test that requirements files exist."""
        req_files = ['requirements.txt', 'requirements-dev.txt']
        for req_file in req_files:
            assert Path(req_file).exists(), f"Requirements file missing: {req_file}"


class TestSampleData:
    """Test sample data files and format."""
    
    def test_sample_parallel_data(self):
        """Test sample parallel data file."""
        sample_path = Path('data/samples/sample_parallel.csv')
        assert sample_path.exists(), "Sample parallel data file missing"
        
        # Load and validate data
        df = pd.read_csv(sample_path)
        assert len(df) > 0, "Sample data should not be empty"
        
        # Check required columns
        required_columns = ['english', 'toaripi', 'verse_id', 'book', 'chapter']
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Check data quality
        assert not df['english'].isna().any(), "English column should not have NaN values"
        assert not df['toaripi'].isna().any(), "Toaripi column should not have NaN values"
        assert all(len(text.strip()) > 0 for text in df['english']), "English text should not be empty"
        assert all(len(text.strip()) > 0 for text in df['toaripi']), "Toaripi text should not be empty"
    
    def test_educational_prompts(self):
        """Test educational prompts JSON file."""
        prompts_path = Path('data/samples/educational_prompts.json')
        assert prompts_path.exists(), "Educational prompts file missing"
        
        with open(prompts_path) as f:
            prompts = json.load(f)
        
        # Check structure
        assert 'story_prompts' in prompts, "Missing story_prompts section"
        assert 'vocabulary_prompts' in prompts, "Missing vocabulary_prompts section"
        assert 'comprehension_prompts' in prompts, "Missing comprehension_prompts section"
        
        # Check story prompts structure
        for prompt in prompts['story_prompts']:
            required_fields = ['id', 'topic', 'prompt', 'age_group', 'length']
            for field in required_fields:
                assert field in prompt, f"Missing field {field} in story prompt"


class TestFunctionality:
    """Test basic functionality."""
    
    def test_data_loading(self):
        """Test that sample data can be loaded."""
        df = pd.read_csv('data/samples/sample_parallel.csv')
        assert len(df) > 0, "Should be able to load sample data"
        
        # Test data access
        first_row = df.iloc[0]
        assert isinstance(first_row['english'], str), "English text should be string"
        assert isinstance(first_row['toaripi'], str), "Toaripi text should be string"
    
    def test_config_loading(self):
        """Test that configurations can be loaded."""
        config_path = Path('configs/data/preprocessing_config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert 'data_sources' in config, "Config should have data_sources section"
        assert 'preprocessing' in config, "Config should have preprocessing section"
        assert 'output' in config, "Config should have output section"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])