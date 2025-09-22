"""
Unit tests for CLI commands and educational content validation.

Tests all major CLI functionality including:
- Data management commands
- Model operations commands  
- Training commands
- Serving commands
- Educational content validation
"""

import os
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest
from click.testing import CliRunner

import pandas as pd

from src.toaripi_slm.cli.main import cli


class TestCLIBasics:
    """Test basic CLI functionality."""
    
    def test_main_cli_help(self):
        """Test main CLI help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Toaripi SLM - Educational Content Generation System' in result.output
    
    def test_main_cli_version(self):
        """Test CLI version display.""" 
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert '0.1.0' in result.output or 'version' in result.output.lower()
    
    def test_status_command(self):
        """Test status command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['status'])
        assert result.exit_code == 0
        assert 'System Status' in result.output or 'Status:' in result.output
    
    def test_validate_command(self):
        """Test validate command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['validate'])
        assert result.exit_code == 0
        assert 'Validating all configuration' in result.output or 'Validation Result:' in result.output


class TestDataCommands:
    """Test data management commands."""
    
    def test_data_help(self):
        """Test data command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['data', '--help'])
        assert result.exit_code == 0
        assert 'Data management commands for Toaripi educational content' in result.output
    
    def test_data_list_empty(self):
        """Test data list with no data files."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, ['data', 'list', '--directory', temp_dir])
            assert result.exit_code == 0
            assert 'No dataset files found' in result.output
    
    def test_data_list_with_files(self):
        """Test data list with sample files."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample CSV file
            temp_path = Path(temp_dir)
            test_file = temp_path / 'test_data.csv'
            
            df = pd.DataFrame({
                'english': ['Hello world', 'How are you?'],
                'toaripi': ['Halo dunia', 'Baimana yu?']
            })
            df.to_csv(test_file, index=False)
            
            result = runner.invoke(cli, ['data', 'list', '--directory', temp_dir])
            assert result.exit_code == 0
            assert 'test_data.csv' in result.output
            assert 'Found 1 dataset files' in result.output
    
    def test_data_validate_missing_file(self):
        """Test data validation with missing file."""
        runner = CliRunner()
        result = runner.invoke(cli, ['data', 'validate', '--file', 'nonexistent.csv'])
        assert result.exit_code != 0
        assert 'does not exist' in result.output
    
    def test_data_prepare_dry_run(self):
        """Test data preparation in dry run mode."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample input file
            temp_path = Path(temp_dir)
            input_file = temp_path / 'input.csv'
            
            df = pd.DataFrame({
                'english': ['Test educational content for young learners.'],
                'toaripi': ['Test educational mea pi young learners.'],
                'verse_id': ['EDU.1.1'],
                'book': ['Education'],
                'chapter': [1]
            })
            df.to_csv(input_file, index=False)
            
            result = runner.invoke(cli, [
                'data', 'prepare',
                '--input', str(input_file),
                '--output', str(temp_path / 'processed'),
                '--dry-run'
            ])
            assert result.exit_code == 0
            assert 'DRY RUN' in result.output or 'would process' in result.output


class TestModelCommands:
    """Test model management commands."""
    
    def test_model_help(self):
        """Test model command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['model', '--help'])
        assert result.exit_code == 0
        assert 'Model management commands' in result.output
    
    def test_model_list_empty(self):
        """Test model list with no models."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, ['model', 'list', '--directory', temp_dir])
            assert result.exit_code == 0
            assert 'No models found' in result.output


class TestServeCommands:
    """Test serving and deployment commands."""
    
    def test_serve_help(self):
        """Test serve command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['serve', '--help'])
        assert result.exit_code == 0
        assert 'Serving and deployment commands' in result.output
    
    def test_serve_status(self):
        """Test serve status."""
        runner = CliRunner()
        result = runner.invoke(cli, ['serve', 'status'])
        assert result.exit_code == 0
        assert 'Server Status' in result.output or 'Not running' in result.output


class TestTrainCommands:
    """Test training commands."""
    
    def test_train_help(self):
        """Test train command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['train', '--help'])
        assert result.exit_code == 0
        assert 'Training operations for Toaripi educational models' in result.output
    
    def test_train_status_no_session(self):
        """Test train status with no active session."""
        runner = CliRunner()
        result = runner.invoke(cli, ['train', 'status'])
        assert result.exit_code == 0  # Should handle gracefully
        assert 'No active training' in result.output or 'Training Status' in result.output


class TestEducationalValidation:
    """Test educational content validation features."""
    
    def test_data_validate_educational_content(self):
        """Test educational content validation."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / 'educational_test.csv'
            
            # Create appropriate educational content
            df = pd.DataFrame({
                'english': [
                    'Children learn to count from one to ten.',
                    'The small fish swims in the clear water.',
                    'Students practice reading simple stories.'
                ],
                'toaripi': [
                    'Ainama sisia haitapu ave herea mai ia gutpela ira.',
                    'Vada vada ia haia hua ia ia vovo au heia.',
                    'Siaia vovo ia parovoro simple stories.'
                ],
                'verse_id': ['EDU.1.1', 'EDU.1.2', 'EDU.1.3'],
                'book': ['Education', 'Education', 'Education'],
                'chapter': [1, 1, 1]
            })
            df.to_csv(test_file, index=False)
            
            result = runner.invoke(cli, [
                'data', 'validate',
                '--file', str(test_file),
                '--check-educational'
            ])
            assert result.exit_code == 0
            assert 'Educational validation' in result.output or 'text pairs processed' in result.output