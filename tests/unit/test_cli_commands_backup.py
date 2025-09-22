"""
Unit tests for CLI commands and educational content validation.

Tests all major CLI functional    def test_data_validate_missing_file(self):
        """Test data validate with missing file."""
        runner = CliRunner()
        result = runner.invoke(cli, ['data', 'validate', '--file', 'nonexistent.csv'])
        assert result.exit_code != 0
        assert 'does not exist' in result.outputcluding:
- Data management commands
- Model operations commands  
- Serve/deployment commands
- Training commands
- Educational content validation
"""

import pytest
import json
import tempfile
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Import CLI modules to test
from src.toaripi_slm.cli.main import cli
from src.toaripi_slm.cli.data import data
from src.toaripi_slm.cli.model import model
from src.toaripi_slm.cli.serve import serve
from src.toaripi_slm.cli.train import train


class TestCLIBasics:
    """Test basic CLI functionality."""
    
    def test_main_cli_help(self):
        """Test main CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Toaripi SLM' in result.output
        assert 'Educational Content Generation' in result.output
    
    def test_main_cli_version(self):
        """Test version command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert '0.1.0' in result.output
    
    def test_status_command(self):
        """Test status command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['status'])
        assert result.exit_code == 0
        assert 'System Status' in result.output
        assert 'Educational Content Settings' in result.output
    
    def test_validate_command(self):
        """Test basic validation command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['validate', '--check', 'python'])
        assert result.exit_code == 0
        assert 'Python Environment' in result.output


class TestDataCommands:
    """Test data management CLI commands."""
    
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
            sample_file = temp_path / 'test_data.csv'
            
            # Create sample parallel data
            df = pd.DataFrame({
                'english': ['Hello world', 'How are you?'],
                'toaripi': ['Test toaripi 1', 'Test toaripi 2'],
                'verse_id': ['GEN.1.1', 'GEN.1.2'],
                'book': ['Genesis', 'Genesis'],
                'chapter': [1, 1]
            })
            df.to_csv(sample_file, index=False)
            
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
                'english': ['Test sentence'],
                'toaripi': ['Test toaripi'],
                'verse_id': ['TEST.1.1'],
                'book': ['Test'],
                'chapter': [1]
            })
            df.to_csv(input_file, index=False)
            
            result = runner.invoke(cli, [
                'data', 'prepare', 
                '--input', str(input_file),
                '--output', str(temp_path / 'output.csv'),
                '--dry-run'
            ])
            assert result.exit_code == 0
            assert 'DRY RUN MODE' in result.output
    
    def test_data_convert_format(self):
        """Test data format conversion."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create source CSV
            source_file = temp_path / 'source.csv'
            df = pd.DataFrame({
                'english': ['Test'],
                'toaripi': ['Test toaripi']
            })
            df.to_csv(source_file, index=False)
            
            result = runner.invoke(cli, [
                'data', 'convert',
                '--input', str(source_file),
                '--output', str(temp_path / 'output.json'),
                '--to-format', 'json'
            ])
            assert result.exit_code == 0
            assert 'Conversion completed' in result.output or 'DRY RUN' in result.output


class TestModelCommands:
    """Test model management CLI commands."""
    
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
    
    def test_model_list_with_files(self):
        """Test model list with sample model files."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy model files
            (temp_path / 'test_model.gguf').write_text('dummy gguf content')
            (temp_path / 'pytorch_model.bin').write_text('dummy pytorch content')
            
            result = runner.invoke(cli, ['model', 'list', '--directory', temp_dir])
            assert result.exit_code == 0
            assert 'test_model.gguf' in result.output or 'pytorch_model.bin' in result.output
    
    def test_model_export_missing_model(self):
        """Test model export with missing model."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'export',
            '--model', 'nonexistent_model',
            '--dry-run'
        ])
        assert result.exit_code != 0
        assert 'not found' in result.output
    
    def test_model_info_missing_model(self):
        """Test model info with missing model."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'info',
            '--model', 'nonexistent_model'
        ])
        assert result.exit_code != 0
        assert 'not found' in result.output
    
    def test_model_test_missing_model(self):
        """Test model test with missing model."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'test',
            '--model', 'nonexistent_model'
        ])
        assert result.exit_code != 0
        assert 'not found' in result.output


class TestServeCommands:
    """Test serving and deployment CLI commands."""
    
    def test_serve_help(self):
        """Test serve command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['serve', '--help'])
        assert result.exit_code == 0
        assert 'Serving and deployment commands' in result.output
    
    def test_serve_start_missing_model(self):
        """Test serve start with missing model."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            'serve', 'start',
            '--model', 'nonexistent_model',
            '--dry-run'
        ])
        assert result.exit_code != 0
        assert 'not found' in result.output
    
    def test_serve_status(self):
        """Test serve status command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['serve', 'status'])
        assert result.exit_code == 0
        assert 'Server Status' in result.output
    
    def test_serve_stop_no_server(self):
        """Test serve stop with no running server."""
        runner = CliRunner()
        result = runner.invoke(cli, ['serve', 'stop'])
        # Should handle gracefully
        assert 'No server information found' in result.output or 'stopped' in result.output
    
    @patch('src.toaripi_slm.cli.serve._test_server_endpoint')
    def test_serve_test(self, mock_test):
        """Test serve test command."""
        mock_test.return_value = {'success': True, 'response_time': 0.1}
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'serve', 'test',
            '--server', 'http://localhost:8000',
            '--count', '1'
        ])
        # Should complete without error (though server might not be running)
        assert result.exit_code == 0 or 'Health check failed' in result.output


class TestTrainCommands:
    """Test training CLI commands."""
    
    def test_train_help(self):
        """Test train command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['train', '--help'])
        assert result.exit_code == 0
        assert 'Training operations for Toaripi educational models' in result.output
    
    def test_train_start_missing_data(self):
        """Test train start with missing data."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            'train', 'start',
            '--data', 'nonexistent_data.csv',
            '--dry-run'
        ])
        assert result.exit_code != 0
        assert 'does not exist' in result.output
    
    def test_train_monitor_no_session(self):
        """Test train monitor with no active session."""
        runner = CliRunner()
        result = runner.invoke(cli, ['train', 'monitor'])
        assert result.exit_code == 0  # Should handle gracefully
        assert 'No active training' in result.output or 'Training Status' in result.output
    
    def test_train_status_no_session(self):
        """Test train status with no active session."""
        runner = CliRunner()
        result = runner.invoke(cli, ['train', 'status'])
        assert result.exit_code == 0  # Should handle gracefully
        assert 'No active training' in result.output
    
    def test_train_stop_no_session(self):
        """Test train stop with no active session."""
        runner = CliRunner()
        result = runner.invoke(cli, ['train', 'stop'])
        assert result.exit_code == 0  # Should handle gracefully
        assert 'No active training' in result.output or 'stopped' in result.output


class TestEducationalValidation:
    """Test educational content validation functionality."""
    
    def test_data_validate_educational_content(self):
        """Test educational content validation in data commands."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test data with educational content
            test_file = temp_path / 'educational_test.csv'
            df = pd.DataFrame({
                'english': [
                    'Children learn to help their families with daily tasks.',
                    'Young people should respect their elders and listen carefully.',
                    'Stories teach us important lessons about life and community.'
                ],
                'toaripi': [
                    'Test toaripi content 1',
                    'Test toaripi content 2', 
                    'Test toaripi content 3'
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
    
    def test_inappropriate_content_detection(self):
        """Test detection of inappropriate content."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test data with potentially inappropriate content
            test_file = temp_path / 'inappropriate_test.csv'
            df = pd.DataFrame({
                'english': [
                    'Violence is never the answer to problems.',  # Contains 'violence'
                    'Adult supervision required for this activity.',  # Contains 'adult'
                ],
                'toaripi': [
                    'Test toaripi content 1',
                    'Test toaripi content 2'
                ],
                'verse_id': ['TEST.1.1', 'TEST.1.2'],
                'book': ['Test', 'Test'],
                'chapter': [1, 1]
            })
            df.to_csv(test_file, index=False)
            
            result = runner.invoke(cli, [
                'data', 'validate',
                '--file', str(test_file)
            ])
            # Should complete validation (implementation may flag content)
            assert result.exit_code == 0


class TestIntegrationWorkflows:
    """Test end-to-end CLI workflows."""
    
    def test_full_data_pipeline(self):
        """Test complete data processing pipeline."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 1: Create sample raw data
            raw_file = temp_path / 'raw_data.csv'
            df = pd.DataFrame({
                'english': ['Hello world', 'How are you today?'],
                'toaripi': ['Test toaripi 1', 'Test toaripi 2'],
                'verse_id': ['GEN.1.1', 'GEN.1.2'],
                'book': ['Genesis', 'Genesis'],
                'chapter': [1, 1]
            })
            df.to_csv(raw_file, index=False)
            
            # Step 2: List data files
            result = runner.invoke(cli, ['data', 'list', '--directory', str(temp_path)])
            assert result.exit_code == 0
            assert 'raw_data.csv' in result.output
            
            # Step 3: Validate data
            result = runner.invoke(cli, ['data', 'validate', '--input', str(raw_file)])
            assert result.exit_code == 0
            
            # Step 4: Prepare data (dry run)
            prepared_file = temp_path / 'prepared_data.csv'
            result = runner.invoke(cli, [
                'data', 'prepare',
                '--input', str(raw_file),
                '--output', str(prepared_file),
                '--dry-run'
            ])
            assert result.exit_code == 0
            assert 'DRY RUN' in result.output
    
    def test_model_management_workflow(self):
        """Test model management workflow."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy model file
            model_file = temp_path / 'test_model.gguf'
            model_file.write_text('dummy model content for testing')
            
            # List models
            result = runner.invoke(cli, ['model', 'list', '--directory', str(temp_path)])
            assert result.exit_code == 0
            assert 'test_model.gguf' in result.output
            
            # Get model info
            result = runner.invoke(cli, ['model', 'info', '--model', str(model_file)])
            assert result.exit_code == 0
            assert 'Model Information' in result.output
    
    def test_error_handling(self):
        """Test error handling across commands."""
        runner = CliRunner()
        
        # Test with invalid paths
        commands_to_test = [
            ['data', 'validate', '--input', 'nonexistent.csv'],
            ['model', 'info', '--model', 'nonexistent_model'],
            ['train', 'start', '--data', 'nonexistent_data.csv', '--dry-run'],
            ['serve', 'start', '--model', 'nonexistent_model', '--dry-run']
        ]
        
        for cmd in commands_to_test:
            result = runner.invoke(cli, cmd)
            # All should handle errors gracefully (non-zero exit or error message)
            assert result.exit_code != 0 or 'not found' in result.output or 'Error' in result.output


class TestConfigurationAndSettings:
    """Test CLI configuration and educational settings."""
    
    def test_verbose_mode(self):
        """Test verbose mode across commands."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--verbose', 'status'])
        assert result.exit_code == 0
        # Verbose mode should work without errors
    
    def test_quiet_mode(self):
        """Test quiet mode."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--quiet', 'status'])
        assert result.exit_code == 0
        # Quiet mode should suppress non-error output
    
    def test_educational_mode_flags(self):
        """Test educational mode flags in serve commands."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_file = temp_path / 'test.gguf'
            model_file.write_text('dummy')
            
            result = runner.invoke(cli, [
                'serve', 'start',
                '--model', str(model_file),
                '--educational-mode',
                '--cultural-validation',
                '--age-filtering',
                '--dry-run'
            ])
            # Should show configuration with all educational features enabled
            assert result.exit_code == 0 or 'not found' in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])