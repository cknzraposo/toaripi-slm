"""
Integration tests for end-to-end CLI workflows.

Tests complete workflows including:
- Data preparation and validation
- Model training and evaluation
- Serving and deployment
- Educational content generation
"""

import pytest
import tempfile
import json
from pathlib import Path
from click.testing import CliRunner
import pandas as pd

from src.toaripi_slm.cli.main import cli


class TestDataWorkflow:
    """Test complete data processing workflow."""
    
    def test_data_preparation_workflow(self):
        """Test complete data preparation from raw to training-ready."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create realistic sample data
            raw_data = temp_path / 'raw_bible.csv'
            sample_data = pd.DataFrame({
                'english': [
                    'In the beginning was the Word, and the Word was with God.',
                    'And the Word was God.',
                    'All things were made through him.',
                    'Children, obey your parents in the Lord.',
                    'Honor your father and mother.',
                    'Love your neighbor as yourself.',
                    'The children gathered around the teacher.',
                    'They listened carefully to the stories.',
                    'Each story taught them about kindness and respect.'
                ],
                'toaripi': [
                    'Aivaravo itiana era ki Biromakae, katauaina Biromakae era ki Maoatali.',
                    'Katauaina Biromakae era ki Maoatali.',
                    'Kauropa iaria avori era umuri.',
                    'Ikevavo vareve, kauropa umua kakou tumararai.',
                    'Kauropa umua mama katauaina tataia.',
                    'Kauropa ara umua nauna avori.',
                    'Ikevavo vareve iaria kea kona kemu.',
                    'Iaria tumararai kemu maia katau kiau.',
                    'Kiau kemu avori aiaku tani katauaina maia.'
                ],
                'verse_id': [
                    'JHN.1.1', 'JHN.1.1', 'JHN.1.3',
                    'EPH.6.1', 'EPH.6.2', 'MAT.22.39',
                    'EDU.1.1', 'EDU.1.2', 'EDU.1.3'
                ],
                'book': [
                    'John', 'John', 'John',
                    'Ephesians', 'Ephesians', 'Matthew',
                    'Education', 'Education', 'Education'
                ],
                'chapter': [1, 1, 1, 6, 6, 22, 1, 1, 1]
            })
            sample_data.to_csv(raw_data, index=False)
            
            # Step 1: List and validate raw data
            result = runner.invoke(cli, ['data', 'list', '--directory', str(temp_path)])
            assert result.exit_code == 0
            assert 'raw_bible.csv' in result.output
            
            # Step 2: Validate educational appropriateness
            result = runner.invoke(cli, [
                'data', 'validate', 
                '--input', str(raw_data),
                '--educational-check'
            ])
            assert result.exit_code == 0
            
            # Step 3: Prepare training data
            train_data = temp_path / 'train_data.csv'
            result = runner.invoke(cli, [
                'data', 'prepare',
                '--input', str(raw_data),
                '--output', str(train_data),
                '--max-length', '100',
                '--educational-focus',
                '--dry-run'
            ])
            assert result.exit_code == 0
            assert 'Educational Content Processing' in result.output
            
            # Step 4: Convert to different format
            json_data = temp_path / 'data.json'
            result = runner.invoke(cli, [
                'data', 'convert',
                '--input', str(raw_data),
                '--output', str(json_data),
                '--format', 'json',
                '--dry-run'
            ])
            assert result.exit_code == 0


class TestTrainingWorkflow:
    """Test model training workflow."""
    
    def test_training_setup_and_monitoring(self):
        """Test training setup and monitoring workflow."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create training data
            train_data = temp_path / 'train.csv'
            sample_data = pd.DataFrame({
                'english': [
                    'Tell me a story about children helping their families.',
                    'Create a vocabulary lesson about family members.',
                    'Generate a dialogue between teacher and student.'
                ],
                'toaripi': [
                    'Aiaku tani ikevavo vareve umua naua.',
                    'Kemu maia katau naua avori.',
                    'Kemu katauaina kea tumararai.'
                ],
                'content_type': ['story', 'vocabulary', 'dialogue'],
                'age_group': ['primary_lower', 'primary_lower', 'primary_upper']
            })
            sample_data.to_csv(train_data, index=False)
            
            # Test training start (dry run)
            result = runner.invoke(cli, [
                'train', 'start',
                '--data', str(train_data),
                '--model', 'microsoft/DialoGPT-small',
                '--mode', 'lora',
                '--educational-focus',
                '--max-epochs', '2',
                '--dry-run'
            ])
            assert result.exit_code == 0
            assert 'Educational Training Configuration' in result.output
            
            # Test training monitoring
            result = runner.invoke(cli, ['train', 'monitor'])
            assert result.exit_code == 0
            # Should handle no active training gracefully
            
            # Test training evaluation (would fail for missing model)
            result = runner.invoke(cli, [
                'train', 'evaluate',
                '--model', 'nonexistent_model',
                '--test-data', str(train_data)
            ])
            assert result.exit_code != 0  # Expected to fail with missing model


class TestModelManagementWorkflow:
    """Test model management workflow."""
    
    def test_model_lifecycle(self):
        """Test complete model lifecycle management."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            models_dir = temp_path / 'models'
            models_dir.mkdir()
            
            # Create dummy model files for testing
            hf_model_dir = models_dir / 'toaripi-educational-v1'
            hf_model_dir.mkdir()
            
            # Create HuggingFace model structure
            (hf_model_dir / 'config.json').write_text(json.dumps({
                'model_type': 'gpt2',
                'vocab_size': 50257,
                'n_positions': 1024,
                'n_embd': 768,
                'n_layer': 12,
                'n_head': 12
            }))
            (hf_model_dir / 'pytorch_model.bin').write_text('dummy model weights')
            (hf_model_dir / 'tokenizer.json').write_text('dummy tokenizer')
            
            # Create GGUF model
            gguf_model = models_dir / 'toaripi-educational-q4.gguf'
            gguf_model.write_text('dummy gguf model content' * 1000)  # Make it larger
            
            # Test model listing
            result = runner.invoke(cli, ['model', 'list', '--directory', str(models_dir)])
            assert result.exit_code == 0
            assert 'toaripi-educational-v1' in result.output or 'toaripi-educational-q4.gguf' in result.output
            
            # Test model info
            result = runner.invoke(cli, [
                'model', 'info', 
                '--model', str(hf_model_dir),
                '--show-config',
                '--educational-check'
            ])
            assert result.exit_code == 0
            assert 'Model Information' in result.output
            
            # Test model export (dry run)
            result = runner.invoke(cli, [
                'model', 'export',
                '--model', str(hf_model_dir),
                '--output', str(models_dir / 'exported'),
                '--quantization', 'q4_k_m',
                '--device-target', 'raspberry_pi',
                '--educational-validation',
                '--dry-run'
            ])
            assert result.exit_code == 0
            assert 'Export Configuration' in result.output
            
            # Test model testing
            result = runner.invoke(cli, [
                'model', 'test',
                '--model', str(gguf_model),
                '--prompt', 'Tell me a story about Toaripi children',
                '--content-type', 'story',
                '--age-group', 'primary_lower'
            ])
            assert result.exit_code == 0
            assert 'Testing Educational Content Generation' in result.output


class TestServingWorkflow:
    """Test serving and deployment workflow."""
    
    def test_server_lifecycle(self):
        """Test complete server lifecycle management."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy model for serving
            model_file = temp_path / 'toaripi-server.gguf'
            model_file.write_text('dummy model for serving' * 100)
            
            # Test server start (dry run)
            result = runner.invoke(cli, [
                'serve', 'start',
                '--model', str(model_file),
                '--host', '127.0.0.1',
                '--port', '8001',
                '--educational-mode',
                '--cultural-validation',
                '--age-filtering',
                '--cpu-only',
                '--dry-run'
            ])
            assert result.exit_code == 0
            assert 'Server Configuration' in result.output
            assert 'Educational Content Endpoints' in result.output
            
            # Test server status
            result = runner.invoke(cli, ['serve', 'status', '--port', '8001'])
            assert result.exit_code == 0
            assert 'Server Status' in result.output
            
            # Test server testing
            result = runner.invoke(cli, [
                'serve', 'test',
                '--server', 'http://127.0.0.1:8001',
                '--prompt', 'Generate educational content for primary students',
                '--content-type', 'story',
                '--count', '1'
            ])
            # Server won't be running, but command should handle gracefully
            assert result.exit_code == 0 or 'Health check failed' in result.output
            
            # Test server stop
            result = runner.invoke(cli, ['serve', 'stop', '--port', '8001'])
            assert result.exit_code == 0  # Should handle no server gracefully


class TestEducationalContentValidation:
    """Test educational content validation across workflows."""
    
    def test_educational_content_pipeline(self):
        """Test educational content validation throughout pipeline."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create educational content samples
            educational_data = pd.DataFrame({
                'english': [
                    'Children learn by playing together and sharing toys.',
                    'Teachers help students understand new concepts through stories.',
                    'Families work together to build strong communities.',
                    'Respect for elders is an important value in our culture.',
                    'Young people can contribute to preserving our language.'
                ],
                'toaripi': [
                    'Ikevavo vareve tumararai kareva katauaina mauva.',
                    'Kemu aiaku ikevavo vareve tumararai katau kiau.',
                    'Naua avori umuarai kariva kemu kiari.',
                    'Maia aukava era kiri katauaina avori kemu.',
                    'Kiari vareve avua kemu katau kiani aiaku.'
                ],
                'content_type': ['story', 'lesson', 'cultural', 'cultural', 'lesson'],
                'age_group': ['primary_lower', 'primary_upper', 'secondary', 'all', 'secondary'],
                'cultural_appropriate': [True, True, True, True, True],
                'educational_value': [0.9, 0.95, 0.85, 0.92, 0.88]
            })
            
            educational_file = temp_path / 'educational_content.csv'
            educational_data.to_csv(educational_file, index=False)
            
            # Test educational validation
            result = runner.invoke(cli, [
                'data', 'validate',
                '--input', str(educational_file),
                '--educational-check',
                '--cultural-check',
                '--age-appropriate-check'
            ])
            assert result.exit_code == 0
            assert 'Educational validation' in result.output or 'text pairs processed' in result.output
            
            # Test data preparation with educational focus
            result = runner.invoke(cli, [
                'data', 'prepare',
                '--input', str(educational_file),
                '--output', str(temp_path / 'prepared_educational.csv'),
                '--educational-focus',
                '--cultural-preservation',
                '--age-appropriate-filter',
                '--dry-run'
            ])
            assert result.exit_code == 0
            assert 'Educational Content Processing' in result.output


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_invalid_file_formats(self):
        """Test handling of invalid file formats."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create invalid CSV file
            invalid_csv = temp_path / 'invalid.csv'
            invalid_csv.write_text('invalid,csv,content\nno,proper,headers')
            
            result = runner.invoke(cli, [
                'data', 'validate',
                '--input', str(invalid_csv)
            ])
            # Should handle gracefully
            assert result.exit_code == 0 or 'error' in result.output.lower()
    
    def test_large_file_handling(self):
        """Test handling of large data files."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create large dataset
            large_data = pd.DataFrame({
                'english': [f'This is test sentence number {i}.' for i in range(1000)],
                'toaripi': [f'Test toaripi sentence {i}.' for i in range(1000)],
                'verse_id': [f'TEST.{i//10}.{i%10}' for i in range(1000)],
                'book': ['Test'] * 1000,
                'chapter': [i//10 for i in range(1000)]
            })
            
            large_file = temp_path / 'large_data.csv'
            large_data.to_csv(large_file, index=False)
            
            # Test processing large file
            result = runner.invoke(cli, [
                'data', 'validate',
                '--input', str(large_file),
                '--batch-size', '100'
            ])
            assert result.exit_code == 0
            assert 'text pairs processed' in result.output
    
    def test_concurrent_operations(self):
        """Test concurrent CLI operations."""
        runner = CliRunner()
        
        # Test multiple status checks
        results = []
        for i in range(3):
            result = runner.invoke(cli, ['status'])
            results.append(result)
        
        # All should succeed
        for result in results:
            assert result.exit_code == 0
    
    def test_cleanup_after_errors(self):
        """Test cleanup after command errors."""
        runner = CliRunner()
        
        # Run commands that will fail
        failing_commands = [
            ['data', 'validate', '--input', 'nonexistent.csv'],
            ['model', 'info', '--model', 'nonexistent_model'],
            ['train', 'start', '--data', 'nonexistent.csv', '--dry-run']
        ]
        
        for cmd in failing_commands:
            result = runner.invoke(cli, cmd)
            # Commands should fail gracefully without corrupting state
            assert result.exit_code != 0 or 'not found' in result.output
        
        # System should still be functional
        result = runner.invoke(cli, ['status'])
        assert result.exit_code == 0


class TestCulturalSensitivity:
    """Test cultural sensitivity features."""
    
    def test_cultural_validation_workflow(self):
        """Test cultural validation throughout the workflow."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create culturally sensitive content
            cultural_data = pd.DataFrame({
                'english': [
                    'Traditional fishing methods passed down through generations.',
                    'Elders share stories around the evening fire.',
                    'Children learn language through songs and games.',
                    'Community celebrations bring families together.'
                ],
                'toaripi': [
                    'Aukava kemu haro kiari maia katauaina.',
                    'Aukava kiari aiaku tani kemu maia.',
                    'Ikevavo vareve tumararai katau kiau kareva.',
                    'Kemu kiari naua avua umuarai kariva.'
                ],
                'cultural_context': ['fishing', 'storytelling', 'education', 'celebration'],
                'cultural_appropriate': [True, True, True, True]
            })
            
            cultural_file = temp_path / 'cultural_content.csv'
            cultural_data.to_csv(cultural_file, index=False)
            
            # Test cultural validation
            result = runner.invoke(cli, [
                'data', 'validate',
                '--input', str(cultural_file),
                '--cultural-check'
            ])
            assert result.exit_code == 0
            
            # Test serving with cultural validation
            model_file = temp_path / 'cultural_model.gguf'
            model_file.write_text('cultural model content')
            
            result = runner.invoke(cli, [
                'serve', 'start',
                '--model', str(model_file),
                '--cultural-validation',
                '--dry-run'
            ])
            assert result.exit_code == 0
            assert 'Cultural Validation' in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])