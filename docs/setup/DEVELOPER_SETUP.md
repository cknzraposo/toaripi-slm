# Developer Setup Guide

## Development Environment Setup

### Prerequisites

- **Python**: 3.10+ (recommended: 3.11 or 3.13)
- **Git**: For version control
- **Virtual Environment**: For dependency isolation
- **System Requirements**: 
  - Memory: 8GB+ RAM (16GB recommended for training)
  - Storage: 10GB+ free space
  - GPU: Optional but recommended for training

### Repository Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd toaripi-slm
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```

4. **Verify installation:**
   ```bash
   toaripi-slm --version
   toaripi-slm validate
   ```

### Project Structure

```
toaripi-slm/
├── src/toaripi_slm/           # Main library code
│   ├── cli/                   # Command-line interface
│   │   ├── main.py           # Main CLI entry point
│   │   ├── data.py           # Data management commands
│   │   ├── model.py          # Model management commands
│   │   ├── train.py          # Training commands
│   │   └── serve.py          # Serving commands
│   ├── core/                  # Core functionality
│   │   ├── model.py          # Model operations
│   │   └── trainer.py        # Training logic
│   ├── data/                  # Data processing
│   │   ├── dataset.py        # Dataset handling
│   │   ├── preprocessing.py  # Data preprocessing
│   │   └── prompts.py        # Educational prompts
│   ├── models/               # Pydantic models
│   │   ├── config.py         # Configuration models
│   │   ├── dataset.py        # Dataset models
│   │   ├── checkpoint.py     # Training checkpoint models
│   │   └── session.py        # Training session models
│   └── inference/            # Model inference
├── app/                      # Web application (future)
├── configs/                  # Configuration files
│   ├── data/                 # Data processing configs
│   ├── training/             # Training configurations
│   └── deployment/           # Deployment configs
├── data/                     # Training and test data
│   ├── raw/                  # Raw parallel data
│   ├── processed/            # Processed training data
│   └── samples/              # Sample data for testing
├── models/                   # Trained models
│   ├── hf/                   # HuggingFace format models
│   └── gguf/                 # Quantized edge models
├── tests/                    # Test suites
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
└── docs/                     # Documentation
    ├── usage/                # User guides
    └── setup/                # Setup instructions
```

## Development Workflow

### Code Standards

- **Python Style**: Follow PEP 8 with Black formatting
- **Type Hints**: Use type hints throughout codebase
- **Docstrings**: Google-style docstrings for all public functions
- **Educational Focus**: All features must prioritize educational content

### Key Design Principles

1. **Educational First**: Every feature should enhance educational content generation
2. **Cultural Sensitivity**: Built-in validation for cultural appropriateness
3. **Age Appropriateness**: Automatic content filtering by age group
4. **Offline Capable**: Support for Raspberry Pi and offline deployment
5. **Teacher-Friendly**: CLI designed for non-technical educators

### Adding New Features

#### 1. Data Processing Features

Location: `src/toaripi_slm/data/`

Example - Adding new data format support:

```python
# In src/toaripi_slm/data/preprocessing.py

def load_usfm_data(file_path: Path) -> pd.DataFrame:
    """Load USFM format Bible data for educational processing."""
    # Implementation here
    pass

def validate_educational_content(df: pd.DataFrame, age_group: str) -> ValidationResult:
    """Validate content for educational appropriateness."""
    # Implementation here
    pass
```

#### 2. CLI Commands

Location: `src/toaripi_slm/cli/`

Example - Adding new model command:

```python
# In src/toaripi_slm/cli/model.py

@model.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--benchmark', is_flag=True, help='Run performance benchmarks')
def benchmark(model_path: str, benchmark: bool):
    """Benchmark model performance for educational content generation."""
    console.print(f"Benchmarking model: {model_path}")
    # Implementation here
```

#### 3. Educational Validation

Location: `src/toaripi_slm/core/`

Example - Adding new validation rules:

```python
# In src/toaripi_slm/core/validation.py

class EducationalValidator:
    def validate_age_appropriateness(self, content: str, age_group: AgeGroup) -> bool:
        """Validate content is appropriate for target age group."""
        # Implementation here
        pass
    
    def validate_cultural_sensitivity(self, content: str) -> ValidationResult:
        """Check cultural appropriateness of generated content."""
        # Implementation here
        pass
```

### Configuration Management

All configuration uses Pydantic models for validation:

```python
# Example configuration model
from pydantic import BaseModel, Field
from typing import List

class TrainingConfig(BaseModel):
    model_name: str = Field(default="microsoft/DialoGPT-small")
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=4, ge=1, le=32)
    learning_rate: float = Field(default=2e-5, gt=0)
    
    # Educational-specific settings
    target_age_groups: List[AgeGroup] = Field(default=[AgeGroup.PRIMARY_LOWER])
    content_types: List[ContentType] = Field(default=[ContentType.STORY])
    validation_level: ValidationLevel = Field(default=ValidationLevel.EDUCATIONAL)
```

### Testing

#### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=src/toaripi_slm --cov-report=html

# Run tests for specific functionality
pytest tests/unit/test_cli_commands.py::TestDataCommands
```

#### Writing Tests

Example test structure:

```python
# tests/unit/test_educational_validation.py

import pytest
from src.toaripi_slm.core.validation import EducationalValidator

class TestEducationalValidation:
    def setup_method(self):
        self.validator = EducationalValidator()
    
    def test_age_appropriate_content_primary_lower(self):
        """Test content validation for primary lower students."""
        content = "The little fish swims in the clear water."
        result = self.validator.validate_age_appropriateness(
            content, AgeGroup.PRIMARY_LOWER
        )
        assert result.is_valid
        assert result.age_score >= 0.8
    
    def test_cultural_sensitivity_validation(self):
        """Test cultural appropriateness validation."""
        content = "Children learn traditional fishing from their elders."
        result = self.validator.validate_cultural_sensitivity(content)
        assert result.is_culturally_appropriate
```

### Educational Content Guidelines

#### Age Group Specifications

**Early Childhood (3-5 years)**
- Vocabulary: 50-100 words
- Sentence length: 3-5 words
- Concepts: Basic colors, numbers, family, animals
- Cultural content: Simple traditions, family roles

**Primary Lower (6-8 years)**
- Vocabulary: 200-500 words  
- Sentence length: 5-10 words
- Concepts: Community, nature, basic science
- Cultural content: Traditional stories, customs

**Primary Upper (9-11 years)**
- Vocabulary: 500-1000 words
- Sentence length: 8-15 words
- Concepts: History, geography, advanced science
- Cultural content: Complex traditions, cultural values

#### Content Type Specifications

**Stories**
- Clear beginning, middle, end
- Age-appropriate plot complexity
- Cultural context integration
- Educational objectives

**Vocabulary**
- Thematic word groups
- Definitions in both languages
- Usage examples
- Cultural context

**Dialogues**
- Natural conversation patterns
- Age-appropriate scenarios
- Cultural communication styles
- Interactive elements

### Model Training Guidelines

#### Data Preparation

```python
# Example training data preparation
from src.toaripi_slm.data.preprocessing import prepare_educational_data

def prepare_training_data():
    """Prepare data with educational validation."""
    config = PreprocessingConfig(
        source_file="data/raw/parallel_data.csv",
        target_age_groups=[AgeGroup.PRIMARY_LOWER, AgeGroup.PRIMARY_UPPER],
        content_types=[ContentType.STORY, ContentType.VOCABULARY],
        validation_level=ValidationLevel.STRICT
    )
    
    processed_data = prepare_educational_data(config)
    return processed_data
```

#### Training Configuration

```yaml
# configs/training/educational_config.yaml
model:
  base_model: "microsoft/DialoGPT-small"
  use_lora: true
  lora_rank: 16
  lora_alpha: 32

training:
  epochs: 3
  batch_size: 4
  learning_rate: 0.00002
  warmup_steps: 100
  save_steps: 500

educational:
  validation_level: "strict"
  target_age_groups:
    - "primary_lower"
    - "primary_upper"
  content_types:
    - "story"
    - "vocabulary"
    - "dialogue"
  cultural_validation: true
  max_content_length: 512
```

### Deployment Guidelines

#### Edge Deployment (Raspberry Pi)

```python
# Example model export for edge deployment
from src.toaripi_slm.core.model import export_for_edge

def export_model_for_classroom():
    """Export model optimized for classroom tablets."""
    config = EdgeExportConfig(
        model_path="models/hf/toaripi-primary",
        target_device="raspberry-pi",
        quantization="q4_k_m",
        educational_validation=True,
        age_filter=AgeGroup.PRIMARY_LOWER
    )
    
    export_for_edge(config)
```

#### API Deployment

```python
# Example educational API setup
from src.toaripi_slm.inference.api import EducationalAPI

def setup_classroom_api():
    """Setup API for classroom use."""
    api = EducationalAPI(
        model_path="models/gguf/toaripi-classroom.gguf",
        age_filter=AgeGroup.PRIMARY_LOWER,
        cultural_validation=True,
        max_generation_length=200,
        temperature=0.7
    )
    
    return api
```

### Contributing Guidelines

#### Pull Request Process

1. **Create Feature Branch**: `git checkout -b feature/educational-validation`
2. **Write Tests**: Ensure all new functionality has tests
3. **Educational Validation**: Verify all content is educationally appropriate
4. **Cultural Review**: Check cultural sensitivity of generated content
5. **Documentation**: Update relevant documentation
6. **Test Suite**: Ensure all tests pass
7. **PR Review**: Submit for code and educational content review

#### Code Review Checklist

- [ ] Educational content appropriateness validated
- [ ] Cultural sensitivity checked
- [ ] Age group targeting implemented correctly
- [ ] Error handling for edge cases
- [ ] Documentation updated
- [ ] Tests cover new functionality
- [ ] Performance impact assessed
- [ ] Offline compatibility maintained

### Debugging and Troubleshooting

#### Common Development Issues

**Import Errors**
```bash
# Ensure package is installed in development mode
pip install -e .
```

**Test Failures**
```bash
# Run with verbose output
pytest -v tests/unit/test_cli_commands.py::TestDataCommands::test_data_validate

# Run with debugging
pytest --pdb tests/unit/test_failing_test.py
```

**Educational Validation Failures**
```python
# Enable detailed validation logging
import logging
logging.getLogger('toaripi_slm.validation').setLevel(logging.DEBUG)
```

### Performance Optimization

#### Model Optimization

- Use LoRA for efficient fine-tuning
- Implement gradient checkpointing for memory efficiency
- Optimize batch sizes for target hardware
- Use mixed precision training when possible

#### Data Processing Optimization

- Implement parallel processing for large datasets
- Use efficient data formats (Parquet over CSV)
- Implement caching for repeated operations
- Optimize educational validation algorithms

### Security Considerations

#### Content Safety

- Validate all user inputs for educational appropriateness
- Implement content filtering for inappropriate material
- Regular review of generated content samples
- Audit logs for content generation requests

#### Model Security

- Validate model inputs and outputs
- Implement rate limiting for API endpoints
- Secure model file storage and access
- Regular security updates for dependencies

This developer guide provides a comprehensive foundation for contributing to the Toaripi SLM project while maintaining its educational focus and cultural sensitivity.