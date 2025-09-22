# Toaripi SLM CLI Reference

## Overview

The Toaripi Small Language Model CLI provides a comprehensive command-line interface for managing educational content generation in the Toaripi language. The CLI is designed for teachers, linguists, and developers working with Toaripi educational materials.

## Installation and Setup

### Prerequisites

- Python 3.10+ (recommended: 3.11 or 3.13)
- Virtual environment (recommended)
- Git for version control

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd toaripi-slm

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e .
pip install -r requirements.txt
```

### Verify Installation

```bash
toaripi-slm --version
toaripi-slm validate
```

## Command Overview

The CLI is organized into five main command groups:

1. **Data Management** (`data`) - Process and validate Toaripi parallel data
2. **Model Management** (`model`) - Handle trained models and exports
3. **Training Operations** (`train`) - Fine-tune models for educational content
4. **Serving & Deployment** (`serve`) - Run educational content API
5. **System Operations** (`status`, `validate`) - Check system health

## Global Options

```bash
toaripi-slm [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

### Global Flags

- `-c, --config PATH` - Configuration file path (YAML/TOML)
- `-v, --verbose` - Enable verbose logging  
- `-q, --quiet` - Suppress non-error output
- `--working-dir DIRECTORY` - Working directory for operations
- `--version` - Show version and exit
- `--help` - Show help message

## Data Management Commands

### `toaripi-slm data`

Manage and process Toaripi educational data with cultural validation.

#### `data list` - List Available Datasets

```bash
toaripi-slm data list [OPTIONS]
```

**Options:**
- `--directory PATH` - Data directory to scan (default: `data/`)
- `--format [csv|json|parquet]` - Filter by file format
- `--educational-only` - Show only educationally validated datasets

**Example:**
```bash
toaripi-slm data list --directory data/processed --educational-only
```

#### `data validate` - Validate Data Quality

```bash
toaripi-slm data validate [OPTIONS]
```

**Options:**
- `-f, --file PATH` - Data file to validate [required]
- `--check-cultural` - Check cultural appropriateness
- `--check-educational` - Check educational suitability
- `--min-length INTEGER` - Minimum text length
- `--max-length INTEGER` - Maximum text length
- `-v, --verbose` - Show detailed validation results

**Example:**
```bash
toaripi-slm data validate --file data/raw/parallel.csv --check-educational --check-cultural
```

#### `data prepare` - Process Raw Data

```bash
toaripi-slm data prepare [OPTIONS]
```

**Options:**
- `-i, --input PATH` - Input data file [required]
- `-o, --output PATH` - Output directory
- `--train-split FLOAT` - Training data split (default: 0.8)
- `--validation-split FLOAT` - Validation split (default: 0.1)
- `--test-split FLOAT` - Test data split (default: 0.1)
- `--age-groups [early_childhood|primary_lower|primary_upper|secondary|adult|teacher]` - Target age groups
- `--content-types [story|vocabulary|dialogue|comprehension|grammar|cultural|song|poem|lesson|exercise]` - Content types to generate
- `--dry-run` - Preview operations without making changes

**Example:**
```bash
toaripi-slm data prepare \
  --input data/raw/Full_bible_english_toaripi.csv \
  --output data/processed \
  --age-groups primary_lower primary_upper \
  --content-types story vocabulary dialogue
```

#### `data convert` - Convert Data Formats

```bash
toaripi-slm data convert [OPTIONS]
```

**Options:**
- `-i, --input PATH` - Input file to convert [required]
- `-o, --output PATH` - Output file path
- `--from-format [csv|json|parquet|usfm]` - Source format (auto-detected)
- `--to-format [csv|json|parquet]` - Target format [required]
- `--preserve-metadata` - Preserve metadata and structure

**Example:**
```bash
toaripi-slm data convert \
  --input data/processed/train.csv \
  --to-format parquet \
  --preserve-metadata
```

## Model Management Commands

### `toaripi-slm model`

Manage trained models and prepare for edge deployment.

#### `model list` - List Available Models

```bash
toaripi-slm model list [OPTIONS]
```

**Options:**
- `--directory PATH` - Models directory (default: `models/`)
- `--format [hf|gguf|checkpoint]` - Filter by model format
- `--educational-only` - Show only educational models

#### `model export` - Export Models for Deployment

```bash
toaripi-slm model export [OPTIONS] MODEL_PATH
```

**Options:**
- `--format [gguf|onnx]` - Export format (default: gguf)
- `--quantization [q4_k_m|q5_k_m|q8_0|f16|f32]` - Quantization level
- `--device [auto|cpu|gpu|raspberry-pi]` - Target device
- `--educational-validation` - Include educational validation metadata
- `-o, --output PATH` - Output directory

**Example:**
```bash
toaripi-slm model export models/hf/toaripi-primary \
  --format gguf \
  --quantization q4_k_m \
  --device raspberry-pi \
  --educational-validation
```

#### `model info` - Model Information

```bash
toaripi-slm model info [OPTIONS] MODEL_PATH
```

Shows detailed information about model capabilities, training data, and educational suitability.

#### `model test` - Test Model Educational Output

```bash
toaripi-slm model test [OPTIONS] MODEL_PATH
```

**Options:**
- `--age-group [early_childhood|primary_lower|primary_upper|secondary]` - Target age group
- `--content-type [story|vocabulary|dialogue|comprehension]` - Content type to test
- `--prompt TEXT` - Custom test prompt
- `--cultural-check` - Validate cultural appropriateness

## Training Commands

### `toaripi-slm train`

Fine-tune models specifically for Toaripi educational content generation.

#### `train start` - Start Training Session

```bash
toaripi-slm train start [OPTIONS]
```

**Options:**
- `--data PATH` - Training data file (CSV with English-Toaripi parallel text) [required]
- `--model TEXT` - Base model to fine-tune (default: microsoft/DialoGPT-small)
- `--output PATH` - Output directory for trained model
- `--epochs INTEGER` - Number of training epochs (default: 3)
- `--batch-size INTEGER` - Training batch size (default: 4)
- `--learning-rate FLOAT` - Learning rate (default: 2e-5)
- `--validation-level [basic|educational|cultural|strict]` - Educational content validation level
- `--age-groups [early_childhood|primary_lower|primary_upper|secondary|adult|teacher]` - Target age groups
- `--content-types [story|vocabulary|dialogue|comprehension|grammar|cultural|song|poem|lesson|exercise]` - Types of educational content
- `--device [auto|cpu|gpu|raspberry-pi]` - Target device for deployment
- `--use-lora` - Use LoRA for efficient fine-tuning
- `--lora-rank INTEGER` - LoRA rank (default: 16)
- `--max-length INTEGER` - Maximum sequence length (default: 512)
- `--resume TEXT` - Resume training from session ID
- `--dry-run` - Validate configuration without starting training

**Example:**
```bash
toaripi-slm train start \
  --data data/processed/toaripi_parallel.csv \
  --model microsoft/DialoGPT-small \
  --epochs 5 \
  --age-groups primary_lower primary_upper \
  --content-types story vocabulary dialogue \
  --validation-level strict \
  --use-lora
```

#### `train status` - Check Training Progress

```bash
toaripi-slm train status [OPTIONS]
```

Shows current training status, progress, and educational validation metrics.

#### `train stop` - Stop Training Session

```bash
toaripi-slm train stop [OPTIONS]
```

Safely stops the current training session and saves checkpoints.

#### `train logs` - View Training Logs

```bash
toaripi-slm train logs [OPTIONS]
```

**Options:**
- `--session-id TEXT` - Specific session ID
- `--lines INTEGER` - Number of log lines to show
- `--follow` - Follow logs in real-time

#### `train list` - List Training Sessions

```bash
toaripi-slm train list [OPTIONS]
```

Shows all training sessions with educational metrics and status.

#### `train resume` - Resume Training

```bash
toaripi-slm train resume [OPTIONS] SESSION_ID
```

Resume a previously stopped training session.

## Serving & Deployment Commands

### `toaripi-slm serve`

Deploy educational content generation API for classroom use.

#### `serve start` - Start Educational API Server

```bash
toaripi-slm serve start [OPTIONS]
```

**Options:**
- `--model PATH` - Model file to serve [required]
- `--host TEXT` - Host address (default: localhost)
- `--port INTEGER` - Port number (default: 8000)
- `--age-filter [early_childhood|primary_lower|primary_upper|secondary]` - Age-appropriate filtering
- `--cultural-validation` - Enable cultural appropriateness checks
- `--max-length INTEGER` - Maximum generation length
- `--temperature FLOAT` - Generation randomness (default: 0.7)
- `--workers INTEGER` - Number of worker processes
- `--daemon` - Run as background daemon

**Example:**
```bash
toaripi-slm serve start \
  --model models/gguf/toaripi-primary-q4.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --age-filter primary_lower \
  --cultural-validation \
  --daemon
```

#### `serve status` - Check Server Status

```bash
toaripi-slm serve status
```

Shows current server status, active connections, and performance metrics.

#### `serve stop` - Stop Server

```bash
toaripi-slm serve stop [OPTIONS]
```

Gracefully stops the educational API server.

#### `serve test` - Test API Endpoints

```bash
toaripi-slm serve test [OPTIONS]
```

**Options:**
- `--endpoint TEXT` - Specific endpoint to test
- `--age-group TEXT` - Age group for testing
- `--content-type TEXT` - Content type for testing

Tests educational content generation endpoints with sample requests.

## System Commands

### `toaripi-slm status`

```bash
toaripi-slm status
```

Shows comprehensive system status including:
- Python environment and dependencies
- Available data files and validation status
- Trained models and formats
- Configuration settings
- Educational content capabilities

### `toaripi-slm validate`

```bash
toaripi-slm validate
```

Validates entire system setup including:
- Required dependencies installation
- Data file accessibility and format
- Model availability and compatibility
- Educational content validation systems
- Cultural appropriateness checks

## Educational Content Focus

All commands in the Toaripi SLM CLI are designed with educational content in mind:

### Age-Appropriate Content
- **Early Childhood** (ages 3-5): Simple vocabulary, basic concepts
- **Primary Lower** (ages 6-8): Stories, basic reading comprehension
- **Primary Upper** (ages 9-11): Complex narratives, vocabulary expansion
- **Secondary** (ages 12-18): Advanced content, cultural studies

### Content Types
- **Stories**: Narrative content for reading practice
- **Vocabulary**: Word lists with definitions and examples
- **Dialogue**: Conversational practice scenarios
- **Comprehension**: Reading comprehension questions
- **Grammar**: Language structure exercises
- **Cultural**: Traditional stories and cultural knowledge
- **Songs/Poems**: Rhythmic and poetic content
- **Lessons**: Structured educational materials
- **Exercises**: Practice activities and assessments

### Cultural Validation
- Traditional Toaripi values and customs
- Age-appropriate cultural concepts
- Respectful representation of community knowledge
- Avoidance of inappropriate or sensitive content

## Configuration Files

### Data Processing Configuration (`configs/data/preprocessing_config.yaml`)
```yaml
data_sources:
  parallel_data: "data/raw/Full_bible_english_toaripi.csv"
  
validation:
  min_length: 5
  max_length: 500
  cultural_validation: true
  educational_validation: true
  
output_formats:
  - csv
  - json
  
age_groups:
  - primary_lower
  - primary_upper
```

### Training Configuration (`configs/training/toaripi_educational_config.yaml`)
```yaml
model:
  base_model: "microsoft/DialoGPT-small"
  use_lora: true
  lora_rank: 16
  
training:
  epochs: 3
  batch_size: 4
  learning_rate: 0.00002
  
educational:
  validation_level: "strict"
  target_age_groups:
    - primary_lower
    - primary_upper
  content_types:
    - story
    - vocabulary
    - dialogue
```

## Best Practices

### For Teachers
1. Always validate data for cultural appropriateness
2. Use age-appropriate filters for classroom content
3. Test generated content before using with students
4. Regular validation of educational suitability

### For Developers
1. Use educational validation in all data processing
2. Include cultural sensitivity checks
3. Target specific age groups for content generation
4. Regular testing of model outputs

### For Data Processing
1. Always backup original data before processing
2. Use dry-run mode for preview operations
3. Validate all data transformations
4. Maintain educational metadata throughout pipeline

## Troubleshooting

### Common Issues

#### "Missing package: transformers"
```bash
pip install transformers datasets accelerate
```

#### "No data files found"
- Check data directory path: `toaripi-slm data list --directory your/path`
- Ensure CSV files have correct format (english, toaripi columns)

#### "Model not found"
- Verify model path: `toaripi-slm model list`
- Check model format compatibility

#### "Cultural validation failed"
- Review content for cultural appropriateness
- Use `--check-cultural` flag for detailed feedback

### Getting Help

- Use `--help` with any command for detailed options
- Check `toaripi-slm validate` for system requirements
- Review logs in training/serving operations
- Consult documentation for educational content guidelines

## Examples and Use Cases

### Classroom Preparation Workflow
```bash
# 1. Validate system setup
toaripi-slm validate

# 2. Prepare educational data
toaripi-slm data prepare \
  --input data/raw/classroom_content.csv \
  --age-groups primary_lower \
  --content-types story vocabulary

# 3. Train classroom-specific model
toaripi-slm train start \
  --data data/processed/train.csv \
  --age-groups primary_lower \
  --validation-level educational

# 4. Export for classroom tablet
toaripi-slm model export models/hf/classroom-model \
  --format gguf \
  --device raspberry-pi \
  --quantization q4_k_m

# 5. Start educational API
toaripi-slm serve start \
  --model models/gguf/classroom-model-q4.gguf \
  --age-filter primary_lower \
  --cultural-validation
```

### Content Development Pipeline
```bash
# Validate source materials
toaripi-slm data validate \
  --file data/raw/new_content.csv \
  --check-educational \
  --check-cultural

# Convert formats for processing
toaripi-slm data convert \
  --input data/raw/stories.json \
  --to-format csv

# Test model with new content
toaripi-slm model test models/educational \
  --age-group primary_upper \
  --content-type story \
  --cultural-check
```

This CLI reference provides comprehensive guidance for using the Toaripi SLM system effectively in educational contexts while maintaining cultural sensitivity and age-appropriate content generation.