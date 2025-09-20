# Toaripi SLM Command Line Interface

A comprehensive CLI tool for training, testing, and interacting with the Toaripi Small Language Model for educational content generation.

## Quick Start

```bash
# Check system status
toaripi status

# Run system diagnostics
toaripi doctor

# Train a model (interactive mode)
toaripi train --interactive

# Test the trained model
toaripi test --model ./models/hf

# Interactive chat with the model
toaripi interact --model ./models/hf
```

## Installation

1. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   toaripi status --detailed
   ```

## Commands Overview

### `toaripi status`
Check system status and environment setup.

**Options:**
- `--detailed, -d`: Show detailed system information including dependencies

**Examples:**
```bash
# Quick status check
toaripi status

# Detailed system information
toaripi status --detailed
```

### `toaripi doctor`
Comprehensive system health check and troubleshooting.

**Options:**
- `--detailed, -d`: Show detailed diagnostic information
- `--fix, -f`: Attempt to fix common issues automatically
- `--export-report PATH`: Export diagnostic report to file

**Examples:**
```bash
# Basic health check
toaripi doctor

# Detailed diagnostics with auto-fix
toaripi doctor --detailed --fix

# Export diagnostic report
toaripi doctor --export-report ./health_report.json
```

### `toaripi train`
Train a Toaripi SLM model with guided setup and monitoring.

**Options:**
- `--config, -c PATH`: Training configuration file
- `--data-dir PATH`: Data directory (default: ./data)
- `--resume, -r`: Resume from previous training session
- `--interactive, -i`: Interactive configuration (default: True)
- `--dry-run`: Validate setup without training

**Examples:**
```bash
# Interactive training setup (recommended for beginners)
toaripi train --interactive

# Train with specific config
toaripi train --config ./configs/training/custom_config.yaml

# Resume interrupted training
toaripi train --resume

# Validate setup without training
toaripi train --dry-run
```

### `toaripi test`
Test and evaluate a trained Toaripi SLM model.

**Options:**
- `--model, -m PATH`: Path to trained model (default: ./models/hf)
- `--test-data PATH`: Path to test data (default: ./data/processed/test.csv)
- `--output, -o PATH`: Output path for test report
- `--quick, -q`: Run quick tests only
- `--benchmark, -b`: Include performance benchmarks
- `--interactive, -i`: Interactive content review (default: True)

**Examples:**
```bash
# Basic model testing
toaripi test --model ./models/hf

# Quick tests only
toaripi test --quick

# Full testing with benchmarks
toaripi test --benchmark --output ./test_results.json

# Test with interactive content review
toaripi test --interactive
```

### `toaripi interact`
Interactive chat interface with the Toaripi SLM model.

**Options:**
- `--model, -m PATH`: Path to trained model (default: ./models/hf)
- `--content-type, -t TYPE`: Default content type (story, vocabulary, dialogue, questions, translation)
- `--temperature FLOAT`: Generation temperature (default: 0.7)
- `--max-length INT`: Maximum generation length (default: 200)
- `--save-session`: Automatically save session

**Examples:**
```bash
# Start interactive session
toaripi interact --model ./models/hf

# Start with specific content type
toaripi interact --content-type vocabulary

# Adjust generation parameters
toaripi interact --temperature 0.5 --max-length 150
```

## Interactive Mode Commands

When using `toaripi interact`, you can use these commands within the chat:

| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/type <content_type>` | Change content type (story, vocabulary, dialogue, questions, translation) |
| `/settings` | Adjust generation settings |
| `/history` | Show conversation history |
| `/save` | Save conversation to file |
| `/clear` | Clear conversation history |
| `/quit` or `/exit` | Exit interactive mode |

## Content Types

### Story Generation
Generate educational stories in Toaripi suitable for primary school students.

**Example prompts:**
- "Generate a story about children learning to fish"
- "Create a story about helping parents with traditional cooking"
- "Write about children exploring the forest safely"

### Vocabulary Exercises
Create vocabulary lists with Toaripi words and English translations.

**Example prompts:**
- "Create vocabulary about family members"
- "Generate words related to traditional fishing"
- "Make a list of common household items"

### Dialogue Creation
Generate conversations between characters in educational contexts.

**Example prompts:**
- "Create a dialogue between teacher and student about counting"
- "Generate conversation between children playing traditional games"
- "Make dialogue about asking for help in Toaripi"

### Comprehension Questions
Generate reading comprehension questions for educational content.

**Example prompts:**
- "Create questions about the fishing story"
- "Generate comprehension questions for primary students"
- "Make questions about traditional Toaripi customs"

### Translation
Translate English educational content to Toaripi.

**Example prompts:**
- "Translate: The children are playing by the river"
- "Convert this to Toaripi: We eat fish and vegetables"
- "Translate simple classroom instructions"

## Configuration Files

### Training Configuration (`configs/training/base_config.yaml`)
```yaml
model:
  name: "microsoft/DialoGPT-medium"
  cache_dir: "./models/cache"

training:
  epochs: 3
  learning_rate: 2e-5
  batch_size: 4
  gradient_accumulation_steps: 4

lora:
  enabled: true
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj"]

output:
  checkpoint_dir: "./models/checkpoints"
  save_total_limit: 3

logging:
  use_wandb: false
  project_name: "toaripi-slm"
```

### Data Preprocessing (`configs/data/preprocessing_config.yaml`)
```yaml
input:
  raw_data_dir: "./data/raw"
  file_patterns: ["*.csv", "*.tsv"]

processing:
  min_length: 5
  max_length: 512
  remove_duplicates: true
  language_detection: true

output:
  processed_dir: "./data/processed"
  splits:
    train: 0.8
    validation: 0.1
    test: 0.1
```

## Cross-Platform Compatibility

The CLI is designed to work on both Windows and Linux:

### Windows
```cmd
# Installation
pip install -e .

# Usage
toaripi status
toaripi train --interactive
```

### Linux
```bash
# Installation
pip install -e .

# Usage
toaripi status
toaripi train --interactive
```

### WSL (Windows Subsystem for Linux)
```bash
# Works the same as Linux
toaripi status
toaripi doctor --detailed
```

## Troubleshooting

### Common Issues

1. **"toaripi: command not found"**
   ```bash
   # Reinstall in development mode
   pip install -e .
   
   # Or use Python module syntax
   python -m toaripi_slm.cli
   ```

2. **"Model not found" error**
   ```bash
   # Check if you have trained a model
   toaripi status --detailed
   
   # Train a model first
   toaripi train --interactive
   ```

3. **GPU not detected**
   ```bash
   # Check system capabilities
   toaripi doctor --detailed
   
   # Install CUDA if available
   # Use CPU training if needed
   ```

4. **Missing dependencies**
   ```bash
   # Install all dependencies
   pip install -r requirements.txt
   
   # Check what's missing
   toaripi doctor
   ```

5. **Data not found**
   ```bash
   # Check data directory
   ls -la data/processed/
   
   # Prepare data if needed
   toaripi-prepare-data
   ```

### Getting Help

1. **Built-in help:**
   ```bash
   toaripi --help
   toaripi train --help
   toaripi test --help
   ```

2. **System diagnostics:**
   ```bash
   toaripi doctor --detailed
   ```

3. **Check project status:**
   ```bash
   toaripi status --detailed
   ```

### Debug Mode

Enable debug output for troubleshooting:

```bash
toaripi --debug train --interactive
toaripi --verbose doctor --detailed
```

## Advanced Usage

### Custom Model Training
```bash
# Use custom configuration
toaripi train --config ./my_config.yaml --data-dir ./my_data

# Resume training with different settings
toaripi train --resume --config ./updated_config.yaml
```

### Batch Testing
```bash
# Test multiple models
for model in ./models/*/; do
    toaripi test --model "$model" --output "./results/$(basename $model).json"
done
```

### Automated Workflows
```bash
#!/bin/bash
# training_pipeline.sh

# Check system health
toaripi doctor || exit 1

# Train model
toaripi train --interactive || exit 1

# Test model
toaripi test --benchmark || exit 1

# Start interactive session
toaripi interact
```

## Environment Variables

Set these environment variables for customization:

```bash
# Data directories
export TOARIPI_DATA_DIR="./custom_data"
export TOARIPI_MODELS_DIR="./custom_models"

# Training settings
export TOARIPI_USE_GPU="true"
export TOARIPI_BATCH_SIZE="8"

# Logging
export WANDB_PROJECT="my-toaripi-project"
export WANDB_API_KEY="your-key-here"
```

## Performance Tips

1. **GPU Training:**
   - Ensure CUDA is properly installed
   - Use mixed precision with `fp16: true`
   - Adjust batch size based on GPU memory

2. **CPU Training:**
   - Use LoRA for efficient fine-tuning
   - Reduce batch size if memory limited
   - Consider quantization for inference

3. **Data Processing:**
   - Preprocess data once, reuse multiple times
   - Use appropriate train/validation/test splits
   - Clean and validate data regularly

4. **Model Deployment:**
   - Export to GGUF format for edge deployment
   - Use quantization for smaller model size
   - Test on target hardware before deployment