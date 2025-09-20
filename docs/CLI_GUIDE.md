# Toaripi SLM Command Line Interface

A sleek, intuitive command-line interface for training, testing, and interacting with Toaripi Small Language Models.

## 🚀 Quick Start

### Installation
```bash
# Install the package
pip install -e .

# Verify installation
toaripi --help
```

### First-Time Setup
```bash
# Interactive guided setup
toaripi setup --guided

# Quick setup without samples
toaripi setup --minimal

# Check system status
toaripi --status
```

## 📋 Available Commands

### Core Commands

#### `toaripi setup`
Initialize project structure and configuration files.

```bash
# Interactive setup with guidance
toaripi setup --guided

# Quick setup without sample data
toaripi setup --minimal

# Force recreation of existing files
toaripi setup --force
```

**What it creates:**
- Directory structure (`data/`, `models/`, `configs/`, etc.)
- Configuration files (training configs, data processing configs)
- Sample data files (unless `--minimal`)

#### `toaripi train`
Train a Toaripi language model on educational data.

```bash
# Interactive guided training
toaripi train --guided

# Quick training with defaults
toaripi train --data data/processed --model-name microsoft/DialoGPT-medium

# LoRA fine-tuning (recommended)
toaripi train --config configs/training/lora_config.yaml --use-lora

# Dry run to validate setup
toaripi train --dry-run
```

**Key features:**
- Guided model selection
- Automatic data validation
- Resource estimation
- Progress monitoring
- Resume from checkpoints

#### `toaripi test`
Test and evaluate trained models.

```bash
# Interactive testing
toaripi test --guided

# Test specific model
toaripi test --model-path models/my-model

# Generation quality test
toaripi test --test-type generation --num-samples 20

# Comprehensive evaluation
toaripi test --test-type all --save-results
```

**Test types:**
- `basic`: Model loading and functionality
- `generation`: Text generation quality
- `quality`: Content assessment (requires test data)
- `all`: Complete evaluation suite

#### `toaripi interact`
Real-time interaction with trained models.

```bash
# Interactive mode with model selection
toaripi interact --guided

# Chat with specific model
toaripi interact --model-path models/my-model

# Generate vocabulary exercises
toaripi interact --content-type vocabulary

# Target specific age group
toaripi interact --age-group secondary
```

**Interactive commands:**
- `<prompt>`: Generate content
- `/story <prompt>`: Generate story
- `/vocab <topic>`: Generate vocabulary
- `/qa <text>`: Generate Q&A
- `/dialogue <scenario>`: Generate dialogue
- `/help`: Show commands
- `/stats`: Session statistics
- `/save`: Save session

#### `toaripi models`
Manage trained models.

```bash
# List all models
toaripi models list

# Show model details
toaripi models info my-model

# Convert to GGUF for deployment
toaripi models convert my-model --to-gguf

# Copy model
toaripi models copy my-model models/backup/

# Prepare for deployment
toaripi models deploy my-model --format both
```

#### `toaripi troubleshoot`
Diagnose and fix common issues.

```bash
# Quick diagnostic
toaripi troubleshoot

# Full system report
toaripi troubleshoot --report

# Clean caches
toaripi troubleshoot --clean-cache
```

### Utility Commands

#### System Status
```bash
# Check system requirements
toaripi --status

# Show version
toaripi --version
```

## 🎯 Guided Workflows

### First-Time User Workflow

1. **Setup Project**
   ```bash
   toaripi setup --guided
   ```

2. **Prepare Data** (if you have raw Toaripi text)
   ```bash
   # Place your text files in data/raw/
   toaripi-prepare-data --config configs/data/preprocessing_config.yaml
   ```

3. **Train Model**
   ```bash
   toaripi train --guided
   ```

4. **Test Model**
   ```bash
   toaripi test --guided
   ```

5. **Interact with Model**
   ```bash
   toaripi interact --guided
   ```

### Advanced User Workflow

1. **Quick Setup**
   ```bash
   toaripi setup --minimal
   ```

2. **Custom Training**
   ```bash
   toaripi train \
     --data data/processed \
     --model-name mistralai/Mistral-7B-Instruct-v0.2 \
     --config configs/training/lora_config.yaml \
     --use-lora \
     --epochs 3 \
     --wandb-project toaripi-experiment
   ```

3. **Batch Testing**
   ```bash
   toaripi test \
     --model-path checkpoints/toaripi-mistral-lora \
     --test-type all \
     --num-samples 50 \
     --save-results
   ```

## 🔧 Configuration

### Training Configurations

**Base Config** (`configs/training/base_config.yaml`)
- Small models (DialoGPT-medium)
- Full fine-tuning
- Good for development

**LoRA Config** (`configs/training/lora_config.yaml`)
- Large models (Mistral-7B)
- Parameter-efficient training
- Recommended for production

### Data Processing Config

**Preprocessing Config** (`configs/data/preprocessing_config.yaml`)
- Text cleaning settings
- Data validation rules
- Train/test split configuration

## 🚨 Troubleshooting

### Common Issues

#### "Command not found: toaripi"
```bash
# Reinstall the package
pip install -e .

# Check if it's in PATH
which toaripi
```

#### "Module not found" errors
```bash
# Check dependencies
toaripi troubleshoot --report

# Reinstall requirements
pip install -r requirements.txt
```

#### Training fails with CUDA errors
```bash
# Check GPU status
toaripi --status

# Use CPU-only training
toaripi train --model-name microsoft/DialoGPT-medium --use-lora
```

#### Low disk space warnings
```bash
# Clean caches
toaripi troubleshoot --clean-cache

# Check space usage
toaripi troubleshoot --report
```

### Getting Help

1. **Built-in Help**
   ```bash
   toaripi --help
   toaripi <command> --help
   ```

2. **System Diagnostics**
   ```bash
   toaripi troubleshoot --report
   ```

3. **Interactive Guidance**
   ```bash
   toaripi <command> --guided
   ```

## 🎨 CLI Features

### User Experience
- **🎯 Guided Mode**: Interactive prompts for new users
- **🔍 Smart Defaults**: Sensible defaults with override options
- **✅ Validation**: Comprehensive input validation
- **📊 Progress Tracking**: Real-time progress and estimates
- **🎨 Rich Output**: Colorful, informative output

### Cross-Platform Support
- **🐧 Linux**: Full support with native tools
- **🪟 Windows**: Compatible with PowerShell and Command Prompt
- **🍎 macOS**: Native support

### Error Handling
- **🚨 Clear Error Messages**: Helpful error descriptions
- **💡 Solution Suggestions**: Automatic troubleshooting tips
- **🔄 Recovery Options**: Resume failed operations
- **📋 Diagnostic Reports**: Detailed system information

### Best Practices Implementation
- **📁 Project Structure**: Standard layout for reproducibility
- **⚙️ Configuration Management**: YAML-based configs
- **📊 Experiment Tracking**: Weights & Biases integration
- **💾 Model Versioning**: Organized model storage
- **🔒 Data Validation**: Input/output verification

## 🔮 Advanced Usage

### Scripting and Automation
```bash
# Non-interactive training
toaripi train \
  --data data/processed \
  --model-name microsoft/DialoGPT-medium \
  --output-dir models/production \
  --epochs 5 \
  --batch-size 8

# Batch model testing
for model in models/*/; do
  toaripi test --model-path "$model" --test-type basic
done
```

### Integration with Other Tools
```bash
# Use with make
make train: 
	toaripi train --config configs/training/production.yaml

# Use with Docker
docker run toaripi-slm toaripi train --guided

# Use with CI/CD
toaripi test --model-path $MODEL_PATH --output-dir test-results/
```

### Custom Configurations
```yaml
# Custom training config
model:
  name: "custom/model"
  cache_dir: "./cache"

training:
  epochs: 10
  learning_rate: 1e-5
  custom_parameter: value

# Load with:
# toaripi train --config my-custom-config.yaml
```

## 📚 Examples

### Educational Content Generation
```bash
# Generate primary school stories
toaripi interact --content-type story --age-group primary

# Create vocabulary exercises
toaripi interact --content-type vocabulary --age-group secondary

# Generate Q&A for comprehension
toaripi interact --content-type qa --age-group adult
```

### Model Development Workflow
```bash
# 1. Setup development environment
toaripi setup --guided

# 2. Train initial model
toaripi train --guided --dry-run  # Validate first
toaripi train --guided

# 3. Evaluate performance
toaripi test --guided --save-results

# 4. Interactive testing
toaripi interact --guided

# 5. Deploy if satisfied
toaripi models deploy my-model --format gguf
```

### Production Deployment
```bash
# Prepare production model
toaripi models deploy production-model --format both --output-dir deploy/

# Validate deployment
toaripi test --model-path deploy/hf/production-model --test-type all

# Generate deployment package
cd deploy && tar -czf toaripi-model-v1.0.tar.gz *
```

---

**💡 Need more help?** Run `toaripi <command> --help` for detailed information about any command, or `toaripi troubleshoot --report` for system diagnostics.