# Development Setup Guide — Toaripi SLM

> **Complete guide for setting up the Toaripi Small Language Model development environment**

This guide will help you set up a complete development environment for the Toaripi SLM project, including training, inference, and deployment capabilities.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.10 or higher (3.11 recommended)
- **Memory**: 16GB RAM minimum (32GB recommended for training)
- **Storage**: 50GB+ free space for models and data
- **Git**: Latest version for source control

### Optional but Recommended
- **CUDA GPU**: For faster training (RTX 3080/4080+ with 12GB+ VRAM)
- **Docker**: For containerized deployment
- **Visual Studio Code**: With Python and GitHub Copilot extensions

### Package Managers
Choose one of the following Python environment managers:
- **Option A**: `venv` (built-in, lightweight)
- **Option B**: `conda` (recommended for data science)
- **Option C**: `pyenv` + `pipenv` (advanced users)

## 🚀 Quick Start (5 minutes)

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-org/toaripi-slm.git
cd toaripi-slm

# Create virtual environment
python -m venv venv

# Activate environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows Command Prompt:
.\venv\Scripts\activate.bat
# macOS/Linux:
source venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 2. Verify Installation

```bash
# Check installation
python -c "import src.toaripi_slm; print('✅ Toaripi SLM installed successfully')"

# Run basic tests
pytest tests/ -v --tb=short

# Check CLI tools
toaripi-prepare-data --help
toaripi-finetune --help
toaripi-generate --help
```

### 3. Download Sample Data

```bash
# Create sample data for testing
python -c "
import pandas as pd
import os

# Create sample parallel data
sample_data = {
    'english': [
        'The child is playing in the garden.',
        'Mother is cooking rice for dinner.',
        'Father went fishing in the river.',
        'Children are learning at school.',
        'The sun is shining brightly today.'
    ],
    'toaripi': [
        'Narau apu poroporosi hoi-ia.',
        'Ama rasi emene-vuru koko-ia.',
        'Apa ake varu momu-kava.',
        'Narau-vuru eskul-ia parai-ia.',
        'Lai manu vorovoro pei-ia.'
    ]
}

os.makedirs('data/samples', exist_ok=True)
df = pd.DataFrame(sample_data)
df.to_csv('data/samples/sample_parallel.csv', index=False)
print('✅ Sample data created in data/samples/sample_parallel.csv')
"
```


## 🔧 Detailed Development Setup

### Development Dependencies

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks for code quality
pre-commit install

# Verify pre-commit setup
pre-commit run --all-files
```

### IDE Setup (VS Code)

1. **Install VS Code Extensions**:
   ```bash
   # Install recommended extensions
   code --install-extension ms-python.python
   code --install-extension ms-python.pylance
   code --install-extension ms-python.black-formatter
   code --install-extension ms-python.isort
   code --install-extension GitHub.copilot
   ```

2. **Configure VS Code Settings** (`.vscode/settings.json`):
   ```json
   {
     "python.defaultInterpreterPath": "./venv/bin/python",
     "python.formatting.provider": "black",
     "python.linting.enabled": true,
     "python.linting.pylintEnabled": false,
     "python.linting.flake8Enabled": true,
     "editor.formatOnSave": true,
     "editor.codeActionsOnSave": {
       "source.organizeImports": true
     }
   }
   ```

### Environment Configuration

Create a `.env` file for development configuration:

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your specific settings
# Example .env content:
HUGGINGFACE_TOKEN=your_hf_token_here
WANDB_API_KEY=your_wandb_key_here
MODEL_CACHE_DIR=./models
DATA_CACHE_DIR=./data
LOG_LEVEL=INFO
CUDA_VISIBLE_DEVICES=0
```

## 📊 Data Preparation Workflow

### Option A: Local Data Sources

```bash
# Prepare data from local sources
python scripts/prepare_data.py \
  --english_source web_kjv \
  --toaripi_source local_csv \
  --toaripi_path data/raw/toaripi_bible.csv \
  --output data/processed/toaripi_parallel.csv \
  --config configs/data/preprocessing_config.yaml

# Validate the prepared data
python scripts/validate_data.py \
  --input data/processed/toaripi_parallel.csv \
  --output data/processed/validation_report.json
```

### Option B: Configuration-Based Setup

```bash
# Use configuration file for data preparation
python scripts/prepare_data.py \
  --config configs/data/fetch_config.yaml \
  --output data/processed/toaripi_parallel.csv

# Preview the processed data
python -c "
import pandas as pd
df = pd.read_csv('data/processed/toaripi_parallel.csv')
print(f'Dataset size: {len(df)} parallel sentences')
print(df.head())
"
```

## 🤖 Model Training Pipeline

### Basic Fine-tuning

```bash
# Train a small model for testing
python scripts/finetune.py \
  --config configs/training/base_config.yaml \
  --model_name microsoft/DialoGPT-medium \
  --train_data data/processed/toaripi_parallel.csv \
  --output_dir checkpoints/toaripi-dialogpt \
  --epochs 3 \
  --learning_rate 2e-5 \
  --batch_size 4 \
  --gradient_accumulation_steps 4

# Monitor training with tensorboard
tensorboard --logdir checkpoints/toaripi-dialogpt/logs
```

### Advanced Training with LoRA

```bash
# Fine-tune with LoRA for efficiency
python scripts/finetune.py \
  --config configs/training/lora_config.yaml \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --train_data data/processed/toaripi_parallel.csv \
  --output_dir checkpoints/toaripi-mistral-lora \
  --use_lora true \
  --lora_rank 16 \
  --lora_alpha 32 \
  --epochs 2 \
  --learning_rate 1e-4

# Track experiment with Weights & Biases
python scripts/finetune.py \
  --config configs/training/lora_config.yaml \
  --use_wandb true \
  --wandb_project toaripi-slm \
  --wandb_run_name mistral-7b-lora-v1
```

## 🎯 Content Generation Testing

### Story Generation

```bash
# Generate educational stories
python scripts/generate.py \
  --model_path checkpoints/toaripi-mistral-lora \
  --prompt "Write a simple story in Toaripi about children helping their parents with daily chores. Use 5-6 sentences." \
  --max_length 200 \
  --temperature 0.7 \
  --top_p 0.9

# Generate vocabulary exercises
python scripts/generate.py \
  --model_path checkpoints/toaripi-mistral-lora \
  --prompt "Create a vocabulary list of 10 common household items in Toaripi with English translations." \
  --output_format json
```

### Reading Comprehension

```bash
# Generate comprehension questions
python scripts/generate.py \
  --model_path checkpoints/toaripi-mistral-lora \
  --prompt "Write a short paragraph in Toaripi about fishing, then create 3 simple comprehension questions about it." \
  --task_type comprehension \
  --age_group primary
```

## 🌐 Web Application Development

### Local Development Server

```bash
# Start development server with auto-reload
uvicorn app.server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --log-level debug

# Alternative: Use the development script
python -m app.server --dev

# Access the application
# Web UI: http://localhost:8000
# API docs: http://localhost:8000/docs
# OpenAPI spec: http://localhost:8000/openapi.json
```

### API Testing

```bash
# Test API endpoints
curl -X POST "http://localhost:8000/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Generate a simple greeting in Toaripi",
    "max_length": 50,
    "temperature": 0.7
  }'

# Test with Python requests
python -c "
import requests
response = requests.post('http://localhost:8000/api/generate', json={
    'prompt': 'Create a vocabulary word with example sentence',
    'content_type': 'vocabulary',
    'age_group': 'primary'
})
print(response.json())
"
```

## 📱 Model Export and Deployment

### Export to GGUF Format

```bash
# Export trained model to GGUF for efficient inference
python scripts/export_to_gguf.py \
  --model_path checkpoints/toaripi-mistral-lora \
  --output_dir models/gguf \
  --quantization q4_k_m \
  --vocab_type bpe

# Verify GGUF model
python scripts/test_gguf.py \
  --model_path models/gguf/toaripi-mistral-q4_k_m.gguf \
  --prompt "Hello in Toaripi" \
  --max_tokens 50
```

### Docker Deployment

```bash
# Build development image
docker build -t toaripi-slm:dev .

# Build production image
docker build -f Dockerfile.prod -t toaripi-slm:prod .

# Build Raspberry Pi image
docker build -f Dockerfile.pi -t toaripi-slm:pi .

# Run with Docker Compose
docker-compose -f docker-compose.dev.yml up -d

# Check container logs
docker-compose logs -f toaripi-slm
```

## 🧪 Testing and Quality Assurance

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
pytest tests/test_data_processing.py -v  # Specific test file

# Run tests with coverage
pytest tests/ --cov=src/toaripi_slm --cov-report=html

# Run performance tests
pytest tests/performance/ -v --benchmark-only
```

### Code Quality Checks

```bash
# Format code with black
black src/ scripts/ app/ tests/

# Sort imports
isort src/ scripts/ app/ tests/

# Lint with flake8
flake8 src/ scripts/ app/ tests/

# Type checking with mypy
mypy src/toaripi_slm/

# Security checks
bandit -r src/ scripts/ app/

# Run all quality checks
pre-commit run --all-files
```

## 🔍 Debugging and Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size in config
   export CUDA_VISIBLE_DEVICES=0
   python scripts/finetune.py --config configs/training/small_batch_config.yaml
   ```

2. **Import Errors**:
   ```bash
   # Ensure package is installed in development mode
   pip install -e .
   
   # Check Python path
   python -c "import sys; print('\n'.join(sys.path))"
   ```

3. **Model Loading Issues**:
   ```bash
   # Clear cache and reload
   python -c "
   from transformers import AutoModel
   import shutil
   shutil.rmtree('~/.cache/huggingface/transformers', ignore_errors=True)
   "
   ```

### Development Logs

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# View application logs
tail -f logs/toaripi_slm.log

# Monitor training progress
tail -f checkpoints/toaripi-mistral/training.log
```

## 🚀 Advanced Workflows

### Experiment Tracking

```bash
# Initialize experiment tracking
wandb login

# Start experiment with tags
python scripts/finetune.py \
  --config configs/training/experiment_config.yaml \
  --wandb_project toaripi-slm \
  --wandb_tags baseline,mistral-7b,lora \
  --experiment_name baseline-v1.0
```

### Model Evaluation

```bash
# Evaluate model performance
python scripts/evaluate.py \
  --model_path checkpoints/toaripi-mistral-lora \
  --test_data data/processed/test_set.csv \
  --metrics bleu,rouge,perplexity \
  --output_dir evaluation/results

# Generate evaluation report
python scripts/generate_report.py \
  --evaluation_dir evaluation/results \
  --output_format html \
  --include_plots true
```

### Batch Processing

```bash
# Process multiple prompts
python scripts/batch_generate.py \
  --model_path checkpoints/toaripi-mistral-lora \
  --input_file prompts/educational_prompts.jsonl \
  --output_file results/batch_generation.jsonl \
  --batch_size 8
```

## 📚 Next Steps

After completing the setup:

1. **Explore the codebase**: Start with `src/toaripi_slm/` modules
2. **Review configuration files**: Check `configs/` for training and data options
3. **Run example notebooks**: Open `notebooks/` for interactive examples
4. **Join the community**: Check `CONTRIBUTING.md` for contribution guidelines
5. **Read documentation**: Visit `docs/` for detailed guides

## 🆘 Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join GitHub Discussions for questions
- **Community**: Connect with other contributors

---

**Happy coding! 🎉 Let's build something amazing for the Toaripi community.**