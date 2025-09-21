# Toaripi SLM WSL Setup Guide

> **Complete guide for setting up the Toaripi Small Language Model development environment in Windows Subsystem for Linux (WSL)**

This guide provides step-by-step instructions for initializing the Toaripi SLM project in WSL, ensuring optimal performance and compatibility for machine learning workflows.

## 📋 Prerequisites

### System Requirements
- **Windows 10 version 2004+** or **Windows 11**
- **WSL2** installed and configured
- **16GB RAM minimum** (32GB recommended for training)
- **50GB+ free storage space** for models and data
- **Admin privileges** for initial WSL setup

### Before You Begin
- Ensure Windows Subsystem for Linux is enabled
- Have your GitHub credentials ready
- Consider setting up GPU support if available (CUDA/ROCm)

---

## 🐧 Part 1: WSL Environment Setup

### 1.1 Install and Configure WSL2

```bash
# Check if WSL is already installed
wsl --version

# If WSL is not installed, run in PowerShell as Administrator:
# wsl --install -d Ubuntu

# Update to WSL2 if needed
# wsl --set-default-version 2

# Verify WSL2 is running
wsl --list --verbose
```

### 1.2 Ubuntu System Setup

```bash
# Update Ubuntu packages (run inside WSL)
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y \
    build-essential \
    curl \
    git \
    wget \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release

# Install additional development tools
sudo apt install -y \
    htop \
    tree \
    unzip \
    vim \
    nano
```

### 1.3 Install Python 3.11

```bash
# Add Python 3.11 repository
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# Install Python 3.11 and related packages
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    python3.11-distutils

# Set Python 3.11 as default (optional but recommended)
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo update-alternatives --config python3

# Verify Python installation
python3 --version  # Should show 3.11.x
which python3      # Should show /usr/bin/python3
```

---

## 🚀 Part 2: Project Initialization

### 2.1 Clone and Navigate to Project

```bash
# Navigate to your Windows project directory from WSL
cd /mnt/c/VSProjects/toaripi-slm

# Alternative: Clone fresh repository
# git clone https://github.com/cknzraposo/toaripi-slm.git
# cd toaripi-slm

# Verify you're in the correct directory
pwd
ls -la
```

### 2.2 Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (prompt should show (venv))
which python    # Should point to venv/bin/python
python --version

# Upgrade pip and essential tools
python -m pip install --upgrade pip setuptools wheel
```

### 2.3 Create Project Dependencies

Since the requirements files don't exist yet, we'll create them:

```bash
# Create main requirements file
cat > requirements.txt << 'EOF'
# Core ML libraries
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
accelerate>=0.20.0
peft>=0.4.0
sentencepiece>=0.1.99

# Data processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Web framework
fastapi>=0.100.0
uvicorn[standard]>=0.22.0
pydantic>=2.0.0
jinja2>=3.1.0

# Configuration and utilities
pyyaml>=6.0
toml>=0.10.2
tqdm>=4.65.0
requests>=2.31.0
python-dotenv>=1.0.0

# Logging and monitoring
loguru>=0.7.0
wandb>=0.15.0

# Optional: For GGUF export (uncomment if needed)
# llama-cpp-python>=0.1.78
EOF

# Create development requirements
cat > requirements-dev.txt << 'EOF'
# Include base requirements
-r requirements.txt

# Testing framework
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-asyncio>=0.21.0
pytest-benchmark>=4.0.0

# Code quality tools
black>=23.7.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.5.0
bandit>=1.7.5

# Pre-commit hooks
pre-commit>=3.3.0

# Documentation
mkdocs>=1.5.0
mkdocs-material>=9.0.0

# Jupyter for development
jupyter>=1.0.0
notebook>=6.5.0
ipykernel>=6.25.0
EOF

# Install all dependencies
pip install -r requirements-dev.txt
```

---

## 🏗️ Part 3: Project Structure Setup

### 3.1 Create Configuration Files

```bash
# Create data preprocessing configuration
mkdir -p configs/data
cat > configs/data/preprocessing_config.yaml << 'EOF'
# Data preprocessing configuration for Toaripi SLM
data_sources:
  english:
    type: "web_kjv"
    cache_dir: "./data/cache/english"
    url: "https://www.gutenberg.org/files/10/10-0.txt"
  toaripi:
    type: "local_csv"
    path: "./data/raw/toaripi_bible.csv"
    encoding: "utf-8"

preprocessing:
  text_cleaning:
    min_length: 10
    max_length: 512
    remove_duplicates: true
    normalize_unicode: true
    strip_whitespace: true
  
  alignment:
    method: "sentence_align"
    similarity_threshold: 0.7
    max_ratio: 3.0
  
  filtering:
    remove_empty: true
    remove_non_text: true
    language_detection: true

output:
  format: "csv"
  encoding: "utf-8"
  columns: ["english", "toaripi", "verse_id", "book", "chapter"]
  
validation:
  test_split: 0.1
  dev_split: 0.05
  random_seed: 42
  stratify_by: "book"
EOF

# Create training configurations
mkdir -p configs/training

# Base training config for smaller models
cat > configs/training/base_config.yaml << 'EOF'
# Basic training configuration for development/testing
model:
  name: "microsoft/DialoGPT-medium"
  cache_dir: "./models/cache"
  trust_remote_code: false

training:
  # Training hyperparameters
  epochs: 3
  learning_rate: 2e-5
  batch_size: 4
  gradient_accumulation_steps: 4
  warmup_steps: 100
  weight_decay: 0.01
  
  # Evaluation and saving
  eval_strategy: "steps"
  eval_steps: 500
  save_strategy: "steps"
  save_steps: 500
  logging_steps: 100
  
  # Early stopping
  early_stopping_patience: 3
  early_stopping_threshold: 0.001

data:
  max_length: 512
  padding: true
  truncation: true
  return_tensors: "pt"

optimization:
  optimizer: "adamw"
  lr_scheduler_type: "linear"
  fp16: false  # Set to true if GPU supports it
  dataloader_pin_memory: true
  remove_unused_columns: false

output:
  checkpoint_dir: "./checkpoints"
  save_total_limit: 3
  push_to_hub: false
  hub_model_id: ""

logging:
  use_wandb: false
  project_name: "toaripi-slm"
  run_name: "base-training"
  log_level: "INFO"
EOF

# LoRA configuration for larger models
cat > configs/training/lora_config.yaml << 'EOF'
# LoRA fine-tuning configuration for efficient training
model:
  name: "mistralai/Mistral-7B-Instruct-v0.2"
  cache_dir: "./models/cache"
  trust_remote_code: false
  device_map: "auto"

lora:
  enabled: true
  rank: 16
  alpha: 32
  dropout: 0.1
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  bias: "none"
  task_type: "CAUSAL_LM"

training:
  # Training hyperparameters optimized for LoRA
  epochs: 2
  learning_rate: 1e-4
  batch_size: 1  # Small batch size for memory efficiency
  gradient_accumulation_steps: 16
  warmup_ratio: 0.1
  weight_decay: 0.001
  
  # Evaluation and saving
  eval_strategy: "steps"
  eval_steps: 250
  save_strategy: "steps"
  save_steps: 250
  logging_steps: 50
  
  # Memory optimization
  gradient_checkpointing: true
  dataloader_pin_memory: false

data:
  max_length: 1024
  padding: true
  truncation: true
  return_tensors: "pt"

optimization:
  optimizer: "adamw"
  lr_scheduler_type: "cosine"
  fp16: true
  bf16: false  # Use bf16 if supported by hardware
  tf32: true   # Enable for A100/H100 GPUs

output:
  checkpoint_dir: "./checkpoints"
  save_total_limit: 2
  push_to_hub: false
  hub_model_id: ""

logging:
  use_wandb: true
  project_name: "toaripi-slm"
  run_name: "mistral-7b-lora"
  log_level: "INFO"
  
memory:
  max_memory_mb: 24000  # Adjust based on available GPU memory
  cpu_offload: true
EOF
```

### 3.2 Create Package Structure

```bash
# Ensure all __init__.py files exist
touch src/toaripi_slm/__init__.py
touch src/toaripi_slm/core/__init__.py
touch src/toaripi_slm/data/__init__.py
touch src/toaripi_slm/inference/__init__.py
touch src/toaripi_slm/utils/__init__.py

# Create main package initialization
cat > src/toaripi_slm/__init__.py << 'EOF'
"""
Toaripi Small Language Model (SLM)

A specialized language model for generating educational content in Toaripi,
an endangered language from Papua New Guinea's Gulf Province.

This package provides tools for:
- Fine-tuning language models on Toaripi data
- Generating educational content (stories, vocabulary, Q&A)
- Deploying models for online and offline use
- Supporting language preservation and education efforts

Example usage:
    >>> from toaripi_slm import ToaripiGenerator
    >>> generator = ToaripiGenerator.load("models/toaripi-mistral")
    >>> story = generator.generate_story("children fishing", age_group="primary")
"""

__version__ = "0.1.0"
__author__ = "Toaripi SLM Contributors"
__email__ = "cknzraposo@gmail.com"
__license__ = "MIT"

# Main imports
try:
    from .core import ToaripiTrainer
    from .inference import ToaripiGenerator
    from .data import DataProcessor
    from .utils import load_config, setup_logging
    
    __all__ = [
        "ToaripiTrainer",
        "ToaripiGenerator", 
        "DataProcessor",
        "load_config",
        "setup_logging",
    ]
except ImportError:
    # Handle case where dependencies aren't installed yet
    __all__ = []

# Package metadata
PACKAGE_INFO = {
    "name": "toaripi-slm",
    "version": __version__,
    "description": "Small Language Model for Toaripi Educational Content",
    "language": "Toaripi (tqo)",
    "region": "Gulf Province, Papua New Guinea",
    "purpose": "Educational content generation and language preservation",
    "deployment": ["online", "offline", "edge"],
    "model_sizes": ["1B", "3B", "7B"],
}
EOF

# Create setup.py for development installation
cat > setup.py << 'EOF'
"""Setup configuration for Toaripi SLM package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "readme.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "A small language model for generating educational content in Toaripi language"

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    req_path = Path(__file__).parent / filename
    if req_path.exists():
        with open(req_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="toaripi-slm",
    version="0.1.0",
    author="Toaripi SLM Contributors",
    author_email="cknzraposo@gmail.com",
    description="A small language model for generating educational content in Toaripi language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cknzraposo/toaripi-slm",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "gpu": ["torch[cuda]>=2.0.0"],
        "gguf": ["llama-cpp-python>=0.1.78"],
    },
    entry_points={
        "console_scripts": [
            "toaripi-prepare-data=scripts.prepare_data:main",
            "toaripi-finetune=scripts.finetune:main", 
            "toaripi-generate=scripts.generate:main",
            "toaripi-export=scripts.export_gguf:main",
            "toaripi-serve=app.server:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education", 
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Natural Language :: English",
        "Natural Language :: Other",
    ],
    keywords=[
        "language-model",
        "toaripi",
        "education",
        "nlp",
        "papua-new-guinea",
        "endangered-languages",
        "language-preservation",
        "machine-learning",
        "transformers",
    ],
    project_urls={
        "Bug Reports": "https://github.com/cknzraposo/toaripi-slm/issues",
        "Source": "https://github.com/cknzraposo/toaripi-slm",
        "Documentation": "https://github.com/cknzraposo/toaripi-slm/docs",
    },
)
EOF

# Install package in development mode
pip install -e .
```

### 3.3 Create Sample Data

```bash
# Create sample parallel data for testing
mkdir -p data/samples
python3 << 'EOF'
import pandas as pd
import os

# Extended sample data with more variety
sample_data = {
    'english': [
        'The child is playing in the garden.',
        'Mother is cooking rice for dinner.',
        'Father went fishing in the river.',
        'Children are learning at school.',
        'The sun is shining brightly today.',
        'We gather coconuts from the palm tree.',
        'The boat sails across the calm water.',
        'Grandmother tells stories at night.',
        'Birds sing in the morning forest.',
        'The village celebrates harvest time.',
        'Young people help with house building.',
        'Fishermen return with their catch.',
        'Women weave traditional baskets.',
        'Elders share wisdom with youth.',
        'Rain falls gently on the gardens.',
        'Stars shine over the quiet village.',
        'Dogs bark at strangers passing by.',
        'Pigs root around in the mud.',
        'Chickens peck at grains of rice.',
        'The river flows toward the sea.'
    ],
    'toaripi': [
        'Narau apu poroporosi hoi-ia.',
        'Ama rasi emene-vuru koko-ia.',
        'Apa ake varu momu-kava.',
        'Narau-vuru eskul-ia parai-ia.',
        'Lai manu vorovoro pei-ia.',
        'Niu viru varave ai-kava.',
        'Vaka ake vave-hare lao-ia.',
        'Bubu hahine vuku-ia emo-re.',
        'Mavau-vuru ai-hare biri pei-ia.',
        'Kori ai meho-vuru kekere-ia.',
        'Doa-vuru ruma bou-vuru kiki-ia.',
        'Ika-ai karavi veki imu-kava.',
        'Hahine-vuru kete vavine kumu-ia.',
        'Taubarere doa-vuru parai-kava.',
        'Ame kakaviru apu-ria lou-ia.',
        'Keupa kori motumotu-re malau-ia.',
        'Koli tavani pokai-vuru loko-ia.',
        'Boroma kekei hoi-ia poraki-ia.',
        'Kokorako rasi buni kani-ia.',
        'Varu asi-ria lao-kava.'
    ],
    'verse_id': [f'sample_{i+1:03d}' for i in range(20)],
    'book': ['Samples'] * 20,
    'chapter': [1] * 20
}

os.makedirs('data/samples', exist_ok=True)
df = pd.DataFrame(sample_data)
df.to_csv('data/samples/sample_parallel.csv', index=False)

print('✅ Sample data created in data/samples/sample_parallel.csv')
print(f'Dataset contains {len(df)} parallel sentences')
print('\nFirst few entries:')
for i, row in df.head(3).iterrows():
    print(f'  EN: {row["english"]}')
    print(f'  TQO: {row["toaripi"]}')
    print()
EOF

# Create additional sample files
cat > data/samples/README.md << 'EOF'
# Sample Data for Toaripi SLM

This directory contains sample datasets for development and testing.

## Files

- `sample_parallel.csv`: Basic parallel English-Toaripi sentences
- `educational_prompts.json`: Sample prompts for educational content generation
- `vocabulary_topics.yaml`: Topics for vocabulary generation

## Data Format

### Parallel Data (CSV)
```csv
english,toaripi,verse_id,book,chapter
"The child is playing in the garden.","Narau apu poroporosi hoi-ia.",sample_001,Samples,1
```

### Educational Prompts (JSON)
```json
{
  "story_prompts": [
    {
      "topic": "daily_activities",
      "prompt": "Write a story about children helping with daily chores",
      "age_group": "primary",
      "length": "short"
    }
  ]
}
```

## Usage

Use these samples for:
- Testing data processing pipelines
- Validating model training workflows
- Developing content generation features
- Creating unit tests

**Note:** These are synthetic samples for development only. Real training data should contain authentic Toaripi language content validated by native speakers.
EOF

# Create educational prompts sample
cat > data/samples/educational_prompts.json << 'EOF'
{
  "story_prompts": [
    {
      "id": "story_001",
      "topic": "family_activities",
      "prompt": "Write a simple story about a family preparing food together",
      "age_group": "primary",
      "length": "short",
      "learning_objectives": ["family vocabulary", "cooking verbs", "cooperation"]
    },
    {
      "id": "story_002", 
      "topic": "nature_exploration",
      "prompt": "Tell a story about children discovering animals in the forest",
      "age_group": "primary",
      "length": "medium",
      "learning_objectives": ["animal names", "forest vocabulary", "exploration"]
    }
  ],
  "vocabulary_prompts": [
    {
      "id": "vocab_001",
      "topic": "household_items",
      "prompt": "Create a list of 10 common household items with Toaripi names",
      "age_group": "primary",
      "format": "list",
      "include_examples": true
    },
    {
      "id": "vocab_002",
      "topic": "food_items", 
      "prompt": "Generate vocabulary for traditional foods and cooking",
      "age_group": "primary",
      "format": "flashcards",
      "include_examples": true
    }
  ],
  "comprehension_prompts": [
    {
      "id": "comp_001",
      "text_topic": "fishing_trip",
      "prompt": "Write a short paragraph about fishing, then create 3 questions",
      "age_group": "primary",
      "question_types": ["who", "what", "where"]
    }
  ]
}
EOF
```

---

## 🧪 Part 4: Testing and Validation

### 4.1 Create Test Structure

```bash
# Create test directories and files
mkdir -p tests/{unit,integration,fixtures}
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py

# Create comprehensive test suite
cat > tests/test_installation.py << 'EOF'
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
EOF

# Create basic unit tests
cat > tests/unit/test_package.py << 'EOF'
"""Unit tests for core package functionality."""

import pytest
from pathlib import Path


def test_package_metadata():
    """Test package metadata is accessible."""
    import src.toaripi_slm as pkg
    
    assert hasattr(pkg, '__version__')
    assert hasattr(pkg, '__author__')
    assert hasattr(pkg, 'PACKAGE_INFO')
    
    # Check package info structure
    info = pkg.PACKAGE_INFO
    required_keys = ['name', 'version', 'description', 'language', 'purpose']
    for key in required_keys:
        assert key in info, f"Missing key {key} in PACKAGE_INFO"


def test_import_structure():
    """Test that expected modules can be imported."""
    # Test that modules exist (even if classes aren't implemented yet)
    module_paths = [
        'src.toaripi_slm.core',
        'src.toaripi_slm.data', 
        'src.toaripi_slm.inference',
        'src.toaripi_slm.utils'
    ]
    
    for module_path in module_paths:
        try:
            __import__(module_path)
        except ImportError as e:
            pytest.fail(f"Could not import {module_path}: {e}")
EOF

# Run tests to verify setup
python -m pytest tests/test_installation.py -v
echo "✅ Installation tests completed"
```

### 4.2 Set up Code Quality Tools

```bash
# Create .gitignore file
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
.venv/

# IDEs and editors
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store
Thumbs.db

# Model files and caches
checkpoints/
models/cache/
models/gguf/*.gguf
models/hf/*/pytorch_model.bin
models/hf/*/model.safetensors

# Data files (exclude samples)
data/raw/
data/processed/
!data/samples/
*.csv
*.jsonl
!data/samples/*.csv
!data/samples/*.json

# Logs and outputs
logs/
*.log
wandb/
.wandb/

# Environment variables
.env
.env.local

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Testing
.coverage
.coverage.*
htmlcov/
.tox/
.pytest_cache/

# Documentation
docs/_build/
site/

# Cache directories
.cache/
*/.cache/
__pycache__/

# Temporary files
*.tmp
*.temp
.tmp/

# OS specific
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Large files
*.bin
*.safetensors
*.pt
*.pth
*.ckpt
*.pkl
*.pickle

# Hugging Face cache
.huggingface/
transformers_cache/
EOF

# Set up pre-commit configuration
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
        exclude: \.md$
      - id: end-of-file-fixer
        exclude: \.md$
      - id: check-yaml
        args: ['--unsafe']
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=10240']  # 10MB limit
      - id: check-merge-conflict
      - id: check-symlinks
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.11
        args: ['--line-length=88']

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ['--profile', 'black', '--line-length=88']

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=88', '--extend-ignore=E203,W503,E501']
        additional_dependencies: [flake8-docstrings]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML, types-requests]
        args: ['--ignore-missing-imports', '--no-strict-optional']

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['--recursive', '--skip', 'B101,B601']
        exclude: ^tests/
EOF

# Install and configure pre-commit
pre-commit install
echo "✅ Pre-commit hooks installed"

# Create environment file
cat > .env.example << 'EOF'
# Hugging Face Configuration
HUGGINGFACE_TOKEN=your_hf_token_here
HF_HOME=./models/cache

# Weights & Biases Configuration  
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=toaripi-slm
WANDB_ENTITY=your_wandb_entity

# Model and Data Paths
MODEL_CACHE_DIR=./models/cache
DATA_CACHE_DIR=./data/cache
CHECKPOINT_DIR=./checkpoints

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/toaripi_slm.log

# CUDA Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Application Settings
HOST=0.0.0.0
PORT=8000
DEBUG=false
WORKERS=1

# Training Configuration
MAX_EPOCHS=10
BATCH_SIZE=4
LEARNING_RATE=2e-5
WARMUP_STEPS=100

# Generation Settings
MAX_LENGTH=512
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50
EOF

cp .env.example .env
echo "✅ Environment configuration created"
```

---

## 🎯 Part 5: Final Verification and Next Steps

### 5.1 Comprehensive System Check

```bash
# Create verification script
cat > verify_setup.py << 'EOF'
#!/usr/bin/env python3
"""Comprehensive setup verification for Toaripi SLM."""

import sys
import subprocess
import importlib
from pathlib import Path
import pandas as pd
import yaml
import json

def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version >= (3, 10):
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (3.10+ required)")
        return False

def check_virtual_environment():
    """Check if running in virtual environment."""
    print("🏠 Checking virtual environment...")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("   ✅ Running in virtual environment")
        return True
    else:
        print("   ⚠️  Not running in virtual environment")
        return False

def check_package_installation():
    """Check if package is installed."""
    print("📦 Checking package installation...")
    try:
        import src.toaripi_slm
        print(f"   ✅ Package version: {src.toaripi_slm.__version__}")
        return True
    except ImportError as e:
        print(f"   ❌ Package import failed: {e}")
        return False

def check_dependencies():
    """Check core dependencies."""
    print("🔗 Checking dependencies...")
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets', 
        'accelerate': 'Accelerate',
        'peft': 'PEFT',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'yaml': 'PyYAML',
        'tqdm': 'TQDM'
    }
    
    success_count = 0
    for package, name in required_packages.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"   ✅ {name}: {version}")
            success_count += 1
        except ImportError:
            print(f"   ❌ {name}: not installed")
    
    print(f"   📊 {success_count}/{len(required_packages)} dependencies available")
    return success_count == len(required_packages)

def check_project_structure():
    """Check project directory structure."""
    print("🏗️  Checking project structure...")
    required_paths = [
        'src/toaripi_slm/__init__.py',
        'configs/data/preprocessing_config.yaml',
        'configs/training/base_config.yaml',
        'configs/training/lora_config.yaml',
        'data/samples/sample_parallel.csv',
        'tests/test_installation.py',
        'requirements.txt',
        'requirements-dev.txt',
        'setup.py'
    ]
    
    success_count = 0
    for path in required_paths:
        if Path(path).exists():
            print(f"   ✅ {path}")
            success_count += 1
        else:
            print(f"   ❌ {path}")
    
    print(f"   📊 {success_count}/{len(required_paths)} required files found")
    return success_count == len(required_paths)

def check_sample_data():
    """Check sample data quality."""
    print("📊 Checking sample data...")
    try:
        df = pd.read_csv('data/samples/sample_parallel.csv')
        print(f"   ✅ Sample data loaded: {len(df)} rows")
        
        # Check columns
        required_cols = ['english', 'toaripi', 'verse_id', 'book', 'chapter']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"   ❌ Missing columns: {missing_cols}")
            return False
        else:
            print(f"   ✅ All required columns present")
        
        # Check data quality
        if df['english'].isna().any() or df['toaripi'].isna().any():
            print("   ❌ Data contains NaN values")
            return False
        else:
            print("   ✅ No missing values found")
        
        print("   📝 Sample entries:")
        for i, row in df.head(2).iterrows():
            print(f"      EN: {row['english']}")
            print(f"      TQO: {row['toaripi']}")
        
        return True
    except Exception as e:
        print(f"   ❌ Sample data check failed: {e}")
        return False

def check_configuration():
    """Check configuration files."""
    print("⚙️  Checking configuration files...")
    config_files = [
        'configs/data/preprocessing_config.yaml',
        'configs/training/base_config.yaml', 
        'configs/training/lora_config.yaml'
    ]
    
    success_count = 0
    for config_file in config_files:
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
            print(f"   ✅ {config_file}: valid YAML")
            success_count += 1
        except Exception as e:
            print(f"   ❌ {config_file}: {e}")
    
    return success_count == len(config_files)

def check_tests():
    """Run basic tests."""
    print("🧪 Running basic tests...")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/test_installation.py', '-v', '--tb=short'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("   ✅ All tests passed")
            return True
        else:
            print("   ❌ Some tests failed")
            print("   Test output:")
            print(result.stdout)
            if result.stderr:
                print("   Errors:")
                print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("   ❌ Tests timed out")
        return False
    except Exception as e:
        print(f"   ❌ Test execution failed: {e}")
        return False

def display_summary(checks):
    """Display setup summary."""
    print("\n" + "="*60)
    print("🎯 SETUP VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(checks.values())
    total = len(checks)
    
    for check_name, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check_name}")
    
    print("-"*60)
    print(f"📊 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Set up Hugging Face token: huggingface-cli login")
        print("2. Configure Weights & Biases: wandb login") 
        print("3. Add real Toaripi data to data/raw/")
        print("4. Start development server: uvicorn app.server:app --reload")
        print("5. Begin model training with sample data")
    else:
        print("⚠️  Setup incomplete. Please address failed checks above.")
    
    return passed == total

def main():
    """Run comprehensive setup verification."""
    print("🚀 Toaripi SLM Setup Verification")
    print("="*60)
    
    checks = {
        "Python Version": check_python_version(),
        "Virtual Environment": check_virtual_environment(),
        "Package Installation": check_package_installation(),
        "Dependencies": check_dependencies(),
        "Project Structure": check_project_structure(),
        "Sample Data": check_sample_data(),
        "Configuration": check_configuration(),
        "Basic Tests": check_tests(),
    }
    
    success = display_summary(checks)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
EOF

# Run comprehensive verification
python verify_setup.py
```

### 5.2 Create Development Shortcuts

```bash
# Create useful development scripts
mkdir -p scripts/dev

# Development server startup script
cat > scripts/dev/start_server.sh << 'EOF'
#!/bin/bash
# Start development server with proper environment

echo "🚀 Starting Toaripi SLM Development Server..."

# Activate virtual environment if not already active
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Create logs directory
mkdir -p logs

# Start server with development settings
echo "Starting server on http://localhost:8000"
uvicorn app.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level debug \
    --access-log
EOF

# Quick test script
cat > scripts/dev/quick_test.sh << 'EOF'
#!/bin/bash
# Run quick tests and checks

echo "🧪 Running Quick Tests..."

# Activate virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    source venv/bin/activate
fi

# Run basic tests
echo "Running installation tests..."
python -m pytest tests/test_installation.py -v

echo "Running code quality checks..."
# Format check (don't modify files)
black --check --diff src/ tests/ scripts/ || echo "ℹ️  Run 'black src/ tests/ scripts/' to format"

# Import sorting check
isort --check-only --diff src/ tests/ scripts/ || echo "ℹ️  Run 'isort src/ tests/ scripts/' to sort imports"

# Linting
flake8 src/ tests/ scripts/ || echo "ℹ️  Fix linting issues above"

echo "✅ Quick tests completed"
EOF

# Make scripts executable
chmod +x scripts/dev/*.sh

echo "✅ Development scripts created"
```

### 5.3 Documentation and Next Steps

```bash
# Create development notes
cat > DEVELOPMENT.md << 'EOF'
# Toaripi SLM Development Notes

## Quick Start Commands

```bash
# Activate environment
source venv/bin/activate

# Start development server
./scripts/dev/start_server.sh

# Run tests
./scripts/dev/quick_test.sh

# Format code
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Install new dependencies
pip install package_name
pip freeze > requirements.txt
```

## Development Workflow

1. **Data Preparation**
   ```bash
   # Add real data to data/raw/toaripi_bible.csv
   python scripts/prepare_data.py --config configs/data/preprocessing_config.yaml
   ```

2. **Model Training**
   ```bash
   # Train small model for testing
   python scripts/finetune.py --config configs/training/base_config.yaml
   
   # Train with LoRA for efficiency
   python scripts/finetune.py --config configs/training/lora_config.yaml
   ```

3. **Content Generation**
   ```bash
   # Generate educational content
   python scripts/generate.py --model_path checkpoints/latest --prompt "Create a story about fishing"
   ```

4. **Model Export**
   ```bash
   # Export to GGUF for edge deployment
   python scripts/export_gguf.py --model_path checkpoints/latest --output models/gguf/
   ```

## Project Status

### ✅ Completed
- [x] WSL environment setup
- [x] Python environment and dependencies
- [x] Project structure and configuration
- [x] Sample data and basic tests
- [x] Code quality tools setup

### 🚧 In Progress
- [ ] Core training pipeline implementation
- [ ] Content generation modules
- [ ] Web API development
- [ ] Model export utilities

### 📋 Planned
- [ ] Real Toaripi data integration
- [ ] Advanced training features
- [ ] Educational content templates
- [ ] Edge deployment optimization
- [ ] Community documentation

## Useful Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PEFT (LoRA) Guide](https://huggingface.co/docs/peft/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Weights & Biases Guides](https://docs.wandb.ai/)

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Reinstall package in development mode
   pip install -e .
   ```

2. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in configs/training/*.yaml
   # Use gradient accumulation instead
   ```

3. **Dependency Conflicts**
   ```bash
   # Create fresh environment
   deactivate
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements-dev.txt
   ```

## Contributing

1. Create feature branch: `git checkout -b feature-name`
2. Make changes and test: `./scripts/dev/quick_test.sh`
3. Commit with pre-commit hooks: `git commit -m "Description"`
4. Push and create pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.
EOF

# Final status message
echo "
🎉 Toaripi SLM Project Successfully Initialized in WSL!

📁 Project Location: $(pwd)
🐍 Python Version: $(python --version)
📦 Package: $(python -c 'import src.toaripi_slm; print(src.toaripi_slm.__version__)')"

echo "
🚀 Next Steps:

1. Set up external services:
   huggingface-cli login
   wandb login

2. Start development:
   ./scripts/dev/start_server.sh

3. Run tests:
   ./scripts/dev/quick_test.sh

4. Begin development:
   - Add real Toaripi data to data/raw/
   - Implement core training modules
   - Develop content generation features

📚 Documentation:
   - Setup Guide: docs/setup/WSL_SETUP_GUIDE.md
   - Development: DEVELOPMENT.md
   - Main README: readme.md

Happy coding! 🎯 Let's preserve and promote the Toaripi language through technology.
"
```

---

## 📚 Summary

This comprehensive WSL setup guide provides:

1. **Complete WSL2 Environment** - Ubuntu with Python 3.11 and development tools
2. **Project Structure** - Full directory hierarchy with proper Python packaging
3. **Dependencies** - All required ML libraries and development tools
4. **Configuration** - Training configs, data processing, and deployment settings
5. **Sample Data** - Test datasets for development and validation
6. **Testing Framework** - Comprehensive test suites and verification scripts
7. **Code Quality** - Pre-commit hooks, formatting, and linting tools
8. **Development Tools** - Helper scripts and development workflows
9. **Documentation** - Setup verification and development guides

The project is now ready for:
- ✅ Model training and fine-tuning
- ✅ Educational content generation
- ✅ Web API development
- ✅ Edge deployment preparation
- ✅ Community collaboration

You can now begin implementing the core Toaripi SLM functionality with a solid foundation! 🚀