# Toaripi Small Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **A small language model for generating educational content in Toaripi language**

**Status:** Research Prototype | **Language:** Toaripi (East Elema), Gulf Province, Papua New Guinea (ISO 639‑3: `tqo`)

This project develops a **small language model (SLM)** for the Toaripi language to support **language preservation** and **education**. The model is trained on aligned English↔Toaripi Bible text and designed to generate original educational content for primary school learners and teachers.

## 🎯 About the project

**Empower local educators with AI tools** that create culturally relevant learning materials while preserving the Toaripi language through:

- 📚 **Original educational content generation** (stories, vocabulary, Q&A, dialogues)
- 🌐 **Online and offline deployment** (including Raspberry Pi and low-resource devices)
- 🤝 **Open-source collaboration** with Toaripi speakers, linguists, and developers
- 🎓 **Educational focus** over general-purpose chatbot functionality

## ✨ Key Features

### 🧠 **Smart Content Generation**
- Create educational stories and vocabulary exercises
- Generate reading comprehension questions and dialogues
- Produce culturally relevant educational materials in Toaripi
- Support primary school curriculum development

### 💻 **Flexible Deployment**
- **Online**: Web UI and REST API for connected environments
- **Offline**: Quantized models (GGUF) for CPU-only devices
- **Edge**: Optimized for Raspberry Pi and low-resource hardware

### 🔧 **Developer-Friendly**
- Modular Python architecture with clear APIs
- Comprehensive configuration management (YAML/TOML)
- Docker support for easy deployment
- Extensive testing and documentation

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ (3.11 recommended)
- 16GB RAM minimum (32GB recommended for training)
- 50GB+ free storage space
- Git for version control

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/toaripi-slm.git
cd toaripi-slm

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# source venv/bin/activate    # macOS/Linux

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Verify Installation

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

## 🏗️ Project Structure

```
toaripi-slm/
├── app/                    # Web application and API
│   ├── api/               # REST API endpoints
│   ├── config/            # App configuration
│   └── ui/                # Web interface
├── configs/               # Training and deployment configs
│   ├── data/              # Data processing configurations
│   ├── deployment/        # Deployment configurations
│   └── training/          # Model training configurations
├── data/                  # Data storage
│   ├── processed/         # Cleaned and aligned data
│   ├── raw/              # Source data files
│   └── samples/          # Sample datasets
├── docs/                  # Documentation
├── models/               # Trained models
│   ├── gguf/             # Quantized models for edge deployment
│   └── hf/               # Hugging Face format models
├── scripts/              # Training and utility scripts
├── src/toaripi_slm/      # Core library code
│   ├── core/             # Model and training components
│   ├── data/             # Data processing utilities
│   ├── inference/        # Model inference and generation
│   └── utils/            # Helper functions
└── tests/                # Test suites
```

## 🤖 Usage Examples

### Generate Educational Content

```python
from src.toaripi_slm.inference import ToaripiGenerator

# Initialize generator
generator = ToaripiGenerator.load("models/toaripi-mistral-lora")

# Generate a story
story = generator.generate_story(
    prompt="Children helping parents with fishing",
    age_group="primary",
    max_length=200
)

# Generate vocabulary exercise
vocab = generator.generate_vocabulary(
    topic="household items",
    count=10,
    include_examples=True
)
```

### Web API Usage

```bash
# Start the web server
uvicorn app.server:app --host 0.0.0.0 --port 8000

# Generate content via API
curl -X POST "http://localhost:8000/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a simple story about family dinner",
    "content_type": "story",
    "age_group": "primary",
    "max_length": 150
  }'
```

## 🔧 Development

For detailed development setup, training procedures, and contribution guidelines, see:

- **[Development Setup Guide](devsetup.md)** - Complete environment setup
- **[docs/contributing/](docs/contributing/)** - Training procedures and contribution guidelines
- **[docs/usage/](docs/usage/)** - Usage documentation and examples

### Training Your Own Model

```bash
# Prepare training data
python scripts/prepare_data.py \
  --config configs/data/preprocessing_config.yaml \
  --output data/processed/toaripi_parallel.csv

# Fine-tune with LoRA
python scripts/finetune.py \
  --config configs/training/lora_config.yaml \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --train_data data/processed/toaripi_parallel.csv \
  --output_dir checkpoints/toaripi-mistral-lora

# Export to GGUF for edge deployment
python scripts/export_to_gguf.py \
  --model_path checkpoints/toaripi-mistral-lora \
  --output_dir models/gguf \
  --quantization q4_k_m
```

## 🌐 Deployment Options

### Local Development
```bash
# Development server with auto-reload
uvicorn app.server:app --reload --port 8000
```

### Docker Deployment
```bash
# Build and run with Docker
docker build -t toaripi-slm .
docker run -p 8000:8000 toaripi-slm

# Or use Docker Compose
docker-compose up -d
```

### Raspberry Pi / Edge Devices
```bash
# Build ARM-compatible image
docker build -f Dockerfile.pi -t toaripi-slm:pi .

# Run quantized model on CPU
python scripts/run_gguf.py \
  --model_path models/gguf/toaripi-mistral-q4_k_m.gguf \
  --host 0.0.0.0 --port 8000
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/toaripi_slm --cov-report=html

# Run specific test categories
pytest tests/unit/ -v           # Unit tests
pytest tests/integration/ -v    # Integration tests
```

## 🤝 Contributing

We welcome contributions from developers, linguists, educators, and the Toaripi community! See our [Contributing Guidelines](CONTRIBUTING.md) for:

- Setting up the development environment
- Code style and testing requirements
- How to submit issues and pull requests
- Guidelines for adding training data
- Community code of conduct

## 📊 Tech Stack

- **Language**: Python 3.10+
- **ML Libraries**: `transformers`, `datasets`, `accelerate`, `peft` (LoRA)
- **Web Framework**: `fastapi` + `uvicorn`
- **Edge Inference**: `llama.cpp` (GGUF quantized models)
- **Configuration**: YAML/TOML
- **Testing**: `pytest`, `pre-commit`
- **Deployment**: Docker, Docker Compose

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Toaripi Community**: For language knowledge and cultural guidance
- **Bible Society PNG**: For providing aligned text resources
- **Open Source Contributors**: For tools and libraries that make this possible
- **Educational Partners**: For curriculum guidance and feedback

---

**Preserving language and empowering education through technology.**