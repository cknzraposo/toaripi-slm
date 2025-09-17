# Toaripi SLM — Educational Content Generator 🏫



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


> **A small language model for generating educational content in Toaripi language**> **A small language model for generating educational content in Toaripi language**



**Status:** Research Prototype | **Language:** Toaripi (East Elema), Gulf Province, Papua New Guinea (ISO 639‑3: `tqo`)**Status:** Research Prototype | **Language:** Toaripi (East Elema), Gulf Province, Papua New Guinea (ISO 639‑3: `tqo`)



This project develops a **small language model (SLM)** for the Toaripi language to support **language preservation** and **education**. The model is trained on aligned English↔Toaripi Bible text and designed to generate original educational content for primary school learners and teachers.This project develops a **small language model (SLM)** for the Toaripi language to support **language preservation** and **education**. The model is trained on aligned English↔Toaripi Bible text and designed to generate original educational content for primary school learners and teachers.



## 🎯 Project Mission## 🎯 Project Mission



**Empower local educators with AI tools** that create culturally relevant, age-appropriate learning materials while preserving the Toaripi language through:**Empower local educators with AI tools** that create culturally relevant, age-appropriate learning materials while preserving the Toaripi language through:



- 📚 **Original educational content generation** (stories, vocabulary, Q&A, dialogues)- 📚 **Original educational content generation** (stories, vocabulary, Q&A, dialogues)

- 🌐 **Online and offline deployment** (including Raspberry Pi and low-resource devices)- 🌐 **Online and offline deployment** (including Raspberry Pi and low-resource devices)  

- 🤝 **Open-source collaboration** with Toaripi speakers, linguists, and developers- 🤝 **Open-source collaboration** with Toaripi speakers, linguists, and developers

- 🎓 **Educational focus** over general-purpose chatbot functionality- 🎓 **Educational focus** over general-purpose chatbot functionality



## ✨ Key FeaturesThis project focuses on creating a small AI language model (SLM) for the Toaripi language of Papua New Guinea to support language preservation and education. It uses aligned English–Toaripi Bible text as training data and aims to:



### 🧠 **Smart Content Generation**Generate original educational content (stories, vocabulary lists, comprehension questions) for primary school learners and teachers.

- Create age-appropriate stories and vocabulary exercisesWork both online and offline, including on low-resource devices like Raspberry Pi, through model quantization and lightweight deployment.

- Generate reading comprehension questions and dialoguesBe open-source, hosted on GitHub with clear documentation, scripts, and contribution guidelines to invite collaboration from developers, linguists, and the Toaripi-speaking community.

- Produce culturally relevant educational materials in Toaripi

The ultimate goal is to empower local educators with AI tools that create culturally relevant, age-appropriate learning materials while preserving the Toaripi language.

### 💻 **Flexible Deployment**

- **Online**: Web UI and REST API for connected environments**Status:** research prototype • **Focus:** Toaripi (East Elema) language, Gulf Province, Papua New Guinea (ISO 639‑3: `tqo`)  

- **Offline**: Quantized models (GGUF) for CPU-only devices**Purpose:** A small language model (SLM) for **language preservation** and **education**, trained primarily from parallel **English↔Toaripi Bible** text (aligned by chapter/verse). The model is designed to **generate original educational content** (e.g., simple stories, vocabulary lists, reading comprehension questions) suitable for **primary school learners**, and to operate **both online and offline** in low‑connectivity environments.

- **Edge**: Optimized for Raspberry Pi and low-resource hardware

> *This repository will host the code, data tooling, model checkpoints (or pointers), and deployment patterns for educators, local contributors, and researchers.*

### 🔧 **Developer-Friendly**

- Modular Python architecture with clear APIs

- Comprehensive configuration management## Why this project?

- Docker support for easy deployment

- Extensive testing and documentation- **Language preservation:** Support Toaripi literacy by bootstrapping written content and teaching resources.

- **Education first:** Help teachers quickly create exercises, stories and quizzes **in Toaripi**, not just translations.

## 🚀 Quick Start- **Small & efficient:** Use **small language models (≤7B params)** to enable **offline** use on modest hardware (e.g., low‑cost laptops, Raspberry Pi).

- **Open collaboration:** Publish as an open repo to enable contributions from **Toaripi speakers**, educators, linguists, and developers.

### Prerequisites

- Python 3.10 or higher---

- 8GB+ RAM (16GB recommended for training)

- Optional: CUDA-compatible GPU for faster training## Project goals



### Installation1. **Build a Toaripi-capable small language model**  

   Fine‑tune a compact, open model (e.g., 1–7B parameters) using **aligned English↔Toaripi** scripture to learn vocabulary and structure.

1. **Clone the repository**

   ```bash2. **Generate original learning materials (not scripture regurgitation)**  

   git clone https://github.com/your-org/toaripi-slm.git   Produce **age‑appropriate** Toaripi content: vocabulary drills, simple narratives, fill‑in‑the‑blank, Q&A, dialogues.

   cd toaripi-slm

   ```3. **Accessible online and offline**  

   - **Online:** Simple web UI/API for content generation where internet exists.  

2. **Set up Python environment**   - **Offline:** Quantised inference (e.g., 4‑bit) on **CPU‑only** devices; minimise dependencies to run in rural schools.

   ```bash

   python -m venv venv4. **Community‑centred & open**  

   source venv/bin/activate  # On Windows: venv\Scripts\activate   Publish code and reproducible pipelines; document data sourcing & permissions; welcome community corrections and content contributions.

   pip install -r requirements.txt

   ```**Non‑goals (for clarity):**  

- This is **not** a theological tool. Scripture is used **as bilingual training data** to teach the model Toaripi.  

3. **Install the package**- This is **not** a general‑purpose chatbot. The focus is **educational content generation** in Toaripi.

   ```bash

   pip install -e .---
   ```

### Basic Usage

#### Generate Educational Content
```python
from src.toaripi_slm.core.generator import ContentGenerator

# Initialize the generator
generator = ContentGenerator(model_path="models/toaripi-slm-7b")

# Generate a simple story
story = generator.generate_story(
    topic="fishing in the river",
    age_group="primary",
    length="short"
)

# Create vocabulary exercises
vocab = generator.generate_vocabulary(
    theme="family",
    exercise_type="matching",
    difficulty="beginner"
)
```

#### Command Line Interface
```bash
# Prepare training data
toaripi-prepare-data --english_source web_kjv --toaripi_source local_csv --toaripi_path data/raw/toaripi_bible.csv

# Fine-tune the model
toaripi-finetune --config configs/training/base_config.yaml --output checkpoints/toaripi-model

# Generate content
toaripi-generate --model checkpoints/toaripi-model --prompt "Create a story about children playing" --type story
```

#### Web Interface
```bash
# Start the web application
uvicorn app.server:app --host 0.0.0.0 --port 8000

# Access at http://localhost:8000
```

## 📁 Project Structure

```
toaripi-slm/
├── app/                    # Web application (FastAPI)
│   ├── api/               # REST API endpoints
│   ├── ui/                # Web interface
│   └── config/            # App configuration
├── src/toaripi_slm/       # Core library
│   ├── core/              # Model and training logic
│   ├── data/              # Data processing utilities
│   ├── inference/         # Model inference
│   └── utils/             # Helper functions
├── scripts/               # Training and utility scripts
├── configs/               # Configuration files
│   ├── training/          # Model training configs
│   ├── data/              # Data processing configs
│   └── deployment/        # Deployment configs
├── data/                  # Data directory
│   ├── raw/               # Source data
│   ├── processed/         # Processed training data
│   └── samples/           # Sample data
├── models/                # Model storage
│   ├── hf/                # Hugging Face format
│   └── gguf/              # Quantized models
├── docs/                  # Documentation
├── tests/                 # Test suite
└── tools/                 # Development tools
```

## 🔧 Development

### Setting Up Development Environment

1. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

3. **Run tests**
   ```bash
   pytest tests/ -v
   ```

4. **Code formatting**
   ```bash
   black src/ scripts/ app/
   isort src/ scripts/ app/
   ```

### Training Your Own Model

1. **Prepare your data**
   ```bash
   python scripts/prepare_data.py --config configs/data/preprocessing_config.yaml
   ```

2. **Start training**
   ```bash
   python scripts/finetune.py --config configs/training/lora_config.yaml
   ```

3. **Export to GGUF for deployment**
   ```bash
   python scripts/export_to_gguf.py --model_path checkpoints/your-model --output models/gguf/
   ```

## 🌐 Deployment Options

### Local Development
```bash
# Using Docker
docker-compose up -d

# Direct Python
uvicorn app.server:app --reload
```

### Raspberry Pi Deployment
```bash
# Build Pi-optimized image
docker build -f Dockerfile.pi -t toaripi-slm:pi .

# Run with CPU-only inference
docker run -p 8000:8000 -v ./models:/app/models toaripi-slm:pi
```

### Cloud Deployment
- AWS/Azure: Use provided deployment configs
- Heroku: Ready-to-deploy with included `Procfile`
- Docker: Multi-stage builds for production

## 📊 Model Performance

| Model Size | Training Time | Inference Speed | Memory Usage | BLEU Score* |
|------------|---------------|-----------------|--------------|-------------|
| 1B params  | ~4 hours      | 50 tokens/sec   | 2GB         | TBD         |
| 3B params  | ~12 hours     | 25 tokens/sec   | 6GB         | TBD         |
| 7B params  | ~24 hours     | 10 tokens/sec   | 14GB        | TBD         |

*Evaluation metrics are being developed for educational content quality

## 🤝 Contributing

We welcome contributions from:
- **Toaripi speakers**: Language validation, cultural context, content review
- **Educators**: Curriculum alignment, age-appropriateness feedback
- **Linguists**: Language structure insights, evaluation metrics
- **Developers**: Code improvements, new features, bug fixes

### How to Contribute

1. **Check existing issues** or create a new one
2. **Fork the repository** and create a feature branch
3. **Follow our coding standards** (black, isort, type hints)
4. **Add tests** for new functionality
5. **Submit a pull request** with clear description

See [CONTRIBUTING.md](docs/contributing/guidelines.md) for detailed guidelines.

## 📚 Documentation

- **[Installation Guide](docs/setup/installation.md)**: Detailed setup instructions
- **[User Manual](docs/usage/)**: How to use the model and tools
- **[API Reference](docs/usage/api-reference.md)**: Complete API documentation
- **[Training Guide](docs/usage/training.md)**: Custom model training
- **[Deployment Guide](docs/setup/)**: Production deployment options
- **[Research Notes](docs/research/)**: Methodology and evaluation

## 🛡️ Ethical Considerations

### Data and Privacy
- Uses publicly available Biblical text as training data
- No personal data collection in the application
- Transparent data sourcing and processing

### Cultural Sensitivity
- Designed **with** and **for** the Toaripi community
- Focus on educational content, not religious interpretation
- Respect for traditional knowledge and cultural context

### Model Limitations
- Trained on limited domain data (Biblical text)
- May not capture full linguistic diversity
- Requires community validation for cultural accuracy

## 🎓 Educational Use Cases

### For Teachers
- Generate vocabulary exercises and quizzes
- Create reading comprehension activities
- Develop culturally relevant story content
- Prepare lesson plans with Toaripi materials

### For Students
- Practice reading with generated stories
- Learn new vocabulary in context
- Engage with interactive language exercises
- Develop writing skills through prompts

### For Community
- Support literacy programs
- Create community newsletters
- Develop cultural preservation materials
- Enable digital storytelling

## 🔬 Research & Evaluation

### Current Research Questions
- How effective are small models for low-resource languages?
- What training strategies work best for educational content generation?
- How can we measure cultural appropriateness in generated content?

### Evaluation Metrics
- **Language Quality**: Fluency, grammatical correctness
- **Educational Value**: Age-appropriateness, curriculum alignment
- **Cultural Relevance**: Community feedback, expert review
- **Technical Performance**: Speed, memory usage, reliability

## 📋 Roadmap

### Phase 1: Foundation (Current)
- [x] Project setup and documentation
- [x] Basic model training pipeline
- [x] Simple content generation
- [ ] Community feedback integration

### Phase 2: Enhancement
- [ ] Advanced content generation features
- [ ] Multi-modal support (images, audio)
- [ ] Mobile application
- [ ] Teacher training materials

### Phase 3: Expansion
- [ ] Support for related languages
- [ ] Advanced evaluation metrics
- [ ] Community contribution platform
- [ ] Educational impact studies

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Toaripi Community**: For language knowledge and cultural guidance
- **Bible Society PNG**: For providing aligned text resources
- **Open Source Contributors**: For tools and libraries that make this possible
- **Educational Partners**: For curriculum guidance and feedback

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/your-org/toaripi-slm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/toaripi-slm/discussions)
- **Email**: toaripi-slm@example.com
- **Documentation**: [Project Docs](https://toaripi-slm.readthedocs.io)

---

**Together, we're preserving language and empowering education through technology.** 🌟