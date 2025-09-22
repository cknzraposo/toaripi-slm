# GitHub Copilot Instructions — Toaripi SLM (Educational Content Generator)
**Note:** This file provides context and instructions for GitHub Copilot to assist with code generation in this repository. It is not part of the user-facing documentation.
## 🧭 Project Context (what Copilot should know)

- **Mission:** Build a **small language model (SLM)** for **Toaripi (ISO 639‑3: `tqo`)** to generate **original educational content** (stories, vocabulary, Q&A, dialogues) for primary learners and teachers.
- **Approach:** Fine‑tune a compact open model (≈1–7B params) on **aligned English↔Toaripi Bible** data; run **online and offline** (Raspberry Pi / CPU‑only) using **quantization** and **llama.cpp**.
- **Users:** Teachers, Toaripi speakers, contributors (linguists, devs) with varying technical experience.
- **Non‑goals:** Theology tooling / doctrinal outputs; general-purpose chatbot. Focus is **education & language preservation**.

---

## 🧰 Tech Stack & Key Tools

- **Language:** Python 3.10+
- **Core libs:** `transformers`, `datasets`, `accelerate`, `peft` (LoRA), `sentencepiece`/tokenizers
- **Serving/UI:** `fastapi` + `uvicorn` (or `streamlit` for quick demo)
- **Edge inference:** `llama.cpp` (GGUF quantized weights)
- **Data formats:** CSV/TSV for parallel verses; optional USFM ingestion
- **Config:** YAML/TOML for data sources & training params

---

## 📁 Repository Shape (expected files and roles)

```
toaripi-slm/
├── src/toaripi_slm/           # Main library code
│   ├── core/                  # Model training & fine-tuning
│   ├── data/                  # Data processing & alignment
│   ├── inference/             # Generation & serving
│   └── utils/                 # Helper functions
├── app/                       # Web application
│   ├── api/                   # FastAPI endpoints
│   ├── ui/                    # Web interface
│   └── config/                # App configuration
├── scripts/                   # Training & utility scripts
├── configs/                   # YAML/TOML configuration files
├── data/                      # Training data (Bible texts)
├── models/                    # Trained models & checkpoints
│   ├── hf/                    # HuggingFace format
│   └── gguf/                  # Quantized for edge deployment
└── tests/                     # Test suites
```

---

## 🎯 Code Generation Guidelines

### **Educational Content Focus**
- Always prioritize **educational content generation** over general chat capabilities
- Generate code for creating: stories, vocabulary exercises, Q&A, reading comprehension, dialogues
- Include **age-appropriate content** parameters (primary school focus)
- Consider **cultural relevance** and **language preservation** goals

### **Language Model Architecture**
```python
# Prefer small, efficient models (1-7B parameters)
model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # Good choice
# model_name = "meta-llama/Llama-2-70b-chat-hf"    # Too large

# Use LoRA for efficient fine-tuning
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # Low rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)
```

### **Data Processing Patterns**
```python
# Always expect English↔Toaripi parallel data
def load_parallel_data(csv_path: str) -> pd.DataFrame:
    """Load aligned English-Toaripi text pairs."""
    df = pd.read_csv(csv_path)
    required_cols = ["english", "toaripi"]
    assert all(col in df.columns for col in required_cols)
    return df

# Prepare training prompts for educational content
def format_educational_prompt(english_text: str, toaripi_text: str, 
                            content_type: str = "story") -> str:
    return f"""Create educational content in Toaripi language.

English reference: {english_text}
Toaripi translation: {toaripi_text}

Generate a {content_type} suitable for primary school students."""
```

### **Configuration Management**
```python
# Use YAML for all configuration files
import yaml
from pathlib import Path
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_name: str
    learning_rate: float = 2e-5
    batch_size: int = 4
    epochs: int = 3
    use_lora: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: Path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
```

### **Edge Deployment Considerations**
```python
# Always include GGUF export capabilities
def export_to_gguf(model_path: str, output_dir: str, 
                   quantization: str = "q4_k_m"):
    """Export model for Raspberry Pi/CPU deployment."""
    # Use llama.cpp for quantization
    pass

# Optimize for low-resource environments
def create_cpu_optimized_config():
    return {
        "max_memory": {"cpu": "8GB"},
        "device_map": "cpu",
        "torch_dtype": "float16",
        "quantization_config": {"load_in_4bit": True}
    }
```

---

## 🔧 Development Patterns

### **Module Structure**
```python
# src/toaripi_slm/core/trainer.py
class ToaripiTrainer:
    """Fine-tune models for Toaripi educational content."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
    
    def load_model(self, model_name: str):
        """Load base model for fine-tuning."""
        pass
    
    def prepare_dataset(self, parallel_data: pd.DataFrame):
        """Prepare training data with educational prompts."""
        pass

# src/toaripi_slm/inference/generator.py  
class ToaripiGenerator:
    """Generate educational content in Toaripi."""
    
    def generate_story(self, prompt: str, age_group: str = "primary") -> str:
        """Generate educational story."""
        pass
    
    def generate_vocabulary(self, topic: str, count: int = 10) -> List[Dict]:
        """Generate vocabulary exercises."""
        pass
    
    def generate_comprehension(self, text: str) -> List[str]:
        """Generate reading comprehension questions."""
        pass
```

### **API Endpoints**
```python
# app/api/generate.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

class GenerationRequest(BaseModel):
    prompt: str
    content_type: str = "story"  # story, vocabulary, dialogue, qa
    age_group: str = "primary"   # primary, secondary
    max_length: int = 200
    temperature: float = 0.7

@router.post("/api/generate")
async def generate_content(request: GenerationRequest):
    """Generate educational content in Toaripi."""
    # Always validate content is educational and appropriate
    # Include cultural sensitivity checks
    pass
```

### **Testing Patterns**
```python
# tests/test_generation.py
import pytest
from src.toaripi_slm.inference import ToaripiGenerator

class TestToaripiGeneration:
    def test_story_generation_primary_level(self):
        """Test story generation for primary school."""
        generator = ToaripiGenerator.load("test_model")
        story = generator.generate_story(
            prompt="Children helping with fishing",
            age_group="primary"
        )
        
        assert len(story) > 0
        assert story.count('.') >= 3  # Multiple sentences
        # Add language-specific validation
    
    def test_vocabulary_generation(self):
        """Test vocabulary exercise generation."""
        pass
```

---

## 🚫 Important Constraints

### **Content Guidelines**
- **NO theological content generation** - avoid religious interpretations
- **NO general-purpose chatbot features** - focus only on education
- **NO adult content** - all outputs must be primary school appropriate
- **NO cultural appropriation** - respect Toaripi cultural context

### **Technical Constraints**
- **Model size limit**: ≤7B parameters for edge deployment
- **Memory limit**: Must run on 8GB RAM minimum
- **Storage limit**: Models must fit in <5GB when quantized
- **Offline requirement**: Must work without internet connection

### **Data Handling**
```python
# Always validate data sources
def validate_parallel_data(df: pd.DataFrame):
    """Ensure data quality and cultural appropriateness."""
    assert "english" in df.columns
    assert "toaripi" in df.columns
    assert len(df) > 100  # Minimum data size
    
    # Check for inappropriate content
    inappropriate_words = ["violence", "adult", "religious_doctrine"]
    for col in ["english", "toaripi"]:
        text_content = " ".join(df[col].str.lower())
        for word in inappropriate_words:
            assert word not in text_content
```

---

## 🎓 Educational Content Templates

### **Story Generation**
```python
STORY_TEMPLATE = """
Write a simple story in Toaripi about {topic}.
Target audience: {age_group} students
Length: {length} sentences
Include: {elements}

The story should teach about {learning_objective}.
Use simple vocabulary and clear sentences.
"""

VOCABULARY_TEMPLATE = """
Create a vocabulary list for {topic}:
- {count} words in Toaripi
- English translations
- Example sentences
- Suitable for {age_group} learners
"""
```

### **Assessment Generation**
```python
def generate_comprehension_questions(text: str, count: int = 3):
    """Generate reading comprehension questions."""
    question_types = [
        "Who did what?",
        "Where did it happen?", 
        "Why did it happen?",
        "What happened next?"
    ]
    # Ensure age-appropriate difficulty
```

---

## 🛠️ CLI Command Patterns

When generating CLI scripts, use this pattern:
```python
# scripts/finetune.py
import click
from pathlib import Path

@click.command()
@click.option('--config', type=Path, required=True, help='Training config YAML')
@click.option('--data', type=Path, required=True, help='Parallel data CSV')
@click.option('--output-dir', type=Path, required=True, help='Model output directory')
@click.option('--model-name', default='mistralai/Mistral-7B-Instruct-v0.2')
def finetune(config: Path, data: Path, output_dir: Path, model_name: str):
    """Fine-tune Toaripi educational content model."""
    pass

if __name__ == "__main__":
    finetune()
```

---

## 📦 Deployment Patterns

### **Docker Configuration**
```dockerfile
# Dockerfile optimized for edge deployment
FROM python:3.11-slim

# Install minimal dependencies for Toaripi SLM
RUN pip install torch==2.0.0+cpu transformers accelerate peft

# Copy only necessary files
COPY src/ /app/src/
COPY models/gguf/ /app/models/

# Run with CPU optimization
CMD ["python", "-m", "app.server", "--cpu-only", "--model-path", "/app/models/toaripi-q4.gguf"]
```

### **Raspberry Pi Optimization**
```python
# Optimize for ARM64 and limited resources
def get_pi_config():
    return {
        "device_map": "cpu",
        "torch_dtype": torch.float16,
        "max_memory": {"cpu": "4GB"},
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16
        }
    }
```

Remember: **Education first, technology second.** Always prioritize the learning outcomes for Toaripi students and cultural preservation over technical sophistication.
