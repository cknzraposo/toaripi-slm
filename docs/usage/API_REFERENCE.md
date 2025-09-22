# API Reference

## Overview

The Toaripi SLM provides both programmatic APIs and CLI interfaces for educational content generation. This reference covers the Python API for developers who want to integrate Toaripi functionality into their applications.

## Installation

```bash
pip install toaripi-slm
```

## Core APIs

### Data Processing API

#### ToaripiDataProcessor

```python
from toaripi_slm.data.preprocessing import ToaripiDataProcessor
from toaripi_slm.models.config import PreprocessingConfig
from toaripi_slm.models.common import AgeGroup, ContentType

# Initialize processor
config = PreprocessingConfig(
    source_file="data/parallel.csv",
    target_age_groups=[AgeGroup.PRIMARY_LOWER],
    content_types=[ContentType.STORY, ContentType.VOCABULARY],
    validation_level="educational"
)

processor = ToaripiDataProcessor(config)

# Load and validate data
data = processor.load_parallel_data()
validated_data = processor.validate_educational_content(data)

# Prepare training data
train_data = processor.prepare_training_data(validated_data)
```

#### Methods

**`load_parallel_data() -> pd.DataFrame`**
- Load English-Toaripi parallel text data
- Returns: DataFrame with 'english' and 'toaripi' columns
- Raises: `DataLoadError` if file format invalid

**`validate_educational_content(data: pd.DataFrame) -> ValidationResult`**
- Validate content for educational appropriateness
- Parameters:
  - `data`: DataFrame with parallel text
- Returns: `ValidationResult` with validation details
- Educational checks: age appropriateness, cultural sensitivity, content safety

**`prepare_training_data(data: pd.DataFrame) -> TrainingDataset`**
- Prepare data for model training
- Parameters:
  - `data`: Validated parallel text data
- Returns: `TrainingDataset` ready for model training
- Features: Educational prompt formatting, age group targeting

### Model Training API

#### ToaripiTrainer

```python
from toaripi_slm.core.trainer import ToaripiTrainer
from toaripi_slm.models.config import TrainingConfig

# Initialize trainer
config = TrainingConfig(
    model_name="microsoft/DialoGPT-small",
    epochs=3,
    batch_size=4,
    learning_rate=2e-5,
    target_age_groups=[AgeGroup.PRIMARY_LOWER],
    educational_validation=True
)

trainer = ToaripiTrainer(config)

# Train model
training_result = trainer.train(train_dataset)
```

#### Methods

**`train(dataset: TrainingDataset) -> TrainingResult`**
- Train Toaripi educational model
- Parameters:
  - `dataset`: Prepared training data
- Returns: `TrainingResult` with training metrics and model path
- Features: Educational content validation during training

**`load_model(model_path: str) -> ToaripiModel`**
- Load pre-trained Toaripi model
- Parameters:
  - `model_path`: Path to model files
- Returns: `ToaripiModel` ready for inference
- Supports: HuggingFace format, GGUF quantized models

**`save_checkpoint(model: ToaripiModel, checkpoint_dir: str) -> str`**
- Save training checkpoint
- Parameters:
  - `model`: Current model state
  - `checkpoint_dir`: Directory for checkpoint
- Returns: Path to saved checkpoint
- Includes: Model weights, training state, educational metadata

### Model Inference API

#### ToaripiGenerator

```python
from toaripi_slm.inference.generator import ToaripiGenerator
from toaripi_slm.models.common import AgeGroup, ContentType

# Initialize generator
generator = ToaripiGenerator(
    model_path="models/toaripi-primary.gguf",
    age_group=AgeGroup.PRIMARY_LOWER,
    validation_level="strict"
)

# Generate educational content
story = generator.generate_story(
    prompt="Children learning to fish",
    age_group=AgeGroup.PRIMARY_LOWER,
    max_length=200
)

vocabulary = generator.generate_vocabulary(
    topic="ocean animals",
    count=10,
    age_group=AgeGroup.PRIMARY_LOWER
)
```

#### Methods

**`generate_story(prompt: str, age_group: AgeGroup, **kwargs) -> GeneratedStory`**
- Generate educational story in Toaripi
- Parameters:
  - `prompt`: Story theme or starting point
  - `age_group`: Target age group for vocabulary and complexity
  - `max_length`: Maximum story length (default: 200 tokens)
  - `temperature`: Generation randomness (default: 0.7)
- Returns: `GeneratedStory` with text, metadata, and validation scores
- Features: Age-appropriate vocabulary, cultural validation, educational objectives

**`generate_vocabulary(topic: str, count: int, age_group: AgeGroup) -> VocabularyList`**
- Generate vocabulary exercises
- Parameters:
  - `topic`: Vocabulary theme (e.g., "animals", "family", "nature")
  - `count`: Number of words to generate
  - `age_group`: Target learner age group
- Returns: `VocabularyList` with words, definitions, examples
- Features: Thematic grouping, difficulty scaling, cultural context

**`generate_dialogue(scenario: str, age_group: AgeGroup, **kwargs) -> GeneratedDialogue`**
- Generate conversational dialogues
- Parameters:
  - `scenario`: Conversation context
  - `age_group`: Target age group
  - `participants`: Number of speakers (default: 2)
- Returns: `GeneratedDialogue` with conversation and educational notes
- Features: Natural speech patterns, cultural communication styles

**`generate_comprehension(text: str, question_count: int) -> ComprehensionQuestions`**
- Generate reading comprehension questions
- Parameters:
  - `text`: Source text for questions
  - `question_count`: Number of questions to generate
- Returns: `ComprehensionQuestions` with questions and answer guidelines
- Features: Age-appropriate question complexity, multiple question types

### Model Management API

#### ToaripiModelManager

```python
from toaripi_slm.core.model import ToaripiModelManager

# Initialize model manager
manager = ToaripiModelManager()

# List available models
models = manager.list_models()

# Load model for inference
model = manager.load_model("toaripi-primary-v1.0")

# Export model for edge deployment
manager.export_gguf(
    model_path="models/hf/toaripi-primary",
    output_path="models/gguf/toaripi-primary-q4.gguf",
    quantization="q4_k_m"
)
```

#### Methods

**`list_models() -> List[ModelInfo]`**
- List available Toaripi models
- Returns: List of `ModelInfo` objects with model metadata
- Includes: Model names, versions, target age groups, file sizes

**`load_model(model_name: str) -> ToaripiModel`**
- Load model by name or path
- Parameters:
  - `model_name`: Model identifier or file path
- Returns: `ToaripiModel` ready for inference
- Supports: Automatic format detection, device optimization

**`export_gguf(model_path: str, output_path: str, quantization: str) -> ExportResult`**
- Export model to GGUF format for edge deployment
- Parameters:
  - `model_path`: Source model path
  - `output_path`: Target GGUF file path
  - `quantization`: Quantization level (e.g., "q4_k_m", "q8_0")
- Returns: `ExportResult` with export details and file information
- Features: Raspberry Pi optimization, size reduction, performance preservation

### Configuration API

#### Configuration Models

```python
from toaripi_slm.models.config import (
    TrainingConfig,
    PreprocessingConfig,
    InferenceConfig,
    ValidationConfig
)
from toaripi_slm.models.common import AgeGroup, ContentType, ValidationLevel

# Training configuration
training_config = TrainingConfig(
    model_name="microsoft/DialoGPT-small",
    epochs=3,
    batch_size=4,
    learning_rate=2e-5,
    use_lora=True,
    lora_rank=16,
    target_age_groups=[AgeGroup.PRIMARY_LOWER, AgeGroup.PRIMARY_UPPER],
    educational_validation=True,
    cultural_validation=True
)

# Inference configuration
inference_config = InferenceConfig(
    model_path="models/toaripi-primary.gguf",
    device="cpu",
    age_filter=AgeGroup.PRIMARY_LOWER,
    validation_level=ValidationLevel.STRICT,
    max_generation_length=512,
    temperature=0.7,
    top_p=0.9
)
```

## Data Models

### Common Types

```python
from toaripi_slm.models.common import (
    AgeGroup,
    ContentType,
    ValidationLevel,
    DeviceType
)

# Age groups for educational content
class AgeGroup(str, Enum):
    EARLY_CHILDHOOD = "early_childhood"    # 3-5 years
    PRIMARY_LOWER = "primary_lower"        # 6-8 years
    PRIMARY_UPPER = "primary_upper"        # 9-11 years
    SECONDARY = "secondary"                # 12+ years

# Content types for generation
class ContentType(str, Enum):
    STORY = "story"
    VOCABULARY = "vocabulary"
    DIALOGUE = "dialogue"
    COMPREHENSION = "comprehension"
    EXERCISE = "exercise"

# Validation levels
class ValidationLevel(str, Enum):
    BASIC = "basic"            # Basic format validation
    EDUCATIONAL = "educational" # Educational appropriateness
    STRICT = "strict"          # Full cultural and educational validation
```

### Response Models

```python
from toaripi_slm.models.response import (
    GeneratedStory,
    VocabularyList,
    GeneratedDialogue,
    ValidationResult
)

# Generated story response
@dataclass
class GeneratedStory:
    text: str                           # Generated story text
    english_reference: Optional[str]    # English reference text
    age_group: AgeGroup                 # Target age group
    word_count: int                     # Story length
    vocabulary_level: float             # Vocabulary complexity score
    cultural_appropriateness: float     # Cultural validation score
    educational_objectives: List[str]   # Learning objectives met
    generation_metadata: Dict[str, Any] # Generation parameters used

# Vocabulary list response
@dataclass
class VocabularyList:
    topic: str                          # Vocabulary theme
    age_group: AgeGroup                 # Target age group
    words: List[VocabularyWord]         # Generated vocabulary items
    total_count: int                    # Number of words generated
    difficulty_level: float             # Overall difficulty score

@dataclass
class VocabularyWord:
    toaripi: str                        # Toaripi word
    english: str                        # English translation
    definition: str                     # Simple definition
    example_sentence: str               # Usage example
    cultural_context: Optional[str]     # Cultural significance
    difficulty_score: float             # Word difficulty rating
```

## Educational Validation API

### EducationalValidator

```python
from toaripi_slm.core.validation import EducationalValidator
from toaripi_slm.models.validation import ValidationConfig

# Initialize validator
validator = EducationalValidator(
    age_group=AgeGroup.PRIMARY_LOWER,
    validation_level=ValidationLevel.STRICT,
    cultural_sensitivity=True
)

# Validate generated content
result = validator.validate_content(
    content="The children helped their grandparents with fishing.",
    content_type=ContentType.STORY,
    age_group=AgeGroup.PRIMARY_LOWER
)
```

#### Methods

**`validate_content(content: str, content_type: ContentType, age_group: AgeGroup) -> ValidationResult`**
- Comprehensive content validation
- Parameters:
  - `content`: Text content to validate
  - `content_type`: Type of educational content
  - `age_group`: Target age group
- Returns: `ValidationResult` with detailed validation scores
- Checks: Age appropriateness, cultural sensitivity, educational value, safety

**`validate_vocabulary_level(text: str, age_group: AgeGroup) -> VocabularyValidation`**
- Validate vocabulary complexity for age group
- Parameters:
  - `text`: Text to analyze
  - `age_group`: Target learners
- Returns: `VocabularyValidation` with complexity analysis
- Features: Word frequency analysis, complexity scoring, recommendations

**`validate_cultural_appropriateness(content: str) -> CulturalValidation`**
- Check cultural sensitivity and appropriateness
- Parameters:
  - `content`: Content to validate
- Returns: `CulturalValidation` with cultural assessment
- Features: Cultural context validation, sensitivity scoring, improvement suggestions

## Web API (Future)

### FastAPI Endpoints

```python
from fastapi import FastAPI, HTTPException
from toaripi_slm.api.models import GenerationRequest, GenerationResponse

app = FastAPI(title="Toaripi Educational Content API")

@app.post("/generate/story")
async def generate_story(request: GenerationRequest) -> GenerationResponse:
    """Generate educational story in Toaripi."""
    # Implementation here
    pass

@app.post("/generate/vocabulary")
async def generate_vocabulary(request: VocabularyRequest) -> VocabularyResponse:
    """Generate vocabulary exercises."""
    # Implementation here
    pass

@app.get("/models")
async def list_models() -> List[ModelInfo]:
    """List available Toaripi models."""
    # Implementation here
    pass
```

## Error Handling

### Exception Types

```python
from toaripi_slm.exceptions import (
    ToaripiError,
    DataLoadError,
    ValidationError,
    ModelError,
    GenerationError
)

try:
    # Generate content
    story = generator.generate_story(prompt, age_group)
except ValidationError as e:
    print(f"Content validation failed: {e.message}")
    print(f"Validation details: {e.validation_result}")
except GenerationError as e:
    print(f"Generation failed: {e.message}")
    print(f"Error type: {e.error_type}")
except ModelError as e:
    print(f"Model error: {e.message}")
    print(f"Model path: {e.model_path}")
```

## Usage Examples

### Basic Story Generation

```python
from toaripi_slm.inference.generator import ToaripiGenerator
from toaripi_slm.models.common import AgeGroup

# Initialize generator
generator = ToaripiGenerator("models/toaripi-primary.gguf")

# Generate simple story
story = generator.generate_story(
    prompt="A child learns about traditional fishing methods",
    age_group=AgeGroup.PRIMARY_LOWER,
    max_length=150,
    temperature=0.7
)

print(f"Generated Story: {story.text}")
print(f"Educational Objectives: {story.educational_objectives}")
print(f"Cultural Score: {story.cultural_appropriateness}")
```

### Vocabulary Exercise Creation

```python
# Generate vocabulary for a lesson
vocabulary = generator.generate_vocabulary(
    topic="ocean and fishing",
    count=8,
    age_group=AgeGroup.PRIMARY_LOWER
)

print(f"Vocabulary Topic: {vocabulary.topic}")
for word in vocabulary.words:
    print(f"Toaripi: {word.toaripi}")
    print(f"English: {word.english}")
    print(f"Example: {word.example_sentence}")
    print(f"Cultural Context: {word.cultural_context}")
    print("---")
```

### Dialogue Generation

```python
# Generate classroom dialogue
dialogue = generator.generate_dialogue(
    scenario="Teacher and student discussing traditional fishing",
    age_group=AgeGroup.PRIMARY_UPPER,
    participants=2
)

print(f"Dialogue Scenario: {dialogue.scenario}")
for turn in dialogue.conversation:
    print(f"{turn.speaker}: {turn.text}")
print(f"Educational Notes: {dialogue.educational_notes}")
```

### Model Training Pipeline

```python
from toaripi_slm.data.preprocessing import ToaripiDataProcessor
from toaripi_slm.core.trainer import ToaripiTrainer
from toaripi_slm.models.config import TrainingConfig, PreprocessingConfig

# 1. Prepare data
data_config = PreprocessingConfig(
    source_file="data/raw/toaripi_parallel.csv",
    target_age_groups=[AgeGroup.PRIMARY_LOWER],
    validation_level="educational"
)

processor = ToaripiDataProcessor(data_config)
dataset = processor.prepare_training_data()

# 2. Configure training
training_config = TrainingConfig(
    model_name="microsoft/DialoGPT-small",
    epochs=3,
    batch_size=4,
    learning_rate=2e-5,
    educational_validation=True
)

# 3. Train model
trainer = ToaripiTrainer(training_config)
result = trainer.train(dataset)

print(f"Training completed: {result.final_model_path}")
print(f"Educational metrics: {result.educational_scores}")
```

### Content Validation Workflow

```python
from toaripi_slm.core.validation import EducationalValidator

# Initialize validator with strict settings
validator = EducationalValidator(
    validation_level=ValidationLevel.STRICT,
    cultural_sensitivity=True
)

# Validate generated content
content = "Children learn traditional fishing from their elders in the village."

validation = validator.validate_content(
    content=content,
    content_type=ContentType.STORY,
    age_group=AgeGroup.PRIMARY_LOWER
)

if validation.is_valid:
    print("Content approved for educational use")
    print(f"Age appropriateness: {validation.age_appropriateness_score}")
    print(f"Cultural sensitivity: {validation.cultural_sensitivity_score}")
else:
    print("Content needs revision")
    print(f"Issues: {validation.issues}")
    print(f"Suggestions: {validation.improvement_suggestions}")
```

This API reference provides comprehensive documentation for integrating Toaripi SLM functionality into educational applications while maintaining focus on age-appropriate, culturally sensitive content generation.