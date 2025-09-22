# Data Model: CLI Management Tool

## Overview

Data model for the interactive CLI management tool supporting offline Toaripi SLM operations with educational-first design.

## Core Entities

### Training Session

**Purpose**: Tracks active and completed model training runs

**Fields**:
- `session_id: str` - Unique identifier (timestamp-based)
- `name: str` - User-friendly session name
- `status: SessionStatus` - Current state (preparing, running, paused, completed, failed)
- `created_at: datetime` - Session creation timestamp
- `started_at: datetime | None` - Training start time
- `completed_at: datetime | None` - Training completion time
- `model_config: ModelConfig` - Training configuration used
- `dataset_info: DatasetInfo` - Training data information
- `progress: TrainingProgress` - Current training metrics
- `checkpoint_path: Path | None` - Latest checkpoint location
- `log_file: Path` - Training log file path

**Validation Rules**:
- `session_id` must be unique across all sessions
- `name` must be non-empty and contain only alphanumeric characters and spaces
- `status` transitions must follow valid state machine
- `created_at` cannot be in the future
- All file paths must be within user's Toaripi data directory

**State Transitions**:
```
preparing → running → completed
preparing → running → paused → running → completed
preparing → running → failed
preparing → failed
```

### Dataset

**Purpose**: Represents prepared training data with validation status

**Fields**:
- `dataset_id: str` - Unique identifier
- `name: str` - User-friendly dataset name
- `source_files: List[Path]` - Original CSV files
- `processed_path: Path` - Location of processed training data
- `validation_status: ValidationStatus` - Data quality check results
- `statistics: DatasetStats` - Word counts, language balance, etc.
- `created_at: datetime` - Dataset preparation timestamp
- `cultural_review_passed: bool` - Cultural appropriateness validation
- `age_group_suitability: AgeGroup` - Target student age group

**Validation Rules**:
- `source_files` must exist and be readable CSV files
- `processed_path` must be within designated data directory
- `statistics` must be populated after successful processing
- `cultural_review_passed` must be True before training use
- `age_group_suitability` must be appropriate for content

**Related Validation**:
- English-Toaripi parallel text validation
- Cultural content appropriateness checks
- Educational vocabulary level validation

### Model Configuration

**Purpose**: Defines training parameters with educational focus

**Fields**:
- `config_id: str` - Unique configuration identifier
- `name: str` - User-friendly configuration name
- `base_model: str` - Hugging Face model identifier
- `training_params: TrainingParameters` - LoRA and training settings
- `educational_settings: EducationalSettings` - Content generation constraints
- `performance_targets: PerformanceTargets` - Edge deployment requirements
- `created_at: datetime` - Configuration creation time
- `is_default: bool` - Whether this is the default configuration

**Training Parameters**:
- `learning_rate: float` - Learning rate (default: 2e-5)
- `batch_size: int` - Training batch size (default: 4)
- `epochs: int` - Number of training epochs (default: 3)
- `lora_r: int` - LoRA rank parameter (default: 16)
- `lora_alpha: int` - LoRA alpha parameter (default: 32)
- `lora_dropout: float` - LoRA dropout rate (default: 0.1)

**Educational Settings**:
- `content_types: List[ContentType]` - Allowed content types (story, vocabulary, qa)
- `age_groups: List[AgeGroup]` - Target age groups (primary)
- `cultural_guidelines: str` - Specific cultural requirements
- `max_generation_length: int` - Maximum output length (default: 200)

**Validation Rules**:
- All numeric parameters must be within valid ranges
- `base_model` must be available in model registry or Hugging Face
- Educational settings must align with constitutional requirements
- Performance targets must be achievable on edge devices

### Model Checkpoint

**Purpose**: Represents saved model state during training

**Fields**:
- `checkpoint_id: str` - Unique checkpoint identifier
- `session_id: str` - Parent training session
- `epoch: int` - Training epoch when saved
- `step: int` - Training step when saved
- `loss: float` - Training loss at checkpoint
- `checkpoint_path: Path` - Model weights file location
- `metadata_path: Path` - Checkpoint metadata file
- `created_at: datetime` - Checkpoint creation time
- `is_final: bool` - Whether this is the final trained model
- `model_size_mb: float` - File size in megabytes

**Validation Rules**:
- `session_id` must reference existing training session
- `checkpoint_path` must exist and be valid model file
- `model_size_mb` must be within edge deployment limits (<5GB)
- `epoch` and `step` must be non-negative integers
- `loss` must be positive float value

### Training Log

**Purpose**: Detailed record of training events and system information

**Fields**:
- `log_id: str` - Unique log identifier
- `session_id: str` - Associated training session
- `log_level: LogLevel` - DEBUG, INFO, WARNING, ERROR
- `timestamp: datetime` - Event timestamp
- `component: str` - System component that generated log
- `message: str` - Human-readable log message
- `data: Dict[str, Any] | None` - Structured log data
- `performance_metrics: Dict[str, float] | None` - Training metrics

**Validation Rules**:
- `session_id` must reference existing training session
- `timestamp` must be chronologically consistent
- `message` must be non-empty and educational for ERROR level
- Performance metrics must include required fields for training events

## Supporting Types

### Enumerations

```python
class SessionStatus(str, Enum):
    PREPARING = "preparing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class ValidationStatus(str, Enum):
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    NEEDS_REVIEW = "needs_review"

class ContentType(str, Enum):
    STORY = "story"
    VOCABULARY = "vocabulary"
    QA = "qa"
    DIALOGUE = "dialogue"

class AgeGroup(str, Enum):
    PRIMARY = "primary"  # Ages 5-12
    SECONDARY = "secondary"  # Ages 13-18

class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
```

### Composite Types

```python
class DatasetStats(BaseModel):
    total_pairs: int
    english_words: int
    toaripi_words: int
    avg_english_length: float
    avg_toaripi_length: float
    language_balance_ratio: float  # toaripi/english word ratio
    cultural_content_flags: List[str]

class TrainingProgress(BaseModel):
    current_epoch: int
    total_epochs: int
    current_step: int
    total_steps: int
    loss: float
    learning_rate: float
    estimated_completion: datetime | None
    gpu_memory_usage: float | None
    cpu_usage: float

class PerformanceTargets(BaseModel):
    max_memory_gb: float  # Default: 8.0
    max_inference_time_sec: float  # Default: 2.0
    max_model_size_gb: float  # Default: 5.0
    target_platform: str  # Default: "cpu"
```

## Relationships

### Primary Relationships

- `TrainingSession` → `Dataset` (many-to-one)
- `TrainingSession` → `ModelConfiguration` (many-to-one)
- `TrainingSession` → `ModelCheckpoint` (one-to-many)
- `TrainingSession` → `TrainingLog` (one-to-many)

### File System Organization

```
~/.toaripi/
├── sessions/
│   ├── {session_id}/
│   │   ├── metadata.json
│   │   ├── checkpoints/
│   │   └── logs/
├── datasets/
│   ├── {dataset_id}/
│   │   ├── metadata.json
│   │   ├── processed/
│   │   └── validation/
├── configs/
│   └── {config_id}.yaml
└── models/
    ├── registry.json
    └── exports/
```

## Validation Framework

### Data Integrity Validation

- **File System Consistency**: All referenced paths must exist
- **Referential Integrity**: Foreign keys must reference existing entities
- **Constitutional Compliance**: Educational and cultural requirements
- **Performance Constraints**: Memory, size, and timing limits

### Educational Content Validation

- **Age Appropriateness**: Content suitable for target age group
- **Cultural Sensitivity**: No inappropriate cultural references
- **Language Complexity**: Appropriate vocabulary level
- **Learning Objectives**: Clear educational value

### System Resource Validation

- **Disk Space**: Sufficient space for training and checkpoints
- **Memory Limits**: Training fits within system constraints
- **File Permissions**: Read/write access to required directories
- **Dependency Availability**: Required ML libraries present

## Error Handling

### Validation Errors

Educational error messages following constitutional requirements:

```
❌ Data Error: Missing Toaripi translations in row 15
🔍 Cause: Empty 'toaripi' column in training data
✅ Solution: Add Toaripi translation or remove this row
```

### System Errors

Resource constraint errors with recovery guidance:

```
❌ System Error: Insufficient memory for training
🔍 Cause: Training batch size (8) requires 12GB RAM, only 8GB available
✅ Solution: Reduce batch size to 4 in configuration or free system memory
```

### Cultural Validation Errors

Content appropriateness errors with educational context:

```
❌ Content Error: Potentially inappropriate content detected
🔍 Cause: Text contains themes not suitable for primary school students
✅ Solution: Review flagged content in cultural review panel
```

## Summary

This data model supports the constitutional requirements for educational-first development, defensive programming, and edge deployment constraints while providing clear validation and error handling for Toaripi educators.