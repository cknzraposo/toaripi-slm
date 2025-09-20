# Data Model: Offline-Capable Toaripi Educational SLM

## Entities

### ContentRequest
Fields:
- id (UUIDv4)
- topic (string, 2-80 chars)
- seed_text (string, optional, ≤300 chars)
- content_type (enum: story|vocabulary|qa|dialogue)
- target_length (int, optional, bounded by type constraints)
- mode (enum: stable|standard)
- include_english_support (bool)
- requested_at (datetime ISO8601)

Validation:
- topic required unless seed_text present
- content_type required
- length bounds enforced per type

### GeneratedContentArtifact
Fields:
- id (UUIDv4)
- request_id (UUIDv4 FK ContentRequest)
- body (string or JSON array depending on type)
- content_type (enum)
- sentences_or_items_count (int >0)
- latency_ms (int >=0)
- safety_status (enum: pass|blocked|warning)
- model_version (semver)
- data_checksum (string sha256)
- config_version (string semver or hash)
- created_at (datetime)

### ParallelCorpusEntry
Fields:
- id (UUIDv4)
- english_text (string)
- toaripi_text (string)
- source (string)
- checksum (sha256)
- approved (bool default true)
- added_at (datetime)

### SafetyRule
Fields:
- id (UUIDv4)
- category (enum: violence|adult|theological|cultural)
- pattern_or_term (string)
- severity (enum: medium|high)
- active (bool)
- created_at (datetime)

### EvaluationPack
Fields:
- id (UUIDv4)
- generated_at (datetime)
- model_version (semver)
- sample_ids (array<UUIDv4>)
- reviewer_status (enum: pending|approved|changes_requested)

## Relationships
- ContentRequest 1--* GeneratedContentArtifact
- EvaluationPack *--* GeneratedContentArtifact (through sample_ids)
- SafetyRule (no direct FK usage; applied at generation time)

## State & Transitions
GeneratedContentArtifact.safety_status:
- pass → (final)
- blocked → (final, no body distributed)
- warning → (may trigger manual review, can be escalated to blocked)

EvaluationPack.reviewer_status:
- pending → approved | changes_requested
- changes_requested → pending (after revisions)

## Derived Fields & Computations
- sentences_or_items_count: computed by parsing body according to type
- data_checksum: computed from concatenated sorted corpus lines (sha256) at train time

## Validation Rules Summary
| Field | Rule |
|-------|------|
| topic | length 2-80 chars; alphabetic + space/punctuation |
| seed_text | ≤300 chars |
| target_length | within bounds for type |
| content_type=story | target_length 3-8 default 5 |
| content_type=vocabulary | target_length 5-20 default 10 |
| content_type=qa | target_length 2-5 default 3 |
| content_type=dialogue | target_length 4-10 default 7 |
| safety_status | one of pass/blocked/warning |

---

# Web Interface Data Model Extension

**Feature**: Web Interface for CSV Data Upload and Model Training  
**Date**: 2025-09-20

## Additional Entities for Web Interface

### TrainingDataUpload

**Purpose**: Represents a CSV file upload containing parallel English↔Toaripi training data

**Fields**:
- `upload_id: str` - UUID for tracking upload session
- `filename: str` - Original uploaded filename
- `file_size: int` - File size in bytes (max 50MB)
- `mime_type: str` - File MIME type validation
- `upload_timestamp: datetime` - ISO8601 timestamp of upload
- `status: UploadStatus` - Current processing status
- `checksum: str` - SHA256 hash of file content
- `row_count: int | None` - Number of data rows after validation
- `error_details: list[ValidationError] | None` - Validation failures if any

**Validation Rules**:
- `file_size` ≤ 52,428,800 bytes (50MB)
- `mime_type` must be "text/csv" or "application/csv"
- `filename` must end with ".csv"
- `checksum` must be unique within 24-hour window to prevent duplicates

**State Transitions**:
```
PENDING → VALIDATING → VALID | INVALID
VALID → PROCESSING → PROCESSED | FAILED
```

### ParallelTrainingData

**Purpose**: Validated English↔Toaripi text pairs extracted from CSV uploads

**Fields**:
- `pair_id: str` - UUID for individual text pair
- `upload_id: str` - Foreign key to TrainingDataUpload
- `english_text: str` - English source text (5-300 characters)
- `toaripi_text: str` - Toaripi translation (5-300 characters)
- `row_number: int` - Original CSV row for error tracking
- `content_safety_score: float` - Safety validation score (0.0-1.0)
- `safety_flags: list[str]` - Specific safety concerns if any
- `character_validation: bool` - Unicode/charset validation result
- `length_validation: bool` - Text length validation result

**Validation Rules**:
- Both `english_text` and `toaripi_text` required and non-empty
- Text length between 5-300 characters each
- Character set validation for Toaripi orthography
- Content safety score ≥ 0.7 required for training inclusion
- No more than 2 safety flags per pair

### TrainingSession

**Purpose**: Represents a model training run initiated from validated data

**Fields**:
- `session_id: str` - UUID for training session
- `upload_ids: list[str]` - Source data uploads for this training
- `config_version: str` - Training configuration version hash
- `model_base: str` - Base model identifier (e.g., "mistral-7b-instruct")
- `training_params: TrainingConfig` - Embedded training parameters
- `start_timestamp: datetime` - Training start time
- `end_timestamp: datetime | None` - Training completion time
- `status: TrainingStatus` - Current training status
- `progress_percentage: float` - Training progress (0.0-100.0)
- `current_step: int` - Current training step
- `total_steps: int` - Total estimated steps
- `loss_history: list[float]` - Training loss values
- `evaluation_metrics: dict | None` - Final evaluation results
- `output_model_path: str | None` - Path to trained model

**Validation Rules**:
- At least 150 valid training pairs required before session creation
- `training_params` must validate against constitutional constraints
- `model_base` must be ≤7B parameters
- Progress updates required every 30 seconds during training

**State Transitions**:
```
QUEUED → STARTING → TRAINING → COMPLETED | FAILED | CANCELLED
```

### TrainingConfig

**Purpose**: Embedded configuration for training sessions

**Fields**:
- `learning_rate: float` - LoRA learning rate (default: 2e-4)
- `lora_rank: int` - LoRA rank parameter (default: 16)
- `lora_alpha: int` - LoRA alpha parameter (default: 32)
- `lora_dropout: float` - LoRA dropout rate (default: 0.1)
- `batch_size: int` - Training batch size (default: 4)
- `max_epochs: int` - Maximum training epochs (default: 3)
- `early_stopping_patience: int` - Early stopping rounds (default: 2)
- `warmup_steps: int` - Learning rate warmup steps (default: 10)
- `eval_steps: int` - Evaluation frequency (default: 50)
- `save_steps: int` - Model checkpoint frequency (default: 100)
- `max_seq_length: int` - Maximum sequence length (default: 512)
- `target_modules: list[str]` - LoRA target modules (default: ["q_proj", "v_proj"])

**Validation Rules**:
- All parameters must be within constitutional safe ranges
- Combined settings must not exceed 7B parameter constraint
- Memory usage estimation must fit baseline hardware (8GB RAM)

## Web Interface API Response Models

### UploadResponse

**Purpose**: Response format for file upload endpoints

**Fields**:
- `success: bool` - Whether upload succeeded
- `upload_id: str | None` - Upload identifier if successful
- `message: str` - Human-readable status message
- `validation_errors: list[ValidationError] | None` - Specific validation issues
- `preview_data: list[dict] | None` - Sample of validated data for user review
- `total_pairs: int | None` - Count of valid training pairs
- `safety_warnings: list[str] | None` - Content safety concerns

### TrainingProgressUpdate

**Purpose**: Real-time training progress via Server-Sent Events

**Fields**:
- `session_id: str` - Training session identifier
- `progress_percentage: float` - Current progress (0.0-100.0)
- `current_step: int` - Current training step
- `total_steps: int` - Total estimated steps
- `current_loss: float | None` - Latest loss value
- `estimated_time_remaining: int | None` - Seconds until completion
- `status: TrainingStatus` - Current training status
- `message: str` - Human-readable progress message

### ValidationError

**Purpose**: Structured validation error information

**Fields**:
- `field: str` - Field name or CSV column that failed validation
- `row_number: int | None` - CSV row number if applicable
- `error_code: str` - Machine-readable error code
- `message: str` - Human-readable error description
- `suggested_fix: str | None` - Suggested correction for user

## Web Interface Enums

### UploadStatus
- `PENDING` - Upload received, not yet processed
- `VALIDATING` - CSV validation in progress
- `VALID` - Validation passed, ready for training
- `INVALID` - Validation failed, cannot be used for training

### TrainingStatus
- `QUEUED` - Training session created, waiting to start
- `STARTING` - Initializing training environment
- `TRAINING` - Active training in progress
- `COMPLETED` - Training finished successfully
- `FAILED` - Training failed due to error
- `CANCELLED` - Training cancelled by user

