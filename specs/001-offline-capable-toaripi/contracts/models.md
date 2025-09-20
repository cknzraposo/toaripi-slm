# Model Management API Contract

**Feature**: Web Interface Model Management and Generation  
**Contract**: `/api/models` endpoints  
**Date**: 2025-09-20

## List Available Models

### `GET /api/models`

**Purpose**: Get list of all trained models available for content generation

**Authentication**: None (public endpoint)

**Request Parameters**:
- `format: str` - Filter by model format ("hf", "gguf", "all", default: "all")
- `status: str` - Filter by status ("active", "archived", "all", default: "active")

**Request**:
```http
GET /api/models?format=gguf&status=active
```

**Success Response** (200):
```json
{
  "total_models": 12,
  "active_model": "toaripi-educational-v2.1",
  "models": [
    {
      "model_id": "toaripi-educational-v2.1",
      "display_name": "Toaripi Educational Content v2.1",
      "training_session_id": "session-uuid-abc123",
      "created_timestamp": "2025-09-20T15:12:00Z",
      "model_type": "educational",
      "format": "gguf",
      "file_size_mb": 3934,
      "status": "active",
      "training_data_pairs": 2847,
      "evaluation_metrics": {
        "perplexity": 3.42,
        "bleu_score": 0.78,
        "safety_score": 0.96
      },
      "capabilities": [
        "story_generation",
        "vocabulary_exercises", 
        "dialogue_creation",
        "qa_generation"
      ],
      "download_url": "/api/models/toaripi-educational-v2.1/download"
    },
    {
      "model_id": "toaripi-stories-v1.8",
      "display_name": "Toaripi Stories v1.8",
      "training_session_id": "session-uuid-def456",
      "created_timestamp": "2025-09-19T10:03:00Z",
      "model_type": "educational",
      "format": "gguf",
      "file_size_mb": 3821,
      "status": "active",
      "training_data_pairs": 1523,
      "evaluation_metrics": {
        "perplexity": 3.67,
        "bleu_score": 0.74,
        "safety_score": 0.94
      },
      "capabilities": [
        "story_generation",
        "dialogue_creation"
      ],
      "download_url": "/api/models/toaripi-stories-v1.8/download"
    }
  ]
}
```

## Get Model Details

### `GET /api/models/{model_id}`

**Purpose**: Get detailed information about a specific model

**Authentication**: None (public endpoint)

**Request**:
```http
GET /api/models/toaripi-educational-v2.1
```

**Success Response** (200):
```json
{
  "model_id": "toaripi-educational-v2.1",
  "display_name": "Toaripi Educational Content v2.1",
  "description": "Fine-tuned model for generating educational content in Toaripi language",
  "training_session_id": "session-uuid-abc123",
  "created_timestamp": "2025-09-20T15:12:00Z",
  "status": "active",
  "model_type": "educational",
  "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
  "formats": [
    {
      "format": "hf",
      "file_path": "/models/hf/toaripi-educational-v2.1",
      "file_size_mb": 13420,
      "download_url": "/api/models/toaripi-educational-v2.1/download?format=hf"
    },
    {
      "format": "gguf",
      "file_path": "/models/gguf/toaripi-educational-v2.1-q4.gguf", 
      "file_size_mb": 3934,
      "quantization": "q4_k_m",
      "download_url": "/api/models/toaripi-educational-v2.1/download?format=gguf"
    }
  ],
  "training_details": {
    "total_training_pairs": 2847,
    "training_duration_minutes": 42,
    "final_training_loss": 0.187,
    "lora_config": {
      "rank": 16,
      "alpha": 32,
      "dropout": 0.1,
      "target_modules": ["q_proj", "v_proj"]
    }
  },
  "evaluation_metrics": {
    "perplexity": 3.42,
    "bleu_score": 0.78,
    "safety_score": 0.96,
    "constitutional_compliance": "PASSED"
  },
  "capabilities": [
    "story_generation",
    "vocabulary_exercises",
    "dialogue_creation", 
    "qa_generation"
  ],
  "performance_specs": {
    "inference_speed_tokens_per_second": 45,
    "memory_usage_mb": 4200,
    "cpu_compatible": true,
    "raspberry_pi_compatible": true
  }
}
```

**Error Response** (404):
```json
{
  "success": false,
  "message": "Model not found",
  "model_id": "toaripi-educational-v2.1"
}
```

## Set Active Model

### `POST /api/models/{model_id}/activate`

**Purpose**: Set a model as the active model for content generation

**Authentication**: None (public endpoint)

**Request**:
```http
POST /api/models/toaripi-educational-v2.1/activate
```

**Success Response** (200):
```json
{
  "success": true,
  "message": "Model activated successfully",
  "model_id": "toaripi-educational-v2.1",
  "previous_active_model": "toaripi-stories-v1.8",
  "activation_timestamp": "2025-09-20T16:45:00Z"
}
```

**Error Response** (400):
```json
{
  "success": false,
  "message": "Cannot activate model",
  "reason": "Model status is not active",
  "model_id": "toaripi-educational-v2.1",
  "current_status": "archived"
}
```

## Download Model

### `GET /api/models/{model_id}/download`

**Purpose**: Download model files for local deployment

**Authentication**: None (public endpoint)

**Request Parameters**:
- `format: str` - Model format ("hf" or "gguf", default: "gguf")

**Request**:
```http
GET /api/models/toaripi-educational-v2.1/download?format=gguf
```

**Success Response** (200):
```http
Content-Type: application/octet-stream
Content-Disposition: attachment; filename="toaripi-educational-v2.1-q4.gguf"
Content-Length: 4127856640

[Binary model file data]
```

**Redirect Response** (302):
```http
Location: https://cdn.example.com/models/toaripi-educational-v2.1-q4.gguf
```

## Archive Model

### `POST /api/models/{model_id}/archive`

**Purpose**: Archive a model (move to inactive status but keep files)

**Authentication**: None (public endpoint)

**Request**:
```http
POST /api/models/toaripi-educational-v2.1/archive
```

**Request Body** (optional):
```json
{
  "reason": "Superseded by newer version",
  "archive_note": "Archived after deploying v2.2 with improved training data"
}
```

**Success Response** (200):
```json
{
  "success": true,
  "message": "Model archived successfully",
  "model_id": "toaripi-educational-v2.1",
  "archived_timestamp": "2025-09-20T17:00:00Z",
  "files_preserved": true
}
```

**Error Response** (400):
```json
{
  "success": false,
  "message": "Cannot archive active model",
  "model_id": "toaripi-educational-v2.1",
  "current_status": "active",
  "suggestion": "Activate a different model first"
}
```

## Delete Model

### `DELETE /api/models/{model_id}`

**Purpose**: Permanently delete a model and all associated files

**Authentication**: None (public endpoint)

**Request**:
```http
DELETE /api/models/toaripi-educational-v1.0
```

**Request Body** (optional):
```json
{
  "confirm_deletion": true,
  "reason": "Outdated model with poor performance"
}
```

**Success Response** (200):
```json
{
  "success": true,
  "message": "Model deleted successfully",
  "model_id": "toaripi-educational-v1.0",
  "deleted_timestamp": "2025-09-20T17:15:00Z",
  "files_deleted": [
    "/models/hf/toaripi-educational-v1.0",
    "/models/gguf/toaripi-educational-v1.0-q4.gguf"
  ],
  "disk_space_freed_mb": 17354
}
```

**Error Response** (400):
```json
{
  "success": false,
  "message": "Cannot delete active model",
  "model_id": "toaripi-educational-v1.0",
  "current_status": "active"
}
```

## Model Performance Metrics

### `GET /api/models/{model_id}/metrics`

**Purpose**: Get detailed performance and evaluation metrics for a model

**Authentication**: None (public endpoint)

**Request**:
```http
GET /api/models/toaripi-educational-v2.1/metrics
```

**Success Response** (200):
```json
{
  "model_id": "toaripi-educational-v2.1",
  "evaluation_timestamp": "2025-09-20T15:12:00Z",
  "training_metrics": {
    "final_loss": 0.187,
    "best_loss": 0.182,
    "training_perplexity": 3.42,
    "validation_perplexity": 3.45,
    "training_steps": 2135,
    "convergence_step": 1847
  },
  "language_quality": {
    "bleu_score": 0.78,
    "rouge_l": 0.82,
    "meteor_score": 0.75,
    "semantic_similarity": 0.81
  },
  "safety_evaluation": {
    "overall_safety_score": 0.96,
    "content_appropriateness": 0.98,
    "cultural_sensitivity": 0.94,
    "educational_alignment": 0.97,
    "constitutional_compliance": "PASSED"
  },
  "generation_quality": {
    "story_coherence": 0.89,
    "vocabulary_accuracy": 0.92,
    "dialogue_naturalness": 0.85,
    "qa_relevance": 0.91
  },
  "performance_benchmarks": {
    "inference_speed_tokens_per_second": 45,
    "memory_usage_peak_mb": 4200,
    "cpu_efficiency_score": 0.87,
    "edge_device_compatibility": "excellent"
  }
}
```

## Model Comparison

### `POST /api/models/compare`

**Purpose**: Compare performance metrics between multiple models

**Authentication**: None (public endpoint)

**Request**:
```json
{
  "model_ids": [
    "toaripi-educational-v2.1",
    "toaripi-stories-v1.8",
    "toaripi-vocabulary-v1.5"
  ],
  "comparison_metrics": [
    "bleu_score",
    "safety_score", 
    "inference_speed",
    "perplexity"
  ]
}
```

**Success Response** (200):
```json
{
  "comparison_timestamp": "2025-09-20T17:30:00Z",
  "models": [
    {
      "model_id": "toaripi-educational-v2.1",
      "metrics": {
        "bleu_score": 0.78,
        "safety_score": 0.96,
        "inference_speed": 45,
        "perplexity": 3.42
      },
      "ranking": {
        "bleu_score": 1,
        "safety_score": 1, 
        "inference_speed": 2,
        "perplexity": 1
      }
    },
    {
      "model_id": "toaripi-stories-v1.8",
      "metrics": {
        "bleu_score": 0.74,
        "safety_score": 0.94,
        "inference_speed": 48,
        "perplexity": 3.67
      },
      "ranking": {
        "bleu_score": 2,
        "safety_score": 2,
        "inference_speed": 1,
        "perplexity": 3
      }
    },
    {
      "model_id": "toaripi-vocabulary-v1.5", 
      "metrics": {
        "bleu_score": 0.71,
        "safety_score": 0.93,
        "inference_speed": 42,
        "perplexity": 3.59
      },
      "ranking": {
        "bleu_score": 3,
        "safety_score": 3,
        "inference_speed": 3,
        "perplexity": 2
      }
    }
  ],
  "overall_rankings": {
    "1st": "toaripi-educational-v2.1",
    "2nd": "toaripi-stories-v1.8", 
    "3rd": "toaripi-vocabulary-v1.5"
  }
}
```

## Model Status Enums

### ModelStatus
- `active` - Available for content generation
- `archived` - Inactive but files preserved
- `training` - Currently being trained
- `failed` - Training failed, incomplete model
- `deleted` - Permanently removed

### ModelFormat
- `hf` - HuggingFace transformers format
- `gguf` - Quantized format for edge deployment

### ModelType
- `educational` - General educational content generation
- `stories` - Specialized for story generation
- `vocabulary` - Specialized for vocabulary exercises
- `dialogue` - Specialized for dialogue creation

## Rate Limiting

- **Model List**: 60 requests per minute per IP
- **Model Details**: 100 requests per minute per IP  
- **Download**: 3 downloads per hour per model per IP
- **Activate**: 10 activations per hour per IP
- **Archive/Delete**: 5 operations per hour per IP

## Implementation Notes

1. **Model Loading**: Active model kept in memory for fast inference
2. **Download CDN**: Large model files served via CDN for performance
3. **Cleanup**: Deleted models permanently removed after 7-day grace period
4. **Versioning**: Model versions tracked via semantic versioning
5. **Backup**: Critical models automatically backed up to cloud storage
6. **Monitoring**: Model performance continuously monitored for degradation