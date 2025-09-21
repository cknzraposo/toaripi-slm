# Training API Contract

**Feature**: Web Interface Model Training Management  
**Contract**: `/api/training` endpoints  
**Date**: 2025-09-20

## Start Training Session

### `POST /api/training/start`

**Purpose**: Start a new model training session from validated CSV uploads

**Authentication**: None (public endpoint)

**Request**:
```json
{
  "upload_ids": ["upload-uuid-1", "upload-uuid-2"],
  "training_config": {
    "learning_rate": 2e-4,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "batch_size": 4,
    "max_epochs": 3,
    "max_seq_length": 512
  },
  "model_base": "mistralai/Mistral-7B-Instruct-v0.2",
  "session_name": "Toaripi Educational Content v2.1"
}
```

**Success Response** (201):
```json
{
  "success": true,
  "session_id": "session-uuid-abc123",
  "message": "Training session started successfully",
  "estimated_duration_minutes": 45,
  "total_training_pairs": 2847,
  "training_config": {
    "learning_rate": 0.0002,
    "lora_rank": 16,
    "lora_alpha": 32,
    "batch_size": 4,
    "max_epochs": 3,
    "estimated_steps": 2135
  },
  "progress_stream_url": "/api/training/session-uuid-abc123/progress"
}
```

**Error Response** (400):
```json
{
  "success": false,
  "message": "Invalid training configuration",
  "validation_errors": [
    {
      "field": "upload_ids",
      "error_code": "INSUFFICIENT_DATA",
      "message": "Minimum 150 training pairs required, found 89"
    },
    {
      "field": "batch_size", 
      "error_code": "VALUE_TOO_HIGH",
      "message": "Batch size exceeds memory limits for available hardware"
    }
  ]
}
```

**Error Response** (409):
```json
{
  "success": false,
  "message": "Training session already in progress",
  "active_session_id": "session-uuid-xyz789"
}
```

**Validation Rules**:
1. **Data Requirements**: Minimum 150 validated training pairs
2. **Resource Limits**: Training config must fit hardware constraints  
3. **Single Session**: Only one training session allowed at a time
4. **Upload Status**: All upload_ids must have status "VALID"
5. **Model Constraints**: Base model must be ≤7B parameters
6. **Constitutional**: Config must comply with educational safety rules

## Get Training Status

### `GET /api/training/{session_id}/status`

**Purpose**: Get current status of training session

**Authentication**: None (public endpoint)

**Request**:
```http
GET /api/training/session-uuid-abc123/status
```

**Success Response** (200):
```json
{
  "session_id": "session-uuid-abc123",
  "status": "TRAINING",
  "progress_percentage": 67.5,
  "current_step": 1441,
  "total_steps": 2135,
  "current_epoch": 2,
  "total_epochs": 3,
  "start_timestamp": "2025-09-20T14:30:00Z",
  "estimated_completion": "2025-09-20T15:15:00Z",
  "current_loss": 0.245,
  "best_loss": 0.198,
  "learning_rate": 0.0002,
  "memory_usage_gb": 6.8,
  "gpu_utilization": 85.2
}
```

**In Progress Response** (200):
```json
{
  "session_id": "session-uuid-abc123", 
  "status": "COMPLETED",
  "progress_percentage": 100.0,
  "total_steps": 2135,
  "end_timestamp": "2025-09-20T15:12:00Z",
  "final_loss": 0.187,
  "model_output_path": "/models/hf/toaripi-educational-v2.1",
  "evaluation_metrics": {
    "perplexity": 3.42,
    "bleu_score": 0.78,
    "safety_validation": "PASSED"
  },
  "gguf_export_status": "COMPLETED",
  "gguf_model_path": "/models/gguf/toaripi-educational-v2.1-q4.gguf"
}
```

**Error Response** (404):
```json
{
  "success": false,
  "message": "Training session not found",
  "session_id": "session-uuid-abc123"
}
```

## Training Progress Stream

### `GET /api/training/{session_id}/progress`

**Purpose**: Real-time training progress via Server-Sent Events

**Authentication**: None (public endpoint)

**Request**:
```http
GET /api/training/session-uuid-abc123/progress
Accept: text/event-stream
```

**Server-Sent Events Stream**:
```
data: {"session_id": "session-uuid-abc123", "status": "STARTING", "message": "Initializing training environment"}

data: {"session_id": "session-uuid-abc123", "status": "TRAINING", "progress_percentage": 2.1, "current_step": 45, "total_steps": 2135, "current_loss": 2.456, "estimated_time_remaining": 2640}

data: {"session_id": "session-uuid-abc123", "status": "TRAINING", "progress_percentage": 15.7, "current_step": 335, "total_steps": 2135, "current_loss": 1.234, "estimated_time_remaining": 2140}

data: {"session_id": "session-uuid-abc123", "status": "COMPLETED", "progress_percentage": 100.0, "message": "Training completed successfully", "final_loss": 0.187}
```

**Event Data Format**:
```json
{
  "session_id": "session-uuid-abc123",
  "status": "TRAINING",
  "progress_percentage": 45.2,
  "current_step": 965,
  "total_steps": 2135,
  "current_epoch": 2,
  "current_loss": 0.298,
  "learning_rate": 0.0002,
  "estimated_time_remaining": 1320,
  "memory_usage_gb": 6.8,
  "message": "Training epoch 2 of 3"
}
```

## Cancel Training Session

### `POST /api/training/{session_id}/cancel`

**Purpose**: Cancel an active training session

**Authentication**: None (public endpoint)

**Request**:
```http
POST /api/training/session-uuid-abc123/cancel
```

**Success Response** (200):
```json
{
  "success": true,
  "session_id": "session-uuid-abc123",
  "message": "Training session cancelled successfully",
  "cancelled_at_step": 1205,
  "cancelled_at_progress": 56.5,
  "partial_model_saved": true,
  "partial_model_path": "/models/hf/toaripi-educational-v2.1-partial"
}
```

**Error Response** (400):
```json
{
  "success": false,
  "message": "Cannot cancel training session",
  "reason": "Training already completed"
}
```

## Get Training History

### `GET /api/training/history`

**Purpose**: Get list of all training sessions

**Authentication**: None (public endpoint)

**Request Parameters**:
- `limit: int` - Number of sessions to return (default: 10, max: 100)
- `offset: int` - Pagination offset (default: 0)
- `status: str` - Filter by status (optional)

**Request**:
```http
GET /api/training/history?limit=5&status=COMPLETED
```

**Success Response** (200):
```json
{
  "total_sessions": 47,
  "sessions": [
    {
      "session_id": "session-uuid-abc123",
      "session_name": "Toaripi Educational Content v2.1",
      "status": "COMPLETED",
      "start_timestamp": "2025-09-20T14:30:00Z",
      "end_timestamp": "2025-09-20T15:12:00Z",
      "duration_minutes": 42,
      "total_training_pairs": 2847,
      "final_loss": 0.187,
      "model_output_path": "/models/hf/toaripi-educational-v2.1"
    },
    {
      "session_id": "session-uuid-def456",
      "session_name": "Toaripi Stories v1.8",
      "status": "COMPLETED",
      "start_timestamp": "2025-09-19T09:15:00Z", 
      "end_timestamp": "2025-09-19T10:03:00Z",
      "duration_minutes": 48,
      "total_training_pairs": 1523,
      "final_loss": 0.201,
      "model_output_path": "/models/hf/toaripi-stories-v1.8"
    }
  ],
  "pagination": {
    "limit": 5,
    "offset": 0,
    "has_next": true
  }
}
```

## Download Trained Model

### `GET /api/training/{session_id}/download`

**Purpose**: Download trained model files

**Authentication**: None (public endpoint)

**Request Parameters**:
- `format: str` - Model format ("hf" or "gguf", default: "gguf")

**Request**:
```http
GET /api/training/session-uuid-abc123/download?format=gguf
```

**Success Response** (200):
```http
Content-Type: application/octet-stream
Content-Disposition: attachment; filename="toaripi-educational-v2.1-q4.gguf"
Content-Length: 4127856640

[Binary model file data]
```

**Error Response** (404):
```json
{
  "success": false,
  "message": "Model file not found",
  "session_id": "session-uuid-abc123",
  "format": "gguf"
}
```

## Training Configuration Validation

### `POST /api/training/validate-config`

**Purpose**: Validate training configuration before starting session

**Authentication**: None (public endpoint)

**Request**:
```json
{
  "upload_ids": ["upload-uuid-1", "upload-uuid-2"],
  "training_config": {
    "learning_rate": 2e-4,
    "lora_rank": 16,
    "batch_size": 8,
    "max_epochs": 5
  },
  "model_base": "mistralai/Mistral-7B-Instruct-v0.2"
}
```

**Success Response** (200):
```json
{
  "valid": true,
  "estimated_duration_minutes": 67,
  "estimated_memory_usage_gb": 7.2,
  "total_training_pairs": 2847,
  "estimated_steps": 1784,
  "hardware_compatibility": "COMPATIBLE",
  "constitutional_compliance": "PASSED"
}
```

**Error Response** (400):
```json
{
  "valid": false,
  "validation_errors": [
    {
      "field": "batch_size",
      "error_code": "MEMORY_LIMIT_EXCEEDED", 
      "message": "Batch size too large for available memory",
      "suggested_value": 4
    }
  ],
  "estimated_memory_usage_gb": 11.4,
  "memory_limit_gb": 8.0
}
```

## Training Status Enums

### TrainingStatus
- `QUEUED` - Training session created, waiting to start
- `STARTING` - Initializing training environment and loading model
- `TRAINING` - Active training in progress  
- `EVALUATING` - Running final evaluation metrics
- `EXPORTING` - Converting model to GGUF format
- `COMPLETED` - Training finished successfully
- `FAILED` - Training failed due to error
- `CANCELLED` - Training cancelled by user

## Rate Limiting

- **Start Training**: 1 session per hour per IP address
- **Progress Stream**: 1 concurrent connection per session
- **Status Checks**: 60 requests per minute per session
- **Download**: 3 downloads per hour per session

## Implementation Notes

1. **Resource Management**: Only one training session allowed simultaneously
2. **Progress Updates**: Real-time updates every 5 seconds during training
3. **Model Export**: Automatic GGUF conversion for edge deployment
4. **Cleanup**: Failed sessions cleaned after 24 hours
5. **Persistence**: Training state persisted for recovery after restarts
6. **Safety**: All outputs validated against constitutional requirements