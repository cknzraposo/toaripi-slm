# Health Check API Contract

**Feature**: Web Interface Health and Status Monitoring  
**Contract**: `/api/health` endpoints  
**Date**: 2025-09-20

## System Health Check

### `GET /api/health`

**Purpose**: Check overall system health and availability

**Authentication**: None (public endpoint)

**Request**:
```http
GET /api/health
```

**Success Response** (200):
```json
{
  "status": "healthy",
  "timestamp": "2025-09-20T18:30:00Z",
  "version": "2.1.0",
  "uptime_seconds": 86400,
  "components": {
    "api": "healthy",
    "database": "healthy", 
    "model_engine": "healthy",
    "file_storage": "healthy",
    "training_service": "healthy"
  },
  "system_resources": {
    "memory_usage_percent": 45.2,
    "disk_usage_percent": 67.8,
    "cpu_usage_percent": 23.1,
    "available_memory_gb": 4.4,
    "available_disk_gb": 12.3
  },
  "active_sessions": {
    "training_sessions": 0,
    "generation_requests": 3,
    "upload_validations": 1
  }
}
```

**Degraded Response** (200):
```json
{
  "status": "degraded",
  "timestamp": "2025-09-20T18:30:00Z", 
  "version": "2.1.0",
  "uptime_seconds": 86400,
  "components": {
    "api": "healthy",
    "database": "healthy",
    "model_engine": "degraded",
    "file_storage": "healthy", 
    "training_service": "unhealthy"
  },
  "system_resources": {
    "memory_usage_percent": 89.5,
    "disk_usage_percent": 67.8,
    "cpu_usage_percent": 94.2,
    "available_memory_gb": 0.8,
    "available_disk_gb": 12.3
  },
  "issues": [
    {
      "component": "training_service",
      "severity": "high",
      "message": "Training service unavailable due to resource constraints",
      "since": "2025-09-20T18:15:00Z"
    },
    {
      "component": "model_engine", 
      "severity": "medium",
      "message": "Model inference slower than normal due to high memory usage",
      "since": "2025-09-20T18:20:00Z"
    }
  ],
  "active_sessions": {
    "training_sessions": 0,
    "generation_requests": 8,
    "upload_validations": 2
  }
}
```

**Unhealthy Response** (503):
```json
{
  "status": "unhealthy",
  "timestamp": "2025-09-20T18:30:00Z",
  "version": "2.1.0",
  "uptime_seconds": 86400,
  "components": {
    "api": "healthy",
    "database": "unhealthy",
    "model_engine": "unhealthy", 
    "file_storage": "degraded",
    "training_service": "unhealthy"
  },
  "critical_issues": [
    {
      "component": "database",
      "severity": "critical",
      "message": "Database connection lost",
      "since": "2025-09-20T18:25:00Z"
    }
  ]
}
```

## Detailed Component Health

### `GET /api/health/components`

**Purpose**: Get detailed health information for all system components

**Authentication**: None (public endpoint)

**Request**:
```http
GET /api/health/components
```

**Success Response** (200):
```json
{
  "timestamp": "2025-09-20T18:30:00Z",
  "components": {
    "api": {
      "status": "healthy",
      "response_time_ms": 45,
      "requests_per_minute": 127,
      "error_rate_percent": 0.2,
      "last_check": "2025-09-20T18:30:00Z"
    },
    "database": {
      "status": "healthy",
      "connection_pool_usage": 3,
      "connection_pool_size": 20,
      "query_response_time_ms": 12,
      "last_check": "2025-09-20T18:30:00Z"
    },
    "model_engine": {
      "status": "healthy",
      "active_model": "toaripi-educational-v2.1",
      "model_load_time_ms": 2340,
      "inference_speed_tokens_per_sec": 45,
      "memory_usage_mb": 4200,
      "last_check": "2025-09-20T18:30:00Z"
    },
    "file_storage": {
      "status": "healthy",
      "total_space_gb": 50,
      "used_space_gb": 33.9,
      "available_space_gb": 16.1,
      "read_speed_mbps": 120,
      "write_speed_mbps": 85,
      "last_check": "2025-09-20T18:30:00Z"
    },
    "training_service": {
      "status": "healthy",
      "queue_size": 0,
      "max_concurrent_sessions": 1,
      "active_sessions": 0,
      "gpu_available": false,
      "last_check": "2025-09-20T18:30:00Z"
    }
  }
}
```

## Resource Usage Metrics

### `GET /api/health/metrics`

**Purpose**: Get detailed system resource usage and performance metrics

**Authentication**: None (public endpoint)

**Request Parameters**:
- `timeframe: str` - Time period for metrics ("1h", "6h", "24h", default: "1h")

**Request**:
```http
GET /api/health/metrics?timeframe=6h
```

**Success Response** (200):
```json
{
  "timeframe": "6h",
  "timestamp": "2025-09-20T18:30:00Z",
  "system_metrics": {
    "cpu": {
      "current_usage_percent": 23.1,
      "average_usage_percent": 34.5,
      "peak_usage_percent": 87.2,
      "cores": 8
    },
    "memory": {
      "total_gb": 8.0,
      "used_gb": 3.6,
      "available_gb": 4.4,
      "usage_percent": 45.2,
      "peak_usage_percent": 76.8
    },
    "disk": {
      "total_gb": 50.0,
      "used_gb": 33.9,
      "available_gb": 16.1,
      "usage_percent": 67.8,
      "io_read_mbps": 45.2,
      "io_write_mbps": 23.7
    },
    "network": {
      "bytes_sent": 1547892340,
      "bytes_received": 2341567890,
      "current_bandwidth_mbps": 12.4
    }
  },
  "application_metrics": {
    "requests": {
      "total_requests": 15420,
      "requests_per_minute": 127,
      "average_response_time_ms": 245,
      "error_rate_percent": 0.2
    },
    "uploads": {
      "total_uploads": 47,
      "successful_uploads": 44,
      "failed_uploads": 3,
      "average_file_size_mb": 8.4,
      "total_data_processed_gb": 1.2
    },
    "training": {
      "total_sessions": 12,
      "completed_sessions": 10,
      "failed_sessions": 2,
      "average_duration_minutes": 52,
      "total_training_time_hours": 10.4
    },
    "generation": {
      "total_requests": 234,
      "successful_generations": 228,
      "average_generation_time_ms": 1250,
      "content_types": {
        "story": 89,
        "vocabulary": 67,
        "dialogue": 45,
        "qa": 33
      }
    }
  }
}
```

## Constitutional Compliance Check

### `GET /api/health/constitutional`

**Purpose**: Verify constitutional compliance and safety systems

**Authentication**: None (public endpoint)

**Request**:
```http
GET /api/health/constitutional
```

**Success Response** (200):
```json
{
  "status": "compliant",
  "timestamp": "2025-09-20T18:30:00Z",
  "compliance_checks": {
    "content_safety": {
      "status": "active",
      "model_version": "safety-v1.2",
      "last_updated": "2025-09-20T10:00:00Z",
      "blocked_requests_24h": 7,
      "safety_score_threshold": 0.7
    },
    "age_appropriateness": {
      "status": "active",
      "primary_filter": "enabled",
      "secondary_filter": "enabled",
      "inappropriate_content_blocked_24h": 2
    },
    "cultural_sensitivity": {
      "status": "active", 
      "toaripi_cultural_guidelines": "enforced",
      "cultural_violations_24h": 0
    },
    "educational_alignment": {
      "status": "active",
      "educational_standards": "enforced",
      "non_educational_content_blocked_24h": 1
    },
    "model_constraints": {
      "status": "active",
      "max_model_size": "7B",
      "current_models_compliant": true,
      "memory_limit_enforced": true
    }
  },
  "safety_incidents": [],
  "constitutional_violations": []
}
```

**Non-Compliant Response** (200):
```json
{
  "status": "violation",
  "timestamp": "2025-09-20T18:30:00Z",
  "violations": [
    {
      "rule": "CONTENT_SAFETY",
      "severity": "medium",
      "description": "Safety model not responding properly",
      "since": "2025-09-20T18:15:00Z",
      "impact": "Content generation disabled"
    }
  ],
  "mitigation_actions": [
    "Content generation temporarily disabled",
    "Fallback safety checks activated", 
    "Safety model restart initiated"
  ]
}
```

## Readiness Check

### `GET /api/health/ready`

**Purpose**: Check if system is ready to handle requests (for load balancers)

**Authentication**: None (public endpoint)

**Request**:
```http
GET /api/health/ready
```

**Ready Response** (200):
```json
{
  "ready": true,
  "timestamp": "2025-09-20T18:30:00Z",
  "message": "System ready to handle requests"
}
```

**Not Ready Response** (503):
```json
{
  "ready": false,
  "timestamp": "2025-09-20T18:30:00Z",
  "message": "System not ready",
  "blocking_issues": [
    "Model not loaded",
    "Database migration in progress"
  ]
}
```

## Liveness Check

### `GET /api/health/live`

**Purpose**: Check if system is alive (for container orchestration)

**Authentication**: None (public endpoint)

**Request**:
```http
GET /api/health/live
```

**Alive Response** (200):
```json
{
  "alive": true,
  "timestamp": "2025-09-20T18:30:00Z"
}
```

**Dead Response** (503):
```json
{
  "alive": false,
  "timestamp": "2025-09-20T18:30:00Z"
}
```

## Health Status Enums

### SystemStatus
- `healthy` - All components functioning normally
- `degraded` - Some components experiencing issues but system operational
- `unhealthy` - Critical components failing, system may not function properly

### ComponentStatus  
- `healthy` - Component functioning normally
- `degraded` - Component experiencing issues but still functional
- `unhealthy` - Component not functioning properly

### Severity
- `low` - Minor issue, no impact on functionality
- `medium` - Some impact on performance or features
- `high` - Significant impact, some features may be unavailable
- `critical` - Severe impact, system may be unusable

## Rate Limiting

- **Health Check**: 60 requests per minute per IP
- **Component Health**: 30 requests per minute per IP
- **Metrics**: 10 requests per minute per IP (due to processing overhead)
- **Constitutional Check**: 20 requests per minute per IP

## Implementation Notes

1. **Caching**: Health data cached for 30 seconds to reduce overhead
2. **Background Checks**: Components checked every 60 seconds in background
3. **Alerting**: Critical issues trigger alerts to system administrators
4. **Graceful Degradation**: System continues operating with reduced functionality when possible
5. **Constitutional Enforcement**: Safety violations immediately disable affected features
6. **Resource Monitoring**: Automatic alerts when resource usage exceeds thresholds
- uptime_seconds: integer
- cache_entries: integer (active generation cache keys)
- safety_rules_loaded: integer (active rules count)
- last_generation_at: string|nullable (ISO8601 of most recent successful generation)
- latency_targets: object { story_ms_p95, vocabulary_ms_p95, qa_ms_p95, dialogue_ms_p95 } (only if verbose=true)
- disclaimer: string (same standardized disclaimer text)

## Status Codes

- 200 OK – service healthy (status may still be degraded logically)
- 500 INTERNAL_ERROR – unexpected failure collecting health data

## Error Codes

- 500_INTERNAL_ERROR

## Notes

- status=degraded if any of: safety_rules_loaded == 0, model not loaded, or cache_entries > max_cache_entries threshold.
