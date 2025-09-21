# CSV Upload API Contract

**Feature**: Web Interface CSV Upload and Validation  
**Contract**: `/api/upload` endpoints  
**Date**: 2025-09-20

## Upload CSV Data

### `POST /api/upload/csv`

**Purpose**: Upload CSV file containing parallel English↔Toaripi training data

**Authentication**: None (public endpoint)

**Request**:
```http
POST /api/upload/csv
Content-Type: multipart/form-data

--boundary
Content-Disposition: form-data; name="file"; filename="training_data.csv"
Content-Type: text/csv

english,toaripi
"The children are playing","Teme havere umupama"
"Water is flowing","Vai umuava"
...
--boundary--
```

**Request Parameters**:
- `file: File` - CSV file upload (required)
  - Maximum size: 50MB
  - Required columns: `english`, `toaripi`
  - Optional columns: `category`, `difficulty_level`

**Success Response** (200):
```json
{
  "success": true,
  "upload_id": "uuid-string",
  "message": "File uploaded and validated successfully",
  "total_pairs": 1247,
  "valid_pairs": 1195,
  "preview_data": [
    {
      "row_number": 1,
      "english": "The children are playing",
      "toaripi": "Teme havere umupama",
      "safety_score": 0.95
    },
    {
      "row_number": 2,
      "english": "Water is flowing",
      "toaripi": "Vai umuava",
      "safety_score": 0.98
    }
  ],
  "validation_summary": {
    "duplicates_removed": 12,
    "safety_warnings": 15,
    "length_violations": 25
  }
}
```

**Error Response** (400):
```json
{
  "success": false,
  "message": "CSV validation failed",
  "validation_errors": [
    {
      "field": "english",
      "row_number": 45,
      "error_code": "TEXT_TOO_LONG",
      "message": "English text exceeds 300 character limit",
      "suggested_fix": "Shorten text or split into multiple entries"
    },
    {
      "field": "toaripi",
      "row_number": 67,
      "error_code": "INVALID_CHARACTERS",
      "message": "Contains non-Toaripi characters",
      "suggested_fix": "Use only approved Toaripi orthography"
    }
  ]
}
```

**Error Response** (413):
```json
{
  "success": false,
  "message": "File too large",
  "max_size_mb": 50
}
```

**Validation Rules**:
1. **File Format**: Must be valid CSV with header row
2. **Required Columns**: `english` and `toaripi` columns must exist
3. **Text Length**: Each text field 5-300 characters
4. **Character Validation**: Toaripi text must use approved orthography
5. **Content Safety**: Safety score ≥ 0.7 for inclusion in training
6. **Minimum Data**: At least 150 valid pairs required for training
7. **Duplicates**: Exact duplicates automatically removed

## Get Upload Status

### `GET /api/upload/{upload_id}/status`

**Purpose**: Check processing status of uploaded CSV file

**Authentication**: None (public endpoint)

**Request**:
```http
GET /api/upload/abc-123-def/status
```

**Success Response** (200):
```json
{
  "upload_id": "abc-123-def",
  "status": "VALID",
  "filename": "training_data.csv",
  "upload_timestamp": "2025-09-20T10:30:00Z",
  "total_pairs": 1247,
  "valid_pairs": 1195,
  "processing_progress": 100.0,
  "ready_for_training": true
}
```

**Error Response** (404):
```json
{
  "success": false,
  "message": "Upload not found",
  "upload_id": "abc-123-def"
}
```

## Get Validation Details

### `GET /api/upload/{upload_id}/validation`

**Purpose**: Get detailed validation results for uploaded CSV

**Authentication**: None (public endpoint)

**Request**:
```http
GET /api/upload/abc-123-def/validation
```

**Success Response** (200):
```json
{
  "upload_id": "abc-123-def",
  "validation_summary": {
    "total_rows": 1247,
    "valid_rows": 1195,
    "duplicate_rows": 12,
    "invalid_rows": 40,
    "safety_warnings": 15
  },
  "validation_errors": [
    {
      "row_number": 45,
      "field": "english",
      "error_code": "TEXT_TOO_LONG",
      "message": "English text exceeds 300 character limit"
    }
  ],
  "safety_issues": [
    {
      "row_number": 67,
      "concern": "POTENTIAL_VIOLENCE",
      "score": 0.65,
      "action": "EXCLUDED"
    }
  ],
  "character_issues": [
    {
      "row_number": 89,
      "field": "toaripi",
      "invalid_chars": ["ñ", "ç"],
      "suggestion": "Use standard Toaripi orthography"
    }
  ]
}
```

## Upload Data Preview

### `GET /api/upload/{upload_id}/preview`

**Purpose**: Get sample of validated training data for user review

**Authentication**: None (public endpoint)

**Request Parameters**:
- `limit: int` - Number of samples to return (default: 10, max: 50)
- `offset: int` - Pagination offset (default: 0)

**Request**:
```http
GET /api/upload/abc-123-def/preview?limit=5&offset=0
```

**Success Response** (200):
```json
{
  "upload_id": "abc-123-def",
  "total_valid_pairs": 1195,
  "preview_data": [
    {
      "pair_id": "pair-uuid-1",
      "row_number": 1,
      "english": "The children are playing",
      "toaripi": "Teme havere umupama",
      "safety_score": 0.95,
      "category": "daily_life"
    },
    {
      "pair_id": "pair-uuid-2", 
      "row_number": 2,
      "english": "Water is flowing",
      "toaripi": "Vai umuava",
      "safety_score": 0.98,
      "category": "nature"
    }
  ],
  "pagination": {
    "limit": 5,
    "offset": 0,
    "has_next": true
  }
}
```

## Error Codes

| Code | Description | User Action |
|------|-------------|-------------|
| `FILE_TOO_LARGE` | File exceeds 50MB limit | Split file or compress |
| `INVALID_CSV` | Not a valid CSV format | Fix CSV formatting |
| `MISSING_COLUMNS` | Required columns missing | Add english/toaripi columns |
| `TEXT_TOO_LONG` | Text exceeds 300 chars | Shorten text entries |
| `TEXT_TOO_SHORT` | Text under 5 chars | Provide more complete text |
| `INVALID_CHARACTERS` | Non-Toaripi characters | Use standard orthography |
| `UNSAFE_CONTENT` | Safety score too low | Review content appropriateness |
| `INSUFFICIENT_DATA` | Less than 150 valid pairs | Provide more training data |
| `DUPLICATE_UPLOAD` | Same file uploaded recently | Use existing upload |

## Rate Limiting

- **Upload Limit**: 5 files per hour per IP address
- **File Size**: Maximum 50MB per upload
- **Processing**: Maximum 3 concurrent validations per IP
- **Preview**: 100 requests per minute per upload_id

## Implementation Notes

1. **Async Processing**: CSV validation runs asynchronously to prevent timeout
2. **Progress Updates**: Status endpoint polls validation progress
3. **Memory Management**: Large files processed in streaming chunks
4. **Cleanup**: Failed uploads automatically cleaned after 24 hours
5. **Caching**: Validation results cached for 7 days
6. **Security**: All uploads scanned for malicious content