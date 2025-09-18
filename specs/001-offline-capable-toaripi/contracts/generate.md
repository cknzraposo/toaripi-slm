# Contract: POST /generate

## Request Schema (GenerateRequest)
- topic: string (2-80 chars)
- content_type: enum (story|vocabulary|qa|dialogue)
- target_length: int (optional, bounded per type)
- mode: enum (stable|standard) default: standard
- include_english_support: bool (default varies by type)

## Response Schema (GenerateResponse)
- content_type: enum
- content_body: string or array (vocabulary list)
- metadata: object { request_id, latency_ms, model_version }
- safety_status: enum (pass|blocked|warning)
- disclaimer_shown: bool

## Error Codes
- 400_INVALID_INPUT
- 400_LENGTH_EXCEEDED
- 403_RESTRICTED_TOPIC
- 429_TOO_FREQUENT_DUPLICATE
- 500_INTERNAL_ERROR
