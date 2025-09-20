# Contract: POST /evaluation-pack

## Purpose

Produce a deterministic 12-item evaluation sample (3 per supported content type) for human review and release gating.

## Request Schema (EvaluationPackRequest)

- mode: enum (stable|standard) default: stable – stable forces deterministic generation for reproducibility.
- include_english_support: bool (default true) – apply to story/qa samples; vocabulary always includes glosses; dialogues remain Toaripi only.

## Response Schema (EvaluationPackResponse)

- pack_id: string (UUID)
- model_version: string
- generated_at: string (ISO8601)
- samples: array (SampleItem objects)
  - id: string (UUID)
  - content_type: enum (story|vocabulary|qa|dialogue)
  - content_body: string|array
  - safety_status: enum (pass|warning|blocked)
  - metadata: object { latency_ms, token_count }
- reviewer_status: enum (pending) – always pending on creation
- counts: object { story: int, vocabulary: int, qa: int, dialogue: int }

## Error Codes

- 429_RATE_LIMIT (if prior pack generated within last 10 minutes)
- 500_INTERNAL_ERROR

## Constraints

- Exactly 12 samples (3 per type) unless a type generation fails hard → endpoint returns 500_INTERNAL_ERROR (no partial success).
