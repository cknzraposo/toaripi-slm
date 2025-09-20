# Contract: GET /health

## Purpose

Lightweight liveness and readiness probe for offline deployment. Verifies model + safety + cache layers loaded and reports basic metrics without heavy computation.

## Request

GET /health (no body)

## Query Parameters (optional)

- verbose: bool (default false). If true include extended metrics section.

## Response Schema (HealthResponse)

- status: enum (ok|degraded)
- model_version: string (semver)
- data_checksum: string (sha256) – current training corpus checksum used by loaded model (if available)
- config_version: string
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
