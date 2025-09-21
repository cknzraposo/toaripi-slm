# Contract: GET /metadata/run

## Purpose

Return reproducibility and runtime metadata for auditing a specific or current run.

## Query Parameters

- request_id: string (UUID, optional) – if provided return metadata for that generation; else return aggregate current run environment.

## Response Schema (RunMetadataResponse)

When request_id provided (SingleGenerationMetadata):

- request_id: string
- model_version: string
- data_checksum: string
- config_version: string
- latency_ms: integer
- safety_status: enum (pass|warning|blocked)
- created_at: string (ISO8601)

When request_id omitted (EnvironmentMetadata):

- model_version: string
- data_checksum: string
- config_version: string
- total_generations: integer
- last_request_id: string|nullable
- last_generation_at: string|nullable
- cache_entries: integer

## Error Codes

- 404_NOT_FOUND (request_id not found)
- 500_INTERNAL_ERROR

## Notes

- Endpoint is read-only; no mutation allowed.
