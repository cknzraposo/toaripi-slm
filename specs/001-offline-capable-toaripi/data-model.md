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

