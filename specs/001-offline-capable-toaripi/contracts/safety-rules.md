# Contract: GET /safety/rules

## Purpose

Expose currently active lexical safety rules for transparency and debug; offline only, no remote fetch.

## Request

GET /safety/rules (no body)

## Response Schema (SafetyRulesResponse)

- rules: array (Rule objects)
  - id: string (UUID)
  - category: enum (violence|adult|theological|cultural)
  - pattern_or_term: string
  - severity: enum (medium|high)
  - active: bool
  - created_at: string (ISO8601)
- total: integer
- model_version: string

## Error Codes

- 500_INTERNAL_ERROR

## Notes

- pattern_or_term may be redacted (e.g., partially masked) for sensitive categories if future policy dictates.
