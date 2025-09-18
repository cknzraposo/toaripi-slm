# Implementation Plan: Offline-Capable Toaripi Educational Small Language Model (SLM)

**Branch**: `001-offline-capable-toaripi` | **Date**: 2025-09-18 | **Spec**: `spec.md`
**Input**: Feature specification from `/specs/001-offline-capable-toaripi/spec.md`

## Execution Flow (/plan command scope)

```text
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (single Python library + CLI/inference API)
   → Set Structure Decision based on project type (Option 1: Single project) 
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → All unknowns resolved (no blocking NEEDS CLARIFICATION remain)
6. Execute Phase 1 → contracts, data-model.md, quickstart.md
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

## Summary

Deliver a reproducible, offline-capable small Toaripi educational language model (≤7B params, ≤5GB quantized) that generates age-appropriate stories, vocabulary lists, Q&A pairs, and dialogues with safety filtering, deterministic "stable" mode, caching, and evaluation pack workflow to support language preservation in primary education contexts.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: transformers, datasets, accelerate, peft (LoRA), sentencepiece/tokenizers, fastapi (serving), pydantic, yaml  
**Storage**: Local filesystem (data CSV/YAML configs, model artifacts, JSON logs); no external DB required  
**Testing**: pytest (unit, integration, contract tests)  
**Target Platform**: Linux/Windows CPU (8GB RAM) + optional Raspberry Pi 5 (ARM64) offline  
**Project Type**: single  
**Performance Goals**: Generation latency ≤10s (stories/Q&A/dialogue), ≤8s (vocabulary) p95 on baseline CPU; quantized model load ≤60s cold start  
**Constraints**: Offline after install; model ≤7B params; quantized footprint ≤5GB; reproducible training (config + data checksums); safety filtering mandatory  
**Scale/Scope**: Classroom/local usage; concurrent requests low (≤2 at a time); dataset initially hundreds to low thousands of aligned pairs

## Constitution Check

| Principle | Compliance | Notes |
|-----------|------------|-------|
| Minimal Viable Educational Model | YES | Scope limited strictly to four educational content types |
| Reproducible Data & Training Pipeline | YES | YAML configs + data checksums + run metadata planned |
| Test-First Quality Gates | YES | Contract & integration tests precede implementation; mock generation tests included |
| Responsible & Culturally Safe Content | YES | Safety rules entity + lexical + threshold screening defined |
| Simplicity, Observability & Versioning | YES | Single project layout; JSON structured logs; semantic version tagging |
| Additional Constraints (size/offline) | YES | ≤7B params & offline constraints integrated |
| Data Minimum | PARTIAL | Constitution minimum 100 pairs; spec raises to 150 (rationale: coverage) |
| Storage Hygiene | YES | Artifacts isolated under models/; no checkpoints in src/ |
| Documentation Requirement | YES | Quickstart + docstrings planned |

No blocking violations. Rationale for raising minimum pairs: improved lexical diversity and more stable fine-tuning for low-resource language.

## Project Structure

Retain existing repository layout (single library + specs). Add `contracts/` under spec folder and test contract files under `tests/contract/`.

**Structure Decision**: Option 1 (single project)

## Phase 0: Outline & Research (research.md Summary)

Focus Areas:

1. Model Base Selection: Choose a permissive 7B or smaller instruct-capable model with good multilingual tokenization (e.g., Mistral 7B Instruct) suitable for Toaripi (low-resource) adaptation.
2. Tokenization Strategy: Evaluate existing tokenizer coverage for Toaripi; fallback: train SentencePiece unigram model on combined Toaripi corpus + small English sample; avoid fragmentation of Toaripi morphemes.
3. Fine-Tune Approach: LoRA with low rank (r=16) + 4-bit quantization for memory efficiency; gradient checkpointing for resource limits.
4. Data Augmentation: Only alignment-preserving (no hallucinated parallel pairs). Generate educational prompt templates from existing aligned verse semantics cautiously; avoid synthetic overfitting.
5. Safety Filtering: Keyword set + simple pattern classification; human-in-loop review for evaluation pack.
6. Deterministic Stable Mode: Fixed seed + temperature clamp + top-k fixed; record seed & decoding params.
7. Latency Optimization: Quantized GGUF for llama.cpp inference; batch size=1; prompt trimming; early stop on sentence count.
8. Evaluation Metrics: Loss/perplexity on validation subset; qualitative rubric (fluency, cultural safety, simplicity) via evaluation pack; track latency distribution.
9. Logging & Metadata: JSON lines file with fields: timestamp, request_id, content_type, tokens_in/out, latency_ms, safety_status, model_version.

Decisions (abbrev):

- Base Model: Mistral 7B Instruct (fits ≤7B constraint, strong efficiency)
- Adaptation: LoRA only (no full fine-tune) for speed & reproducibility
- Quantization: Q4_K_M GGUF for inference edge; 4-bit NF4 during training load (PEFT) where feasible
- Tokenizer: Reuse base tokenizer first; measure OOV/fragmentation; if >15% subword fragmentation on Toaripi word list, train custom SP model
- Min Data: 150 aligned pairs (internal); still respects constitution minimum (upgrade justification recorded)

## Phase 1: Design & Contracts

Deliverables: data-model.md, contracts (API + internal service function schemas), quickstart.md, initial failing contract tests.

### Data Model Highlights

Entities refined with validation constraints (ids UUIDv4, topic length 2–80, etc.). See `data-model.md`.

### API Contract Overview (FastAPI-style logical design)

| Endpoint | Method | Purpose | Request Schema | Response Schema |
|----------|--------|---------|----------------|-----------------|
| /generate | POST | Generate educational content | GenerateRequest | GenerateResponse |
| /health | GET | Health/latency probe | n/a | HealthStatus |
| /evaluation-pack | POST | Produce evaluation pack artifacts | EvalPackRequest | EvalPackResponse |
| /safety/rules | GET | List active safety rules | n/a | SafetyRulesResponse |
| /metadata/run | GET | Model & data version metadata | n/a | RunMetadata |

Contract Schemas (summary):

- GenerateRequest: topic, content_type(enum), target_length(optional bounded), mode(enum stable|standard), include_english_support(bool)
- GenerateResponse: content_type, content_body(string or list), metadata(object), safety_status(enum), disclaimer_shown(bool)
- EvalPackRequest: sample_count(optional default 12)
- EvalPackResponse: pack_id, items(array of simplified generated artifacts metadata)

### Quickstart Outline (planned quickstart.md)

1. Install dependencies
2. Place aligned CSV in data/
3. Run preprocessing script to validate & produce train/val splits
4. Launch fine-tune script with YAML config (example provided)
5. Export quantized model artifact
6. Start local generation server (FastAPI) offline
7. Run sample generation requests
8. View logs & evaluation pack

### Constitution Re-Check (Post Design)

All principles still met. Added justification recorded for raising data minimum.

## Phase 2: Task Planning Approach (Description Only)

Task Generation Strategy:

- Each endpoint → contract test + implementation task (test first)
- Each entity field group → model/dataclass + validation tests
- Safety filter pipeline → lexical list + threshold tests before integration
- Training pipeline → config parse test → data loader test → tokenization coverage test → LoRA adapter integration test → generation sanity test
- Performance tasks → latency measurement harness

Ordering Strategy:

1. Data validation & entities
2. Contract schemas + tests
3. Safety rule engine tests
4. Generation service (mock model) tests
5. Training pipeline scaffolding tests
6. Real model integration & quantization export
7. Performance & evaluation pack

Parallelizable ([P]): entity models, individual endpoint contract tests, safety rule list curation.

Estimated tasks: ~28.

## Complexity Tracking

(No violations requiring justification.)

## Progress Tracking

**Phase Status**:

- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [ ] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:

- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented

---
*Based on Constitution v1.0.0 - See `/memory/constitution.md`*
