# Tasks: Offline-Capable Toaripi Educational SLM

Input: Design documents under `specs/001-offline-capable-toaripi/`
Prerequisites: plan.md (required), research.md, data-model.md, contracts/

## Conventions

- [P] denotes tasks that can proceed in parallel (different files / no deps)
- TDD ordering: write failing tests before implementing
- Paths assume single-project layout (`src/toaripi_slm/`, `tests/`)

## Phase 0: Confirm Environment

- [ ] T000 Verify base installation script passes (`scripts/verify_setup.py`) and record Python version

## Phase 1: Setup & Scaffolding

- [ ] T001 Create directories: `src/toaripi_slm/{core,data,inference,utils,safety}` and `tests/{contract,integration,unit,performance}`
- [ ] T002 Add `pyproject` style metadata or enhance `setup.py` with entry points (defer if policy forbids) – ensure package import works
- [ ] T003 [P] Add logging config module `src/toaripi_slm/utils/logging.py` (JSON line logger helper)
- [ ] T004 [P] Add constants module `src/toaripi_slm/utils/constants.py` (disclaimer text, latency targets, cache limits)
- [ ] T005 [P] Add safety rules seed file `data/safety_rules.csv` (medium/high severity columns)

## Phase 2: Contract Tests (Write FIRST, expect failures)

- [ ] T006 [P] Contract test POST /generate `tests/contract/test_generate_post.py`
- [ ] T007 [P] Contract test GET /health `tests/contract/test_health_get.py`
- [ ] T008 [P] Contract test POST /evaluation-pack `tests/contract/test_evaluation_pack_post.py`
- [ ] T009 [P] Contract test GET /safety/rules `tests/contract/test_safety_rules_get.py`
- [ ] T010 [P] Contract test GET /metadata/run `tests/contract/test_metadata_run_get.py`

## Phase 3: Integration Tests (User Stories & Edge Cases)

- [ ] T011 [P] Integration test story generation latency & length `tests/integration/test_story_generation.py`
- [ ] T012 [P] Integration test vocabulary generation list structure `tests/integration/test_vocabulary_generation.py`
- [ ] T013 [P] Integration test QA generation pairs `tests/integration/test_qa_generation.py`
- [ ] T014 [P] Integration test dialogue speaker alternation `tests/integration/test_dialogue_generation.py`
- [ ] T015 [P] Integration test safety blocking restricted topic `tests/integration/test_safety_blocking.py`
- [ ] T016 [P] Integration test caching duplicate prompt reuse `tests/integration/test_caching.py`
- [ ] T017 [P] Integration test stable mode similarity (≥90% token overlap) `tests/integration/test_stable_mode.py`
- [ ] T018 [P] Integration test evaluation pack 12 samples `tests/integration/test_evaluation_pack.py`
- [ ] T019 [P] Integration test corpus threshold training refusal (<150 pairs) `tests/integration/test_corpus_threshold.py`
- [ ] T020 [P] Integration test length caps and validation errors `tests/integration/test_input_validation.py`

## Phase 4: Data & Models (Domain Objects)

- [ ] T021 [P] Pydantic model ContentRequest `src/toaripi_slm/data/models.py`
- [ ] T022 [P] Pydantic model GeneratedContentArtifact `src/toaripi_slm/data/models.py`
- [ ] T023 [P] Pydantic model ParallelCorpusEntry `src/toaripi_slm/data/models.py`
- [ ] T024 [P] Pydantic model SafetyRule `src/toaripi_slm/data/models.py`
- [ ] T025 [P] Pydantic model EvaluationPack `src/toaripi_slm/data/models.py`
- [ ] T026 Validation utilities (length bounds, dialogue speaker checks) `src/toaripi_slm/data/validators.py`

## Phase 5: Core Services

- [ ] T027 [P] SafetyRuleLoader service `src/toaripi_slm/safety/rule_loader.py`
- [ ] T028 [P] SafetyScreening service (apply thresholds) `src/toaripi_slm/safety/screening.py`
- [ ] T029 [P] Checksum & reproducibility helper `src/toaripi_slm/utils/reproducibility.py`
- [ ] T030 [P] Cache service (5 min TTL, size cap) `src/toaripi_slm/core/cache.py`
- [ ] T031 GenerationService skeleton (interface only) `src/toaripi_slm/core/generation_service.py`
- [ ] T032 EvaluationPackService `src/toaripi_slm/core/evaluation_pack_service.py`
- [ ] T033 MetadataService (run + per-request metadata) `src/toaripi_slm/core/metadata_service.py`
- [ ] T034 Logging integration (structured JSON lines) `src/toaripi_slm/core/logging_integration.py`

## Phase 6: Model & Inference Integration

- [ ] T035 Loader: load HF base + optional LoRA adapter stub `src/toaripi_slm/inference/model_loader.py`
- [ ] T036 Tokenization coverage utility (report fragmentation) `src/toaripi_slm/inference/token_coverage.py`
- [ ] T037 Generation pipeline implementation (decoding params, stable mode) `src/toaripi_slm/inference/generator.py`
- [ ] T038 Safety hook integration into generation pipeline `src/toaripi_slm/inference/hooks.py`

## Phase 7: API Layer (FastAPI)

- [ ] T039 FastAPI app factory `app/api/factory.py`
- [ ] T040 POST /generate endpoint implementation `app/api/generate.py`
- [ ] T041 GET /health endpoint implementation `app/api/health.py`
- [ ] T042 POST /evaluation-pack endpoint implementation `app/api/evaluation_pack.py`
- [ ] T043 GET /safety/rules endpoint implementation `app/api/safety_rules.py`
- [ ] T044 GET /metadata/run endpoint implementation `app/api/metadata_run.py`
- [ ] T045 Error handling & standardized error responses `app/api/errors.py`

## Phase 8: Training & Preprocessing Pipeline

- [ ] T046 Data validation script (row count ≥150, checksum) `scripts/validate_parallel.py`
- [ ] T047 Preprocessing script (train/val split, coverage report) `scripts/preprocess.py`
- [ ] T048 Training script with LoRA config load `scripts/finetune.py`
- [ ] T049 Merge & export script (merged HF model) `scripts/merge_export.py`
- [ ] T050 Quantization script to GGUF `scripts/quantize.py`

## Phase 9: Maintenance & Housekeeping

- [ ] T051 Log retention purge (>30 days) `src/toaripi_slm/utils/retention.py`
- [ ] T052 Cache cleanup task `src/toaripi_slm/core/cache.py` (scheduled function)
- [ ] T053 Safety rules reload utility `src/toaripi_slm/safety/rule_loader.py`

## Phase 10: Unit Tests (Fine-Grained)

- [ ] T054 [P] Unit tests validators `tests/unit/test_validators.py`
- [ ] T055 [P] Unit tests safety screening thresholds `tests/unit/test_safety_screening.py`
- [ ] T056 [P] Unit tests cache behavior `tests/unit/test_cache.py`
- [ ] T057 [P] Unit tests reproducibility helper `tests/unit/test_reproducibility.py`
- [ ] T058 [P] Unit tests evaluation pack composition `tests/unit/test_evaluation_pack.py`

## Phase 11: Performance & Stability

- [ ] T059 Performance harness (measure p95 latencies) `tests/performance/test_latency.py`
- [ ] T060 Stable mode similarity test confirm ≥90% overlap `tests/performance/test_stable_mode_similarity.py`

## Phase 12: Documentation & Polish

- [ ] T061 Update `readme.md` usage section with new endpoints
- [ ] T062 Add API reference `docs/api/endpoints.md`
- [ ] T063 Add architecture overview `docs/architecture.md`
- [ ] T064 Add safety policy summary `docs/safety/overview.md`
- [ ] T065 Add reproducibility guide `docs/reproducibility.md`
- [ ] T066 Manual QA checklist `docs/qa/manual_checklist.md`

## Phase 13: Final Validation

- [ ] T067 CI script / GitHub Action skeleton (lint, tests) `.github/workflows/ci.yml` (if allowed)
- [ ] T068 Verify all contract & integration tests pass
- [ ] T069 Verify performance harness meets latency targets
- [ ] T070 Generate evaluation pack and capture review notes

## Dependencies Overview

- Contract tests (T006–T010) must exist before API implementations (T040–T044)
- Models (T021–T026) precede services (T027–T034)
- Services precede generation pipeline (T037) and API endpoints
- Generation pipeline (T037) depends on model loader (T035) & safety services (T027–T028) & cache (T030)
- EvaluationPackService (T032) depends on GenerationService (T031) + models
- Performance tests (T059–T060) after core + API layers implemented
- Documentation tasks after functionality stabilized

## Parallel Execution Examples

Example early parallel batch:
T003, T004, T005 (independent files)

Contract test batch:
T006–T010 in parallel

Integration test batch:
T011–T020 in parallel (distinct files)

Unit test batch:
T054–T058 in parallel

## Validation Checklist

- All contracts have corresponding contract test tasks ✓
- All entities have model tasks ✓
- Tests precede implementations ✓
- Parallel tasks reference unique files ✓
- Latency & reproducibility explicitly tested ✓
- Safety thresholds tested ✓

## Notes

- Keep generation service abstract initially to allow swap of backend model implementation.
- Stable mode: enforce fixed seed + deterministic sampling (temperature, top_p constraints) – tested in T017 & T060.
- Reproducibility metadata must be attached before logging generation completion.
