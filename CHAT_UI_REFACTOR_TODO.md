# Toaripi SLM Chat UI & Refactor Backlog

This document enumerates the refactor and feature tasks required to introduce a safe, educational Chat UI and supporting infrastructure. It is optimized for a coding agent to execute systematically while honoring the project "constitution" (educational focus, cultural respect, offline capability, safety, small-model efficiency).

---
## Constitution (Guiding Principles)

1. Educational Scope Only: Outputs must support stories, vocabulary, Q&A, dialogues, comprehension — no general open‑ended chat or doctrinal/theological expansion.
2. Cultural & Age Appropriateness: Primary school audience; avoid adult, violent, or culturally insensitive content.
3. Safety & Guardrails First: Centralized input/output scanning with deterministic blocking and regeneration pathways.
4. Offline Friendly: No external API calls required for normal operation; model + data local.
5. Efficiency: Target ≤7B parameter base models; support quantized GGUF for CPU / Raspberry Pi.
6. Reproducibility: Deterministic configs, dataset manifests (hashes), documented generation parameters.
7. Transparency: Clear prompts, tracked safety decisions, structured logging (no PII persisted).
8. Minimal Surface Area: Clean abstraction boundaries (config, data pipeline, training, inference, serving, UI).
9. Extensibility: New educational content types can be added via prompt modules + generator interface.
10. Testability: Unit + integration + safety regression + performance baselines automated.

---
 
## Status Legend

- `not-started` – No implementation yet.
- `in-progress` – Actively being implemented (limit: 1 at a time for focus).
- `completed` – Implemented & tests passing.

Agents MUST update both the Markdown checklist and any internal task tracker when changing status.

---
 
## High-Level Phasing

- Phase 0: Core refactors (config, generator, safety, data pipeline).
- Phase 1: Model loading abstraction, streaming groundwork, logging.
- Phase 2: Chat API (REST + WebSocket) and minimal UI.
- Phase 3: Tests (unit/integration/perf/safety) + docs.
- Phase 4: Polish (CLI consistency, pre-commit, model card, release prep).

---
 
## Task Table (Human Readable)

| ID | Title | Priority | Status | Depends On | Summary |
|----|-------|----------|--------|------------|---------|
| 1 | Unify configuration system | P0 | not-started | — | Central pydantic configs + env overrides |
| 2 | Introduce ProjectPaths helper | P0 | not-started | 1 | Standard path resolution |
| 3 | Abstract generation interface | P0 | not-started | 1 | `EducationalContentGenerator` + spec |
| 4 | Centralize prompt templates | P0 | not-started | 3 | Reusable educational prompt modules |
| 5 | Implement safety & guardrails layer | P0 | not-started | 3 | Input/output scanning + decisions |
| 6 | Refactor data pipeline | P0 | not-started | 1 | Pipeline + dataset manifest/hash |
| 7 | LoRA & training utilities module | P1 | not-started | 1 | Standardized LoRA config builder |
| 8 | Model loader abstraction | P1 | not-started | 3 | HF + GGUF loading & streaming hooks |
| 9 | Quantization/export pipeline | P1 | not-started | 8 | HF→GGUF export + CLI flag |
| 10 | Add structured logging & metrics | P1 | not-started | 1 | JSON logs + counters + metrics route |
| 11 | Implement context trimming logic | P1 | not-started | 3 | Sliding token window management |
| 12 | Add streaming generation support | P1 | not-started | 8,11 | Iterator/ callback streaming APIs |
| 13 | Chat REST endpoint | P1 | not-started | 3,4,5,8,11 | POST `/api/chat` |
| 14 | Chat WebSocket endpoint | P1 | not-started | 12,13 | `/ws/chat` streaming tokens |
| 15 | Minimal chat UI (HTML/JS) | P1 | not-started | 13 | Browser UI with streaming fallback |
| 16 | Session/stateless strategy | P1 | not-started | 11 | History trimming / optional session IDs |
| 17 | Post-processing hooks | P1 | not-started | 3,5 | Educational closure enforcement |
| 18 | Performance baseline tests | P2 | not-started | 8,12,13 | Latency & memory baselines |
| 19 | Unit test suite expansion | P0 | not-started | 1–6 | Core coverage (config, prompts, safety) |
| 20 | Integration tests for chat | P1 | not-started | 13–16 | REST & WS behavior tests |
| 21 | Perf & edge case tests | P2 | not-started | 18,20 | Timeouts, rapid WS, long input |
| 22 | Documentation updates | P1 | not-started | 13–17 | Chat quickstart, API, teacher guide |
| 23 | CLI consistency refactor | P2 | not-started | 1,7,9 | Shared options / help standardization |
| 24 | Pre-commit & linting setup | P2 | not-started | 1 | black/ruff/mypy/isort (or chosen set) |
| 25 | Add model card generation | P2 | not-started | 7,9 | Automated metadata export |
| 26 | Release prep & versioning | P2 | not-started | All core | Tag v0.3.0-chat + changelog |
| 27 | Safety regression corpus | P1 | not-started | 5 | Curated disallowed samples gating CI |
| 28 | Offline validation | P1 | not-started | 8,13 | No-network inference test harness |
| 29 | Feature flagging for chat | P0 | not-started | 1 | `CHAT_UI_ENABLED` gating routes/UI |
| 30 | Context: educational constraints | P0 | not-started | 4,5 | Embed constitution in system prompt |

---
 
## Detailed Task Specifications

### 1. Unify configuration system

Goal: Single authoritative configuration layer.

Deliverables:

- `src/toaripi_slm/config/models.py` (pydantic schemas)
- `src/toaripi_slm/config/loader.py` with env override (`TOARIPI__SECTION__KEY` syntax)
- Update scripts & launcher to consume objects

Acceptance:

- Invalid schema raises clear error; tests in Task 19 pass.

### 2. ProjectPaths helper

Goal: Eliminate ad-hoc relative paths.

Deliverables: `src/toaripi_slm/utils/paths.py` with properties (root, data_raw, data_processed, models_hf, models_gguf, configs).

Acceptance: Replaced usages in at least training script & launcher.

### 3. Generation interface

Goal: Stable surface for all educational content generation.

Deliverables: `EducationalContentGenerator`, `GenerationRequestSpec`, `Turn` dataclass.

Acceptance: Chat API (Task 13) depends only on interface, not raw model objects.

### 4. Prompt templates

Goal: Consistency & reuse.

Deliverables: `inference/prompts/{story,vocab,comprehension,dialogue}.py` + registry.

Acceptance: Each returns `PromptBuildResult(prompt:str, metadata:dict)`.

### 5. Safety & guardrails

Goal: Central enforcement.

Deliverables: `safety.py` with `evaluate_input`, `evaluate_output`, `SafetyDecision`.

Acceptance: Block list & regeneration strategy; tests (19,27).

### 6. Data pipeline refactor

Goal: Reusable, testable data prep.

Deliverables: `data/pipeline.py`; dataset manifest JSON with hash of source CSV + record counts.

Acceptance: Re-running without source change skips heavy work.

 
### 7. LoRA utilities

Goal: Standardized training config assembly.
Deliverables: `core/lora.py` helper returning configured PEFT model.
Acceptance: Training script simplified; diff reduces repeated LoRA code.

 
### 8. Model loader abstraction

Goal: Uniform load path for HF & GGUF.
Deliverables: `inference/model_loader.py` with `load(model_format)` and lazy tokenizer.
Acceptance: Supports `hf` path + returns NotImplemented for GGUF until Task 9 done (documented).

 
### 9. Quantization/export pipeline

Goal: Enable offline optimized inference.
Deliverables: Export function + CLI flag or new `scripts/export_model.py`.
Acceptance: Produces `.gguf` file placed in `models/gguf/` with metadata log.

 
### 10. Structured logging & metrics

Goal: Observability baseline.
Deliverables: `utils/logging.py`; optional `/api/metrics` returning counters.
Acceptance: Chat requests increment token + request counters.

 
### 11. Context trimming logic

Goal: Token budget control.
Deliverables: Function `trim_history(turns, max_tokens)` estimating tokens via tokenizer and pruning.
Acceptance: Deterministic output; unit test verifies ordering.

 
### 12. Streaming generation support

Goal: Low-latency user feedback.
Deliverables: Integration with `TextIteratorStreamer` (HF) & placeholder for llama.cpp.
Acceptance: WebSocket prototype can stream tokens (Task 14).

 
### 13. Chat REST endpoint

Goal: Baseline interaction.
Deliverables: `app/api/chat.py` POST `/api/chat`.
Acceptance: Returns structured JSON with safety + usage.

 
### 14. Chat WebSocket endpoint

Goal: Interactive streaming.
Deliverables: `/ws/chat`; token event frames + final summary frame.
Acceptance: Integration test (Task 20) passes.

 
### 15. Minimal chat UI

Goal: Accessible browser UI.
Deliverables: `chat.html`, `chat.js`; fallback to REST if WS unsupported.
Acceptance: Manual smoke test renders & sends message.

 
### 16. Session/stateless strategy

Goal: Simplicity & scalability.
Deliverables: Stateless default + ephemeral cache (dict + TTL) optional.
Acceptance: Documented in code + README section.

 
### 17. Post-processing hooks

Goal: Output shaping.
Deliverables: Hook chain (list of callables) ensuring educational closure and style compliance.
Acceptance: Unit test asserts closure sentence appended if absent.

 
### 18. Performance baseline tests

Goal: Track regressions.
Deliverables: `tests/perf/` storing results JSON (latency, mem) for short prompt.
Acceptance: Test executable locally without external deps.

 
### 19. Unit test suite expansion

Goal: Core reliability.
Deliverables: New tests in `tests/unit/` for tasks 1–6 & 11.
Acceptance: Coverage report ≥85% on new modules.

 
### 20. Integration tests for chat

Goal: End-to-end correctness.
Deliverables: `tests/integration/test_chat_api.py`, `test_chat_ws.py`.
Acceptance: All green in CI.

### 21. Perf & edge case tests
Goal: Robustness under stress & anomalies.
Deliverables: Timeout, rapid WS send, long input cases.
Acceptance: Graceful errors (HTTP 400 / application error code).

### 22. Documentation updates
Goal: User adoption & clarity.
Deliverables: `CHAT_QUICKSTART.md`, updates to API & teacher guides.
Acceptance: Links added to README.

### 23. CLI consistency refactor
Goal: Unified UX.
Deliverables: Shared option builder (e.g., function returning Click options).
Acceptance: Help text aligned across scripts.

### 24. Pre-commit & linting setup
Goal: Code quality automation.
Deliverables: `.pre-commit-config.yaml`, updated contributing docs.
Acceptance: Pre-commit passes locally; CI hook (optional).

### 25. Model card generation
Goal: Transparency.
Deliverables: Script or function emitting `model_card.md` with metadata.
Acceptance: Runs post-training automatically (optional flag).

### 26. Release prep & versioning
Goal: Formal release.
Deliverables: Changelog entry, tag v0.3.0-chat, README highlight.
Acceptance: All tests green; version bump.

### 27. Safety regression corpus
Goal: Prevent drift.
Deliverables: Corpus JSON + test ensuring blocked items remain blocked.
Acceptance: CI fails if any item passes safety.

### 28. Offline validation
Goal: Guarantee isolation.
Deliverables: Test that monkeypatches network libs & asserts no calls.
Acceptance: Pass in CI; doc note in quickstart.

### 29. Feature flagging for chat
Goal: Controlled rollout.
Deliverables: `CHAT_UI_ENABLED` in config gating route registration + UI template inclusion.
Acceptance: Disabled state yields 404 on chat endpoints.

### 30. Context: educational constraints embedding
Goal: Prompt-level enforcement.
Deliverables: System prompt builder referencing constitution; doc snippet.
Acceptance: Unit test ensures banned categories not present in system preamble.

---
## Machine-Readable Task List (JSON)
```json
{
  "version": "1.0",
  "phases": [0,1,2,3,4],
  "tasks": [
    {"id":1, "title":"Unify configuration system", "priority":"P0", "status":"not-started", "deps":[]},
    {"id":2, "title":"Introduce ProjectPaths helper", "priority":"P0", "status":"not-started", "deps":[1]},
    {"id":3, "title":"Abstract generation interface", "priority":"P0", "status":"not-started", "deps":[1]},
    {"id":4, "title":"Centralize prompt templates", "priority":"P0", "status":"not-started", "deps":[3]},
    {"id":5, "title":"Implement safety & guardrails layer", "priority":"P0", "status":"not-started", "deps":[3]},
    {"id":6, "title":"Refactor data pipeline", "priority":"P0", "status":"not-started", "deps":[1]},
    {"id":7, "title":"LoRA & training utilities module", "priority":"P1", "status":"not-started", "deps":[1]},
    {"id":8, "title":"Model loader abstraction", "priority":"P1", "status":"not-started", "deps":[3]},
    {"id":9, "title":"Quantization/export pipeline", "priority":"P1", "status":"not-started", "deps":[8]},
    {"id":10, "title":"Add structured logging & metrics", "priority":"P1", "status":"not-started", "deps":[1]},
    {"id":11, "title":"Implement context trimming logic", "priority":"P1", "status":"not-started", "deps":[3]},
    {"id":12, "title":"Add streaming generation support", "priority":"P1", "status":"not-started", "deps":[8,11]},
    {"id":13, "title":"Chat REST endpoint", "priority":"P1", "status":"not-started", "deps":[3,4,5,8,11]},
    {"id":14, "title":"Chat WebSocket endpoint", "priority":"P1", "status":"not-started", "deps":[12,13]},
    {"id":15, "title":"Minimal chat UI (HTML/JS)", "priority":"P1", "status":"not-started", "deps":[13]},
    {"id":16, "title":"Session/stateless strategy", "priority":"P1", "status":"not-started", "deps":[11]},
    {"id":17, "title":"Post-processing hooks", "priority":"P1", "status":"not-started", "deps":[3,5]},
    {"id":18, "title":"Performance baseline tests", "priority":"P2", "status":"not-started", "deps":[8,12,13]},
    {"id":19, "title":"Unit test suite expansion", "priority":"P0", "status":"not-started", "deps":[1,2,3,4,5,6]},
    {"id":20, "title":"Integration tests for chat", "priority":"P1", "status":"not-started", "deps":[13,14,15,16]},
    {"id":21, "title":"Perf & edge case tests", "priority":"P2", "status":"not-started", "deps":[18,20]},
    {"id":22, "title":"Documentation updates", "priority":"P1", "status":"not-started", "deps":[13,14,15,16,17]},
    {"id":23, "title":"CLI consistency refactor", "priority":"P2", "status":"not-started", "deps":[1,7,9]},
    {"id":24, "title":"Pre-commit & linting setup", "priority":"P2", "status":"not-started", "deps":[1]},
    {"id":25, "title":"Add model card generation", "priority":"P2", "status":"not-started", "deps":[7,9]},
    {"id":26, "title":"Release prep & versioning", "priority":"P2", "status":"not-started", "deps":[1,3,5,8,13,14,15,19,20]},
    {"id":27, "title":"Safety regression corpus", "priority":"P1", "status":"not-started", "deps":[5]},
    {"id":28, "title":"Offline validation", "priority":"P1", "status":"not-started", "deps":[8,13]},
    {"id":29, "title":"Feature flagging for chat", "priority":"P0", "status":"not-started", "deps":[1]},
    {"id":30, "title":"Context: educational constraints", "priority":"P0", "status":"not-started", "deps":[4,5]}
  ]
}
```

---
## Execution Protocol for Agents
1. Select the highest-priority `not-started` task whose dependencies are `completed`.
2. Mark it `in-progress` (both here and internal tracker if any).
3. Implement with minimal, well-scoped commits.
4. Add/Update tests; ensure all tests pass locally.
5. Update status to `completed`; note any follow-up tasks (append if needed).
6. NEVER work on more than one `in-progress` task concurrently.
7. Maintain constitution alignment (run safety tests after relevant changes).

---
## Open Questions (If Future Expansion Needed)
- Should we add a lightweight semantic classifier for safety beyond keywords? (Future milestone)
- Do we need a persisted conversation store for teacher session summaries? (Out of current scope)

---
## Changelog Placeholder (Populate During Implementation)
- v0.3.0-chat: (TBD)

---
End of backlog.
