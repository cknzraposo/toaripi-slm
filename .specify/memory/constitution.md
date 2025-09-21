# Toaripi SLM Constitution

## Core Principles

### I. Minimal Viable Educational Model
Deliver the smallest reliable Toaripi educational language model (≤7B params) that can generate age‑appropriate stories, vocabulary exercises, Q&A, and dialogues. Every addition must support core educational use; non‑essential/chat features are excluded.

### II. Reproducible Data & Training Pipeline
All training runs must be reproducible from: (1) versioned parallel English↔Toaripi CSV sources, (2) immutable YAML configs, (3) deterministic preprocessing scripts. No manual, undocumented data edits. Outputs (artifacts, logs, metrics) are timestamped and checksum‑tracked.

### III. Test-First Quality Gates (NON-NEGOTIABLE)
Before implementation: write (or update) unit + minimal integration tests for data loading, prompt formatting, and generation contracts. Pipeline merges require: import check, config parse success, data schema validation, at least one training dry‑run test (mock), and generation sanity checks (non‑empty, safe, Toaripi tokens present).

### IV. Responsible & Culturally Safe Content
Model and data must exclude: theological exposition, adult/violent themes, cultural misappropriation. Automated filters + manual spot checks are required. Any flagged content blocks release until resolved.

### V. Simplicity, Observability & Versioning
Prefer straight Python + Hugging Face ecosystem; avoid premature abstractions. Structured JSON logs for: data stats, training metrics (loss, perplexity), generation test outputs. Semantic versioning MAJOR.MINOR.PATCH for the library; training runs tagged with model semver + run hash.

## Additional Constraints & Standards
- Model size: ≤7B parameters (base); quantized target ≤5GB (GGUF Q4 or better).  
- Hardware baseline: CPU-only 8GB RAM (Raspberry Pi 5 class) must run inference.  
- Offline-first: No network dependencies at runtime (post-download).  
- Config formats: Only YAML for training/inference parameters.  
- Data minimum: ≥100 aligned verse pairs before any fine‑tune attempt.  
- Storage hygiene: Large artifacts in `models/` only; no raw checkpoints in `src/`.  
- Documentation: Each new script or module requires a top docstring with purpose + usage snippet.

## Development Workflow & Quality Gates
1. Open Issue: Describe change (data, training, inference, generation feature).  
2. Define Contract: Update or add tests first (Red).  
3. Implement until tests pass (Green).  
4. Refactor for clarity (Refactor).  
5. Pre-Merge Automated Gates:  
   - Import/lint pass (ruff or flake8 if configured)  
   - `pytest -q` green  
   - Config load test (`configs/training/*.yaml`)  
   - Data validation (`sample_parallel.csv` schema)  
   - Mock generation test produces ≥1 Toaripi token + ≥3 sentences for story mode  
6. Human Review: Check cultural/ethical compliance.  
7. Tag & Record: Commit message includes `[train-run:<run_id>]` if producing a model.  
8. Release: Update `CHANGELOG` (if present) + bump version when interface or artifact changes.  

Quality Gate Failure Policy: Any gate failure blocks merge; partial waivers require explicit maintainer sign-off recorded in PR comments.

## Governance
This Constitution overrides informal practices. Amendments require:
- Issue proposing change (motivation + migration impact)
- PR updating this file with version bump
- Approval by at least one language advocate + one maintainer

All PR reviewers must verify:
- Principles upheld
- Tests present & meaningful
- No scope creep beyond educational objectives

Non-compliant contributions are rejected or refactored before merge.

Version: 1.0.0 | Ratified: 2025-09-18 | Last Amended: 2025-09-18