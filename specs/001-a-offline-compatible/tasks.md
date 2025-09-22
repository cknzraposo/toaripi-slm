# Tasks: Interactive CLI Management Tool for Toaripi SLM

**Input**: Design documents from `/specs/001-a-offline-compatible/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Tech stack: Python 3.10+, click, rich, pydantic
   → Libraries: transformers, pytest
   → Structure: Single CLI project with integrated ML pipeline
2. Load design documents:
   → data-model.md: 5 entities (TrainingSession, Dataset, ModelConfig, Checkpoint, Log)
   → contracts/: 12 CLI commands across 4 groups (data, train, model, system)
   → research.md: Framework decisions (click, rich, pydantic)
   → quickstart.md: User workflows for validation testing
3. Generate tasks by category:
   → Setup: CLI project structure, dependencies, tooling
   → Tests: CLI contract tests, integration tests, validation tests
   → Core: Data models, CLI commands, interactive mode
   → Integration: ML pipeline, file management, logging
   → Polish: Documentation, performance optimization, error handling
4. Apply task rules:
   → Different CLI commands = mark [P] for parallel
   → Data models = mark [P] for parallel
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph with educational validation
7. Create parallel execution examples
8. SUCCESS: 42 tasks ready for TDD execution
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- All file paths relative to repository root
- Educational validation and constitutional compliance integrated

## Path Conventions
- **CLI Structure**: `src/toaripi_slm/cli/` for command modules
- **Data Models**: `src/toaripi_slm/models/` for pydantic models
- **Tests**: `tests/cli/` for CLI tests, `tests/integration/` for workflows
- **Configuration**: CLI configuration management in `src/toaripi_slm/config/`

## Phase 3.1: Setup & Infrastructure

- [ ] T001 Create CLI project structure with src/toaripi_slm/cli/, src/toaripi_slm/models/, src/toaripi_slm/config/
- [ ] T002 Initialize Python CLI project with click, rich, pydantic dependencies in pyproject.toml
- [ ] T003 [P] Configure pytest, black, and ruff for CLI code quality in pyproject.toml
- [ ] T004 [P] Create CLI configuration schema in src/toaripi_slm/config/schema.py with pydantic models
- [ ] T005 [P] Implement CLI configuration manager in src/toaripi_slm/config/manager.py for user settings

## Phase 3.2: Data Models First (Parallel Implementation)

- [ ] T006 [P] TrainingSession pydantic model in src/toaripi_slm/models/session.py with validation rules
- [ ] T007 [P] Dataset pydantic model in src/toaripi_slm/models/dataset.py with cultural validation
- [ ] T008 [P] ModelConfiguration pydantic model in src/toaripi_slm/models/config.py with educational settings
- [ ] T009 [P] ModelCheckpoint pydantic model in src/toaripi_slm/models/checkpoint.py with edge constraints
- [ ] T010 [P] TrainingLog pydantic model in src/toaripi_slm/models/log.py with structured logging
- [ ] T011 [P] Supporting enums (SessionStatus, ContentType, AgeGroup) in src/toaripi_slm/models/enums.py
- [ ] T012 [P] Composite types (DatasetStats, TrainingProgress) in src/toaripi_slm/models/types.py

## Phase 3.3: CLI Contract Tests (TDD - MUST FAIL BEFORE IMPLEMENTATION)

- [ ] T013 [P] Contract test for `toaripi data prepare` in tests/cli/test_data_prepare.py
- [ ] T014 [P] Contract test for `toaripi data validate` in tests/cli/test_data_validate.py  
- [ ] T015 [P] Contract test for `toaripi data list` in tests/cli/test_data_list.py
- [ ] T016 [P] Contract test for `toaripi train start` in tests/cli/test_train_start.py
- [ ] T017 [P] Contract test for `toaripi train stop` in tests/cli/test_train_stop.py
- [ ] T018 [P] Contract test for `toaripi train status` in tests/cli/test_train_status.py
- [ ] T019 [P] Contract test for `toaripi train list` in tests/cli/test_train_list.py
- [ ] T020 [P] Contract test for `toaripi model list` in tests/cli/test_model_list.py
- [ ] T021 [P] Contract test for `toaripi model test` in tests/cli/test_model_test.py
- [ ] T022 [P] Contract test for `toaripi model export` in tests/cli/test_model_export.py
- [ ] T023 [P] Contract test for `toaripi interactive` in tests/cli/test_interactive.py
- [ ] T024 [P] Contract test for `toaripi config` and `toaripi version` in tests/cli/test_system.py

## Phase 3.4: Integration Tests (User Scenarios from Quickstart)

- [ ] T025 [P] Integration test: Complete data preparation workflow in tests/integration/test_data_workflow.py
- [ ] T026 [P] Integration test: Full training session lifecycle in tests/integration/test_training_workflow.py
- [ ] T027 [P] Integration test: Model testing and export workflow in tests/integration/test_model_workflow.py
- [ ] T028 [P] Integration test: Interactive mode user journey in tests/integration/test_interactive_workflow.py
- [ ] T029 [P] Integration test: Educational content validation pipeline in tests/integration/test_validation_workflow.py

## Phase 3.5: Core CLI Implementation (Only After Tests Fail)

### CLI Framework & Entry Points
- [ ] T030 Main CLI entry point with click groups in src/toaripi_slm/cli/main.py
- [ ] T031 [P] Data command group implementation in src/toaripi_slm/cli/data_commands.py
- [ ] T032 [P] Training command group implementation in src/toaripi_slm/cli/train_commands.py
- [ ] T033 [P] Model command group implementation in src/toaripi_slm/cli/model_commands.py
- [ ] T034 [P] System commands (config, version) in src/toaripi_slm/cli/system_commands.py

### Interactive Mode & Rich UI
- [ ] T035 Interactive menu system with rich in src/toaripi_slm/cli/interactive.py
- [ ] T036 [P] Rich terminal UI components (progress, tables, panels) in src/toaripi_slm/cli/ui.py
- [ ] T037 [P] Educational error formatting with rich in src/toaripi_slm/cli/errors.py

## Phase 3.6: Service Layer & ML Integration

- [ ] T038 Data preparation service in src/toaripi_slm/services/data_service.py with CSV processing
- [ ] T039 Training management service in src/toaripi_slm/services/training_service.py with checkpoint handling
- [ ] T040 Model management service in src/toaripi_slm/services/model_service.py with GGUF export
- [ ] T041 Educational content validator in src/toaripi_slm/services/validation_service.py
- [ ] T042 File system manager in src/toaripi_slm/services/storage_service.py for state management

## Phase 3.7: Integration & System Features

- [ ] T043 Integrate existing toaripi_slm.core modules with CLI commands
- [ ] T044 Implement checkpoint saving and recovery for training interruption
- [ ] T045 Add structured logging with educational context to all CLI operations
- [ ] T046 Cross-platform path handling and configuration management
- [ ] T047 System requirements validation (RAM, disk space, dependencies)

## Phase 3.8: Educational Validation & Cultural Appropriateness

- [ ] T048 [P] Cultural content validation rules in src/toaripi_slm/validation/cultural.py
- [ ] T049 [P] Age-appropriate content scoring in src/toaripi_slm/validation/age_groups.py
- [ ] T050 [P] Educational value assessment in src/toaripi_slm/validation/educational.py
- [ ] T051 Content validation pipeline integration with CLI commands

## Phase 3.9: Polish & Performance

- [ ] T052 [P] Unit tests for all data models in tests/unit/test_models.py
- [ ] T053 [P] Unit tests for validation services in tests/unit/test_validation.py
- [ ] T054 [P] Performance testing for CLI responsiveness (<2s) in tests/performance/test_cli_speed.py
- [ ] T055 [P] Memory usage testing for edge deployment constraints in tests/performance/test_memory.py
- [ ] T056 [P] CLI help documentation and examples update
- [ ] T057 Constitutional compliance audit of all CLI features
- [ ] T058 End-to-end quickstart validation testing

## Dependencies

### Critical TDD Dependencies
- **Phase 3.2 → Phase 3.3**: Data models before contract tests
- **Phase 3.3 → Phase 3.5**: All contract tests MUST FAIL before implementation starts
- **Phase 3.4 → Phase 3.6**: Integration tests before service layer

### Implementation Dependencies  
- **T030 (main CLI)** blocks all command implementations (T031-T034)
- **T038-T042 (services)** must complete before T043 (integration)
- **T041 (validation service)** blocks T048-T051 (educational validation)
- **T036 (UI components)** blocks T035 (interactive mode)

### Educational Dependencies
- **T048-T050 (validation rules)** before T051 (pipeline integration)
- **T041 (validation service)** must integrate with all CLI commands
- **T057 (constitutional audit)** requires all features complete

## Parallel Execution Examples

### Phase 3.2 - Data Models (All Parallel)
```bash
# All data models can be implemented simultaneously:
Task: "TrainingSession pydantic model in src/toaripi_slm/models/session.py"
Task: "Dataset pydantic model in src/toaripi_slm/models/dataset.py"  
Task: "ModelConfiguration pydantic model in src/toaripi_slm/models/config.py"
Task: "ModelCheckpoint pydantic model in src/toaripi_slm/models/checkpoint.py"
Task: "TrainingLog pydantic model in src/toaripi_slm/models/log.py"
```

### Phase 3.3 - Contract Tests (All Parallel)  
```bash
# All CLI contract tests can be written simultaneously:
Task: "Contract test for toaripi data prepare in tests/cli/test_data_prepare.py"
Task: "Contract test for toaripi train start in tests/cli/test_train_start.py"
Task: "Contract test for toaripi model test in tests/cli/test_model_test.py"
Task: "Contract test for toaripi interactive in tests/cli/test_interactive.py"
```

### Phase 3.4 - Integration Tests (All Parallel)
```bash  
# All integration workflow tests can be written simultaneously:
Task: "Integration test: Complete data preparation workflow"
Task: "Integration test: Full training session lifecycle"
Task: "Integration test: Model testing and export workflow"
Task: "Integration test: Interactive mode user journey"
```

### Phase 3.5 - Command Groups (Partially Parallel)
```bash
# Command groups can be implemented in parallel after main CLI:
Task: "Data command group implementation in src/toaripi_slm/cli/data_commands.py"
Task: "Training command group implementation in src/toaripi_slm/cli/train_commands.py"
Task: "Model command group implementation in src/toaripi_slm/cli/model_commands.py"
```

## Task Generation Rules Applied

1. **From CLI Contracts**: 12 CLI commands → 12 contract tests (T013-T024) [P]
2. **From Data Model**: 5 entities + 2 supporting modules → 7 model tasks (T006-T012) [P]
3. **From User Stories**: 5 quickstart workflows → 5 integration tests (T025-T029) [P]
4. **From Research**: Framework decisions → Setup tasks (T001-T005)
5. **Educational Requirements**: Constitutional compliance → Validation tasks (T048-T051)

## Validation Checklist

- [x] All CLI contracts have corresponding contract tests (T013-T024)
- [x] All data entities have model implementation tasks (T006-T012)
- [x] All contract tests come before implementation (Phase 3.3 → 3.5)
- [x] Parallel tasks are truly independent (different files, no shared state)
- [x] Each task specifies exact file path
- [x] Educational and constitutional requirements integrated
- [x] TDD workflow enforced (tests must fail before implementation)
- [x] Integration with existing toaripi_slm modules planned (T043)

## Notes

- **[P] Tasks**: Different files, no dependencies - can run simultaneously
- **Educational Focus**: All tasks include cultural appropriateness and age-group validation
- **Constitutional Compliance**: T057 ensures all CLI features meet constitutional requirements
- **Edge Deployment**: Performance testing (T054-T055) validates <2s response and memory constraints
- **TDD Enforcement**: Phase 3.3 contract tests MUST FAIL before Phase 3.5 implementation begins
- **Integration Ready**: T043 connects new CLI to existing ML pipeline components

## Success Criteria

1. All 58 tasks can be executed in dependency order
2. TDD workflow produces failing tests before implementation
3. CLI tool meets all functional requirements from specification
4. Educational validation integrated into all content generation paths
5. Performance targets achieved for edge deployment
6. Constitutional compliance verified for all features
7. Interactive mode provides guided experience for non-technical users