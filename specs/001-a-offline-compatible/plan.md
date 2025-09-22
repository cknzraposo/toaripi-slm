
# Implementation Plan: Interactive CLI Management Tool for Toaripi SLM

**Branch**: `001-a-offline-compatible` | **Date**: September 22, 2025 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-a-offline-compatible/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary

Interactive CLI management tool for Toaripi Small Language Model providing offline-compatible, cross-platform capabilities for data preparation, training control, and model management. Targets teachers and educators with menu-driven navigation, real-time progress monitoring, and educational content validation. Technical approach emphasizes Python elegance with minimal dependencies, sleek UI design, and robust defensive programming for resource-constrained edge deployment.

## Technical Context

**Language/Version**: Python 3.10+ (aligns with project standards)
**Primary Dependencies**: click (CLI framework), rich (beautiful terminal UI), pydantic (data validation), transformers (core ML), minimal additional libraries for elegance
**Storage**: Local filesystem with JSON/YAML config files, CSV for training data, checkpoint management for model state
**Testing**: pytest with comprehensive contract testing for CLI commands and integration testing for training workflows  
**Target Platform**: Cross-platform (Windows, macOS, Linux) with focus on resource-constrained environments including Raspberry Pi
**Project Type**: single (CLI-focused with integrated ML pipeline)
**Performance Goals**: <2s response time for interactive commands, <5GB model size for edge deployment, efficient memory usage for 8GB RAM systems
**Constraints**: Offline operation mandatory, no internet dependency for core functions, culturally appropriate content generation, educational-first UX
**Scale/Scope**: Local single-user operation, support for multiple training sessions and model versions, designed for teachers/educators with varying technical backgrounds

User Input Integration: "the CLI use the best approach use Python with min libraries to meet requirements, elegance over complex, UI is sleek - use best judgement"

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Initial Check (Pre-Phase 0)

✅ **Educational-First Development**: CLI tool serves Toaripi language preservation mission by enabling teacher-friendly model management for educational content generation

✅ **Defensive Programming**: CLI commands require comprehensive input validation (data files, training parameters), graceful error handling for system constraints, meaningful error messages for educators

✅ **Test-First Quality Gates**: TDD approach planned with contract tests for CLI commands, integration tests for training workflows, educational content validation tests

✅ **Performance for Edge Deployment**: Target 8GB RAM systems, <2s CLI response times, <5GB model exports, offline-first design for resource-constrained environments

✅ **Consistent Educational UX**: Menu-driven navigation for non-technical users, educational error messages, progress indicators for training operations, culturally appropriate content validation

**Assessment**: PASS - All constitutional principles directly supported by CLI design approach

### Post-Design Check (After Phase 1)

✅ **Educational-First Development**: Data model includes educational content validation, age-group targeting, cultural appropriateness checks. All CLI commands prioritize teacher usability with clear educational outcomes.

✅ **Defensive Programming**: Comprehensive validation framework defined with multi-layer input validation (CLI → schema → content → cultural). Error handling provides educational context with clear recovery guidance.

✅ **Test-First Quality Gates**: Contract specifications ready for TDD implementation. CLI command contracts provide clear success/failure criteria. Integration test scenarios defined from user stories.

✅ **Performance for Edge Deployment**: State management designed for minimal memory footprint. Lazy loading patterns, streaming data processing, and quantized model export ensure edge compatibility.

✅ **Consistent Educational UX**: Interactive mode provides guided experience. Error messages follow educational format (What/Why/How). Progress indicators designed for training operations with clear time estimates.

**Assessment**: PASS - Design artifacts maintain constitutional compliance with enhanced educational focus

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure]
```

**Structure Decision**: [DEFAULT to Option 1 unless Technical Context indicates web/mobile app]

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/powershell/update-agent-context.ps1 -AgentType copilot`
     **IMPORTANT**: Execute it exactly as specified above. Do not add or remove any arguments.
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach

*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy for CLI Tool**:

- Load `.specify/templates/tasks-template.md` as base structure
- Generate CLI-specific tasks from Phase 1 design artifacts:
  - Each CLI command contract → contract test task [P]
  - Each data model entity → pydantic model creation task [P]
  - Each user scenario → integration test task
  - Interactive mode → guided UX implementation task
  - Cultural validation → educational content validation task

**CLI-Specific Ordering Strategy**:

- TDD order: Contract tests → Implementation → Integration tests
- Dependency order: Data models → CLI framework → Commands → Interactive mode
- Educational priority: Data validation → Training commands → Content generation → Export
- Mark [P] for parallel execution (independent CLI commands and data models)

**Educational Integration Tasks**:

- Constitutional compliance validation for each CLI command
- Cultural appropriateness testing for content generation
- Age-group validation for educational outputs
- Performance testing for edge deployment requirements
- User experience testing with educator personas

**Estimated Output**: 35-40 numbered, ordered tasks in tasks.md including:

- 12 contract test tasks (one per CLI command) [P]
- 8 data model implementation tasks [P]
- 10 CLI command implementation tasks
- 6 integration test tasks (user scenarios)
- 4 educational validation tasks
- 3 interactive mode tasks

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking

*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved (no unclear items in Technical Context)
- [x] Complexity deviations documented (none required)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
