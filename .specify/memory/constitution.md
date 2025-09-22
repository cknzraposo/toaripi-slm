<!--
Sync Impact Report:
- Version change: Initial version → 1.0.0
- Added sections: All core principles and governance sections (initial creation)
- Templates requiring updates: 
  ✅ .specify/templates/plan-template.md (constitution check section aligns)
  ✅ .specify/templates/spec-template.md (requirements alignment maintained)
  ✅ .specify/templates/tasks-template.md (task categorization reflects principles)
  ✅ .github/copilot-instructions.md (guidance references maintained)
- Follow-up TODOs: None (all placeholders filled)
-->

# Toaripi SLM Constitution

## Core Principles

### I. Educational-First Development

Every feature MUST serve the educational mission of Toaripi language preservation and primary school learning. Features are evaluated against educational value before technical sophistication. All content generation MUST be age-appropriate (5-12 years), culturally sensitive to Papua New Guinea context, and aligned with primary school curriculum needs. No general-purpose chatbot features or theological content generation permitted.

**Rationale**: The project exists to empower Toaripi language preservation through education, not as a general AI platform. This constraint guides all technical decisions and prevents scope creep.

### II. Defensive Programming (NON-NEGOTIABLE)

All code MUST implement comprehensive input validation, error handling, and graceful degradation. Every public function MUST validate inputs, handle edge cases, and provide meaningful error messages. Cultural appropriateness checks are mandatory for all content generation paths. No assumptions about data quality, network availability, or external dependencies.

**Rationale**: Educational tools require exceptional reliability since they serve communities with limited technical support. Defensive programming prevents data loss and ensures consistent user experience.

### III. Test-First Quality Gates

TDD mandatory: Tests written → Stakeholder approved → Tests fail → Implementation begins. All public APIs require contract tests. All educational content generation requires validation tests. All data processing pipelines require integration tests. Test coverage minimum 85% for core modules, 70% for supporting modules.

**Rationale**: Educational software must be exceptionally reliable. Test-first ensures requirements are clear before implementation and prevents regressions that could disrupt learning.

### IV. Performance for Edge Deployment

All models MUST run on 8GB RAM minimum, prefer 4GB. Inference latency MUST be <2 seconds for content generation on CPU-only devices. Model size MUST be <5GB when quantized for Raspberry Pi deployment. Support offline operation without internet connectivity. Batch processing preferred over real-time for resource-intensive operations.

**Rationale**: Target deployment includes remote Papua New Guinea communities with limited computational resources and unreliable internet connectivity.

### V. Consistent Educational UX

All interfaces (CLI, Web UI, API) MUST present consistent terminology and workflows. Error messages MUST be educational ("what went wrong, why, how to fix"). Progress indicators required for operations >3 seconds. All outputs MUST be culturally appropriate and age-group validated. Support both technical and non-technical users through layered complexity.

**Rationale**: Teachers and educators have varying technical backgrounds. Consistent, educational UX reduces training overhead and increases adoption.

## Educational Content Standards

Educational content generation MUST follow primary school pedagogical principles. All generated stories, vocabulary, and exercises require cultural sensitivity validation. Content MUST avoid violence, adult themes, religious doctrine, or inappropriate cultural references. Age-appropriate language complexity enforced through automated scoring. Generated content MUST include learning objectives and assessment criteria.

**Content Validation Pipeline**: Input sanitization → Cultural appropriateness check → Age-group validation → Educational value scoring → Output approval. Content scoring algorithm MUST be transparent and auditable by educators.

## Technical Quality Standards

### Code Quality Requirements

- Python 3.10+ with type hints mandatory
- Black code formatting and flake8 linting enforced
- Docstrings required for all public functions (Google style)
- Configuration via YAML/TOML only, no hardcoded values
- Logging structured (JSON) with appropriate levels
- Resource cleanup mandatory (context managers, try/finally)

### Dependency Management

- Pin exact versions in requirements.txt
- Minimize dependencies for edge deployment compatibility
- Document rationale for each major dependency
- No dependencies with restrictive licenses
- Regular security audits of dependency tree

### Data Processing Standards

- CSV/TSV formats preferred for transparency
- UTF-8 encoding mandatory with BOM handling
- Defensive encoding detection and conversion
- Data validation before and after processing
- Audit trails for all data transformations

## Development Workflow

### Code Review Gates

- All PRs require constitution compliance check
- Educational appropriateness review for content-related changes
- Performance impact assessment for model/inference changes
- Documentation updates for API changes
- Test coverage verification before merge

### Quality Assurance Process

- Unit tests run on every commit
- Integration tests on PR submission
- Educational content validation on content changes
- Performance regression testing on model updates
- Manual testing following documented scenarios

### Release Management

- Semantic versioning (MAJOR.MINOR.PATCH)
- MAJOR: Educational approach changes, breaking API changes
- MINOR: New content types, new features, significant UX improvements
- PATCH: Bug fixes, content improvements, minor optimizations
- All releases require educational stakeholder approval

## Governance

This constitution supersedes all other development practices and technical decisions. Amendments require documentation of rationale, stakeholder approval from educational partners, and migration plan for existing code. All pull requests and code reviews MUST verify constitutional compliance before approval.

Complexity that violates these principles MUST be justified with specific educational needs that simpler approaches cannot address. Technical debt that compromises educational mission or deployment constraints is not permitted.

For runtime development guidance, refer to `.github/copilot-instructions.md` for AI assistance patterns and `README.md` for setup procedures.

**Version**: 1.0.0 | **Ratified**: 2025-09-22 | **Last Amended**: 2025-09-22