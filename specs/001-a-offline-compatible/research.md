# Research: Interactive CLI Management Tool

## Overview

Research findings for implementing an elegant, minimal-dependency Python CLI tool for Toaripi SLM management with sleek UI and offline-first operation.

## Technology Decisions

### CLI Framework Choice

**Decision**: Use `click` for command-line interface framework

**Rationale**: 
- Minimal dependency with excellent developer experience
- Supports nested commands for hierarchical menu structure (data → prepare, train → start/stop/status)
- Built-in parameter validation and help generation
- Widely adopted with stable API
- Aligns with "elegance over complexity" requirement

**Alternatives Considered**:
- `argparse` (stdlib): Too verbose for complex interactive CLI
- `typer`: Heavier dependency, less mature than click
- `fire`: Less control over UX, not suitable for educational users

### Terminal UI Framework

**Decision**: Use `rich` for beautiful terminal output and progress visualization

**Rationale**:
- Modern, sleek terminal UI with minimal additional dependencies
- Built-in progress bars, tables, syntax highlighting, and panels
- Excellent for real-time training progress display
- Educational-friendly with clear visual hierarchy
- Cross-platform terminal compatibility

**Alternatives Considered**:
- `blessed`/`curses`: More complex, platform compatibility issues
- Plain text output: Inadequate UX for non-technical users
- `textual`: Too heavy for CLI tool, better for full TUI apps

### Data Validation

**Decision**: Use `pydantic` for configuration and data validation

**Rationale**:
- Type-safe configuration with clear error messages
- Excellent for validating training parameters and data schemas
- Educational error messages align with constitutional requirements
- Minimal runtime overhead for CLI operations

**Alternatives Considered**:
- `marshmallow`: More complex API, heavier dependency
- `dataclasses` + manual validation: Too verbose, error-prone
- `attrs`: Less validation features than pydantic

### Configuration Management

**Decision**: Use `PyYAML` for configuration files with `pathlib` for cross-platform paths

**Rationale**:
- Human-readable configuration format suitable for educators
- Supports comments for educational documentation
- Standard library `pathlib` ensures cross-platform compatibility
- Minimal dependencies for configuration management

**Alternatives Considered**:
- `TOML`: Less familiar to non-technical users
- JSON: No comments support, less human-friendly
- INI files: Limited nesting capabilities

## Architecture Patterns

### Command Structure

**Decision**: Hierarchical command groups with interactive mode

**Rationale**:
- Matches user mental model: data operations, training operations, model operations
- Supports both direct commands (`toaripi data prepare input.csv`) and interactive menu
- Progressive disclosure: simple commands for experts, guided menus for beginners

**Structure**:
```
toaripi
├── data
│   ├── prepare
│   ├── validate
│   └── preview
├── train
│   ├── start
│   ├── stop
│   ├── resume
│   └── status
├── model
│   ├── list
│   ├── test
│   └── export
└── interactive  # Menu-driven mode
```

### State Management

**Decision**: File-based state with JSON metadata and YAML configuration

**Rationale**:
- Simple, debuggable state storage
- No database dependencies for offline operation
- Human-readable for troubleshooting
- Atomic operations for training checkpoint safety

**State Structure**:
```
~/.toaripi/
├── config.yaml          # User preferences
├── sessions.json         # Training session metadata
├── models.json          # Model registry
└── data/
    ├── prepared/        # Processed training data
    ├── checkpoints/     # Training checkpoints
    └── logs/           # Training logs
```

### Error Handling Strategy

**Decision**: Layered error handling with educational messages

**Rationale**:
- Constitutional requirement for educational error messages
- Defensive programming with graceful degradation
- Clear recovery instructions for non-technical users

**Pattern**:
```python
try:
    operation()
except SpecificError as e:
    rich.print("[red]Error:[/red] What went wrong")
    rich.print("[yellow]Cause:[/yellow] Why it happened") 
    rich.print("[green]Solution:[/green] How to fix it")
    return False
```

## Integration Patterns

### Existing Codebase Integration

**Decision**: Extend existing `src/toaripi_slm/` modules with CLI-specific components

**Rationale**:
- Reuse existing defensive programming patterns from core modules
- Maintain constitutional compliance with existing validation
- Avoid code duplication while adding CLI interface

**Integration Points**:
- `src/toaripi_slm/core/trainer.py` → CLI training commands
- `src/toaripi_slm/data/preprocessing.py` → CLI data commands  
- `src/toaripi_slm/inference/` → CLI model testing commands

### Progress Monitoring

**Decision**: Rich progress bars with structured logging

**Rationale**:
- Real-time feedback required for long training operations
- Educational UX with clear time estimates and metrics
- Structured logs for debugging and constitutional compliance

**Implementation**:
- Training: Epoch progress, loss metrics, time estimates
- Data processing: File validation progress, statistics
- Model operations: Export progress, validation results

## Performance Considerations

### Memory Efficiency

**Decision**: Streaming data processing with lazy loading

**Rationale**:
- Target 8GB RAM systems (constitutional requirement)
- Large training datasets require streaming processing
- Model loading should be lazy and unloadable

### Startup Time

**Decision**: Lazy imports and command-specific loading

**Rationale**:
- CLI responsiveness critical for interactive UX
- Import heavy ML libraries only when needed
- Command help should be instant

## Security & Validation

### Input Validation

**Decision**: Multi-layer validation with pydantic schemas

**Rationale**:
- Constitutional defensive programming requirement
- Educational error messages for invalid configurations
- Cultural appropriateness validation for content

**Validation Layers**:
1. CLI parameter validation (click)
2. Configuration schema validation (pydantic)
3. Data content validation (existing modules)
4. Cultural appropriateness checks (existing validation)

### File System Safety

**Decision**: Atomic operations with backup/rollback capabilities

**Rationale**:
- Prevent data loss during training interruptions
- Constitutional reliability requirement
- Educational context requires exceptional safety

## Testing Strategy

### Contract Testing

**Decision**: pytest with click.testing for CLI contract validation

**Rationale**:
- Test-first quality gates (constitutional requirement)
- CLI commands have clear input/output contracts
- Integration with existing pytest infrastructure

### User Scenario Testing

**Decision**: Pytest scenarios matching feature specification acceptance criteria

**Rationale**:
- Direct validation of user stories from specification
- Educational workflow validation
- End-to-end testing of critical user paths

## Deployment Considerations

### Cross-Platform Compatibility

**Decision**: Python 3.10+ with standard library emphasis

**Rationale**:
- Consistent behavior across Windows, macOS, Linux
- Minimal external dependencies for easier deployment
- Educational users may have diverse computing environments

### Offline Operation

**Decision**: Bundle all required resources with graceful degradation

**Rationale**:
- Constitutional requirement for offline operation
- Remote Papua New Guinea deployment context
- No network dependencies for core functionality

## Implementation Priority

1. **Core CLI structure** (click commands, rich UI)
2. **Data preparation commands** (highest user value)
3. **Training control commands** (core functionality)
4. **Model management commands** (operational necessity)
5. **Interactive mode** (educational UX enhancement)

## Summary

Research validates the elegant, minimal-dependency approach using click + rich + pydantic for a sleek, educational-friendly CLI tool that maintains constitutional compliance while providing excellent UX for Toaripi educators.