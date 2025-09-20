# Research: Offline-Capable Toaripi Educational SLM

## Decisions & Rationale

### Base Model Selection
- Decision: Mistral 7B Instruct (or equivalent permissive 7B instruct model)
- Rationale: Strong efficiency, broad tokenizer coverage, active ecosystem.
- Alternatives: Llama 3 8B (slightly larger, may exceed size goals); Phi-3 Mini (smaller but may underperform in generation richness).

### Adaptation Strategy
- Decision: LoRA fine-tuning (r=16, alpha=32, dropout=0.1) on instruct base.
- Rationale: Memory/compute efficiency, reproducibility, minimal risk of overfitting small corpus.
- Alternatives: Full fine-tune (resource heavy), QLoRA (possible but start simple). 

### Tokenization
- Decision: Reuse base tokenizer; evaluate fragmentation rate on Toaripi lexicon sample.
- Threshold: If >15% words fragment into >3 sub-tokens → Train custom SentencePiece unigram model (32k vocab) blending Toaripi + minimal English anchor tokens.

### Data Minimum
- Decision: 150 aligned pairs (raise above constitution 100).
- Rationale: Slightly improved lexical variety & stable gradients.
- Risk: Still low-resource; mitigate via conservative learning rate & early stopping.

### Safety Filtering
- Decision: Hybrid lexical list + severity scoring; threshold: block if >0 high-risk or >3 medium-risk hits.
- Alternatives: ML classifier (overkill initially) or pure regex (insufficient nuance).

### Deterministic Stable Mode
- Decision: Fixed seed + temperature=0.7 cap + top_p=0.9 + top_k=40 + disable sampling variance features.
- Rationale: High overlap while retaining minor stylistic naturalness.

### Latency Optimization
- Decision: Quantize to GGUF Q4_K_M for inference; restrict max tokens; early stop on sentence/item count.
- Target: p95 ≤10s (stories/Q&A/dialogue), ≤8s vocabulary on 8GB CPU.

### Evaluation Pack
- Decision: 12-item pack (3 per type) per release candidate.
- Rationale: Lightweight yet diverse snapshot.

### Logging & Metadata
- Decision: JSON Lines file `logs/generation.log` with: timestamp, request_id, content_type, tokens_in, tokens_out, latency_ms, safety_status, model_version, data_checksum.

### Reproducibility
- Decision: Hash of training data CSV + config YAML stored in model card metadata.

## Open Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Data sparsity | Poor generalization | Conservative LR, early stopping, evaluation pack review |
| Overfiltering safety | Reduced usefulness | Tune severity list; human feedback loop |
| Tokenizer fragmentation | Inefficient generation | Custom SP model if threshold exceeded |
| Latency spikes | Classroom disruption | Preload model, warm-up run, cap max tokens |

## References
- Hugging Face LoRA best practices (conceptual)
- Low-resource language model adaptation patterns

---

# Research: Web Interface for CSV Data Upload and Model Training

**Date**: 2025-09-20  
**Feature**: Web Interface Extension for Training Data Upload  
**Additional Context**: create a web interface that allows training the model by uploading new data in csv format it needs to be sleek and intuitive for users

## Web Interface Research Questions Resolved

### 1. Web Framework Selection for Educational AI Interface

**Decision**: FastAPI with Jinja2 Templates  
**Rationale**: 
- FastAPI provides automatic OpenAPI documentation generation critical for API contracts
- Native async support for handling file uploads and training orchestration
- Excellent integration with Pydantic for CSV schema validation
- Lightweight compared to Django, aligns with constitutional simplicity principle
- Strong Python ecosystem integration with existing toaripi-slm package

**Alternatives Considered**:
- Flask: Simpler but lacks automatic API documentation and async support
- Django: Too heavyweight for single-purpose educational interface
- Streamlit: Educational-friendly but limited customization for "sleek" UI requirement

### 2. File Upload and Processing Patterns

**Decision**: Chunked upload with immediate validation and background processing  
**Rationale**:
- Handles ≤50MB CSV files efficiently without blocking UI
- Immediate schema validation provides fast user feedback
- Background training initiation with progress streaming
- Aligns with offline-capable requirement (no external dependencies)

**Alternatives Considered**:
- Synchronous processing: Would block UI for large files
- External cloud processing: Violates offline-capable requirement
- Direct file system access: Lacks proper validation and error handling

### 3. UI Framework for Educational Users

**Decision**: HTML5 + CSS3 + Vanilla JavaScript with modern UI patterns  
**Rationale**:
- No external JavaScript dependencies maintains offline capability
- Drag-and-drop file interface meets "sleek and intuitive" requirement  
- Progressive enhancement ensures accessibility for diverse educational environments
- Lightweight approach aligns with constitutional simplicity

**Alternatives Considered**:
- React/Vue.js: Adds build complexity and external dependencies
- Bootstrap: Heavy framework for simple upload interface
- Pure server-side rendering: Less interactive for file upload experience

### 4. CSV Schema Validation Strategy

**Decision**: Pydantic models with educational content validation  
**Rationale**:
- Strong typing ensures data quality for training pipeline
- Built-in validation for constitutional safety requirements (content filtering)
- Clear error messages for teachers uploading data
- Integrates seamlessly with FastAPI automatic validation

**Alternatives Considered**:
- Pandas validation: Less structured error handling
- Manual validation: More error-prone and harder to maintain
- Cerberus: Additional dependency without significant benefits

### 5. Training Progress Communication

**Decision**: Server-Sent Events (SSE) for real-time progress updates  
**Rationale**:
- Browser-native technology, no WebSocket complexity
- Unidirectional communication perfect for training progress
- Works offline once connection established
- Graceful degradation to polling fallback

**Alternatives Considered**:
- WebSockets: Overkill for unidirectional progress updates
- Polling: Higher server load and less responsive
- Long polling: More complex error handling

### 6. Integration with Existing CLI Infrastructure

**Decision**: Web interface as FastAPI extension of existing CLI commands  
**Rationale**:
- Reuses existing training pipeline and validation logic
- Maintains consistency with CLI user experience
- Leverages existing configuration management
- Allows seamless transition between web and CLI workflows

**Alternatives Considered**:
- Separate training implementation: Code duplication and maintenance burden
- CLI wrapper: Adds unnecessary complexity and error points
- Complete rewrite: Violates constitutional principle of building on existing foundation

## Web Interface Technology Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| Backend API | FastAPI | Async, automatic docs, Pydantic integration |
| Frontend | HTML5 + CSS3 + Vanilla JS | Offline-capable, lightweight, accessible |
| File Processing | Python (pandas, csv) | Existing ecosystem, educational data patterns |
| Validation | Pydantic | Type safety, clear errors, FastAPI integration |
| Progress Updates | Server-Sent Events | Real-time, browser-native, offline-friendly |
| Content Safety | Rule-based + Classification | Multi-layer, auditable, teacher-controlled |
| Configuration | YAML (existing pattern) | Constitutional compliance, reproducibility |
| Testing | pytest + httpx | Existing test framework, API contract testing |

## Web Interface Implementation Patterns

### 1. Upload Workflow Pattern
```
1. User drags CSV file to upload area
2. Client validates file size and type
3. Server receives and validates CSV schema
4. Content safety screening on upload
5. Preview of processed data shown to user
6. User confirms training initiation
7. Background training with progress updates
```

### 2. Error Handling Pattern
```
- Validation errors: Immediate UI feedback with specific field issues
- Content safety flags: Warning dialog with alternatives
- Training errors: Graceful degradation with retry options
- System errors: Helpful messages with troubleshooting steps
```

### 3. Educational UX Pattern
```
- Visual progress indicators for all long-running operations
- Clear terminology avoiding technical jargon
- Contextual help for CSV format requirements
- Preview and confirmation steps for safety
- Audit trail for training decisions
```

## Web Interface Security and Safety

### Content Validation Pipeline
1. **File Security**: Validate CSV format, reject executable files
2. **Schema Validation**: Enforce required columns (English, Toaripi)
3. **Content Screening**: Automated filtering for inappropriate content
4. **Length Validation**: Ensure minimum training data requirements
5. **Character Set Validation**: Proper Unicode handling for Toaripi text

### Data Privacy
- Local file processing only (no external uploads)
- Temporary file cleanup after processing
- No persistent storage of user uploaded content beyond training data
- Audit logs for training decisions only

---
*Research complete - both core SLM and web interface technical unknowns resolved*

