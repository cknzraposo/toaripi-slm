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

