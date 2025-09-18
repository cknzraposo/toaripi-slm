# Quickstart: Toaripi Educational SLM

## 1. Prerequisites
- Python 3.11
- 8GB RAM CPU environment (offline capable)

## 2. Install Dependencies
Use requirements.txt (already present). Ensure virtual environment activated.

## 3. Prepare Data
- Place aligned CSV at `data/parallel.csv` with columns: english,toaripi
- Must contain ≥150 rows; run validation script (to be implemented) which outputs checksum.

## 4. Configuration
- Copy a base training YAML from `configs/training/base_config.yaml`
- Set model_name, learning_rate, lora flags, epochs

## 5. Preprocess
Run preprocessing utility (to be added) to split train/val and produce tokenization coverage report.

## 6. Fine-Tune
Invoke training script with config + data path; outputs HF adapter + merged model under `models/hf/` with metadata JSON.

## 7. Quantize
Convert model to GGUF (Q4_K_M) into `models/gguf/` for offline inference.

## 8. Launch Inference API
Start FastAPI app (to be implemented) exposing /generate, /health, /evaluation-pack.

## 9. Generate Content
Send POST /generate with JSON: topic, content_type, target_length, mode, include_english_support.

## 10. Evaluation Pack
POST /evaluation-pack to produce 12-sample pack; review manually.

## 11. Logs & Metadata
Inspect `logs/generation.log` for JSON lines containing generation metadata.

## 12. Stable Mode
Repeat a request with mode=stable to verify ≥90% token overlap.

## 13. Safety Verification
Submit prompts near restricted boundaries to ensure block behavior (no theological, violent, adult content allowed).

## 14. Cleanup
Purge old logs (>30 days) and stale cache entries automatically handled by maintenance job (to be implemented).
