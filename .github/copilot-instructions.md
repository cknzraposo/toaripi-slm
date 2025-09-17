# GitHub Copilot Instructions — Toaripi SLM (Educational Content Generator)
**Note:** This file provides context and instructions for GitHub Copilot to assist with code generation in this repository. It is not part of the user-facing documentation.
## 🧭 Project Context (what Copilot should know)

- **Mission:** Build a **small language model (SLM)** for **Toaripi (ISO 639‑3: `tqo`)** to generate **original educational content** (stories, vocabulary, Q&A, dialogues) for primary learners and teachers.
- **Approach:** Fine‑tune a compact open model (≈1–7B params) on **aligned English↔Toaripi Bible** data; run **online and offline** (Raspberry Pi / CPU‑only) using **quantization** and **llama.cpp**.
- **Users:** Teachers, Toaripi speakers, contributors (linguists, devs) with varying technical experience.
- **Non‑goals:** Theology tooling / doctrinal outputs; general-purpose chatbot. Focus is **education & language preservation**.

---

## 🧰 Tech Stack & Key Tools

- **Language:** Python 3.10+
- **Core libs:** `transformers`, `datasets`, `accelerate`, `peft` (LoRA), `sentencepiece`/tokenizers
- **Serving/UI:** `fastapi` + `uvicorn` (or `streamlit` for quick demo)
- **Edge inference:** `llama.cpp` (GGUF quantized weights)
- **Data formats:** CSV/TSV for parallel verses; optional USFM ingestion
- **Config:** YAML/TOML for data sources & training params

---

## 📁 Repository Shape (expected files and roles)