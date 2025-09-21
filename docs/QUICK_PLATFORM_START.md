# 🔌 Toaripi SLM CLI – Quick Platform Invocation Guide

A minimal, copy‑paste friendly reference for launching and using the Toaripi CLI on Linux / WSL and Windows.

> Focus: create env → install → train (register version) → interact → export → (optional) push.

---
## 1. Prerequisites

| Platform | Requirements |
|----------|-------------|
| Linux / WSL | Python 3.10+, build tools (`gcc`), git |
| Windows | Python 3.10+ (Add to PATH), PowerShell or CMD, git |

Check versions:
```bash
python --version
pip --version
```

---
## 2. Clone Repository
```bash
git clone https://github.com/cknzraposo/toaripi-slm.git
cd toaripi-slm
```

---
## 3. Create & Activate Virtual Environment

### Linux / WSL
```bash
python3 -m venv toaripi_env
source toaripi_env/bin/activate
```

### Windows (PowerShell)
```powershell
python -m venv toaripi_env
./toaripi_env/Scripts/Activate.ps1
```

### Windows (CMD)
```cmd
python -m venv toaripi_env
call toaripi_env\Scripts\activate.bat
```

---
## 4. Install Package & Dependencies
```bash
pip install --upgrade pip
pip install -e .
```
(Optional dev extras):
```bash
pip install -r requirements-dev.txt
```

Verify CLI:
```bash
toaripi --help
```

If command not found (Windows):
```bash
python -m toaripi_slm.cli --help
```

---
## 5. Health & Environment Checks
```bash
toaripi status --detailed
# Optional deeper diagnostics
toaripi doctor --detailed
```

---
## 6. Train (Registers New Version)
```bash
toaripi train --interactive
```
Expected output tail:
```
📦 Registered new model version: v0.0.X
```
List versions:
```bash
toaripi models list
```
Inspect one:
```bash
toaripi models info v0.0.X
```

---
## 7. Interact (Educational Content & Chat)
Latest version automatically:
```bash
toaripi interact
```
Pinned version:
```bash
toaripi interact --version v0.0.X --content-type story
```
Switch modes inside session:
```
/type chat
/type vocabulary
/weights   # toggle weight colouring
/align     # toggle token alignment padding
/legend    # show colour legend again
/save      # persist session JSON under chat_sessions/
```

---
## 8. Export (Checksums + Quant Placeholder)
```bash
toaripi export --version v0.0.X --quant q4_k_m
```
Artifacts:
```
models/export/v0.0.X/
  export_manifest.json  # includes checksums + quantization_placeholder
  README.md             # model card (if not --no-card)
  gguf/QUANTIZATION_PLACEHOLDER.json
```

---
## 9. Push to Hugging Face (Optional)
Set token (Linux / WSL):
```bash
export HF_TOKEN=hf_xxx
```
Windows PowerShell:
```powershell
$env:HF_TOKEN="hf_xxx"
```
Push:
```bash
toaripi export --version v0.0.X --push --repo-id yourname/toaripi-educational-v0-0-X
```

---
## 10. Sessions Management
```bash
toaripi sessions list
# Show summary
toaripi sessions show session_20250920_181103.json
# Replay first 10 exchanges
toaripi sessions replay session_20250920_181103.json --limit 10
```

---
## 11. Windows vs WSL Notes
| Topic | WSL | Native Windows |
|-------|-----|----------------|
| Virtual Env Activation | `source venv/bin/activate` | `Activate.ps1` / `.bat` |
| GPU / CUDA | Prefer WSL for better CUDA alignment | Native requires proper CUDA toolkit |
| File Paths | Linux style `/home/...` | Use escaped paths `C:\path\to` |
| Line Endings | LF | Ensure Git `core.autocrlf` doesn’t break scripts |

---
## 12. Common Issues Quick Fix
| Symptom | Fix |
|---------|-----|
| `toaripi: command not found` | Re-activate venv or run `python -m toaripi_slm.cli` |
| Missing heavy ML deps | `pip install -r requirements.txt` (or skip if just exploring CLI) |
| No versions listed | Run `toaripi train` to register first model |
| Export missing checksums | Ensure model dir has `config.json` / `tokenizer.json` |
| Push fails (auth) | Set `HF_TOKEN` or pass `--token` |

---
## 13. Clean Up
Deactivate:
```bash
deactivate
```
Remove env:
```bash
rm -rf toaripi_env  # Linux/WSL
rd /s /q toaripi_env  # Windows CMD
```

---
## 14. Fast Reference (Copy/Paste Linux / WSL)
```bash
python3 -m venv toaripi_env \
  && source toaripi_env/bin/activate \
  && pip install -e . \
  && toaripi train --interactive \
  && toaripi interact
```

## 15. Fast Reference (Copy/Paste Windows PowerShell)
```powershell
python -m venv toaripi_env; \
./toaripi_env/Scripts/Activate.ps1; \
pip install -e .; \
toaripi train --interactive; \
toaripi interact
```

---
## 16. Next Steps
* Implement real GGUF quantization (`scripts/quantize.py` future)
* Add provenance hashing for training configs + datasets
* Extend educational content templates (quizzes, cloze)

Happy building! 🌺
