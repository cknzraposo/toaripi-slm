"""Model export + Hugging Face Hub helper utilities.

This module centralizes logic for:
 1. Preparing an export directory (copying or referencing HF model)
 2. Generating a lightweight model card with educational + version metadata
 3. (Optional) Pushing artifacts to the Hugging Face Hub

Quantization / GGUF conversion is still a placeholder; the exporter records
intent and metadata for reproducibility.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, UTC
from typing import Optional, Dict, Any, List
import json
import shutil
import hashlib

from rich.console import Console

console = Console()


MODEL_CARD_TEMPLATE = """---
language: [toaripi, en]
license: mit
tags:
  - toaripi
  - education
  - low-resource
  - small-language-model
  - offline
model_version: {version}
base_model: {base_model}
quantization: {quantization}
---

# Toaripi Educational SLM {version}

Fine-tuned small language model specialized for generating age-appropriate educational content in the Toaripi language (Papua New Guinea) including stories, vocabulary items, dialogues, and comprehension questions.

## Intended Use
Offline / edge deployment for classroom and language preservation contexts. Not a general chatbot; avoids theological, adult, or violent content.

## Training Metadata
- Base model: {base_model}
- Created at: {created_at}
- Checkpoint source: {checkpoint_dir}

## Educational Focus
Outputs emphasize simple sentence structure, cultural relevance, and vocabulary reinforcement.

## Safety
Content filters applied during generation to reduce disallowed themes (see project docs).

## Version Registry
This model was registered via the local registry.json ensuring reproducible versioning.

## Quantization
If quantized, GGUF artifacts accompany this card for CPU-friendly inference.

## Limitations
May hallucinate or produce imperfect Toaripi morphology; human review advised for classroom distribution.
"""


def generate_model_card(metadata: Dict[str, Any], *, quantization: str) -> str:
    return MODEL_CARD_TEMPLATE.format(
        version=metadata.get("version", "unknown"),
        base_model=metadata.get("base_model", "unknown"),
        created_at=metadata.get("created_at", "unknown"),
        checkpoint_dir=metadata.get("checkpoint_dir", ""),
        quantization=quantization,
    )


def copy_model_dir(source: Path, dest: Path):
    if dest.exists():
        # Avoid overwriting; user can remove manually
        return
    shutil.copytree(source, dest)


def _hash_file(path: Path, algo: str = "sha256", chunk: int = 65536) -> str:
    h = hashlib.new(algo)
    try:
        with open(path, "rb") as f:  # pragma: no cover - straightforward
            while True:
                data = f.read(chunk)
                if not data:
                    break
                h.update(data)
        return f"{algo}:{h.hexdigest()}"
    except FileNotFoundError:
        return f"{algo}:missing"


def _collect_checksum_targets(model_dir: Path) -> List[Path]:
    patterns = [
        "config.json",
        "tokenizer.json",
        "tokenizer.model",
        "pytorch_model.bin",
        "generation_config.json",
    ]
    out: List[Path] = []
    for p in patterns:
        candidate = model_dir / p
        if candidate.exists():
            out.append(candidate)
    # Include any LoRA adapter weights if present
    for adapter in model_dir.glob("*adapter_model.bin"):
        out.append(adapter)
    return out


def _quantize_placeholder(source_dir: Path, export_dir: Path, quantization: str) -> Optional[Dict[str, Any]]:
    """Create a placeholder metadata file for a future GGUF quantization.

    This does NOT perform real quantization; it records intent so later a
    conversion script can fill in real artifact paths.
    """
    gguf_dir = export_dir / "gguf"
    gguf_dir.mkdir(exist_ok=True)
    meta = {
        "quantization": quantization,
        "status": "placeholder",
        "source": str(source_dir),
        "message": "Run scripts/quantize.py to produce real GGUF artifacts.",
    }
    with open(gguf_dir / "QUANTIZATION_PLACEHOLDER.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def prepare_export(
    version_metadata: Dict[str, Any],
    *,
    export_root: Path,
    include_card: bool = True,
    quantization: str = "q4_k_m",
    add_quant_placeholder: bool = True,
) -> Path:
    """Create export directory with manifest + optional model card.

    Adds SHA256 checksums for core model files and (optionally) a quantization
    placeholder record.
    """
    version = version_metadata.get("version") or "unknown"
    source_path = Path(version_metadata.get("path", ""))
    target_dir = export_root / version
    target_dir.mkdir(parents=True, exist_ok=True)

    checksum_targets = _collect_checksum_targets(source_path)
    checksums = {p.name: _hash_file(p) for p in checksum_targets}

    quant_meta: Optional[Dict[str, Any]] = None
    if add_quant_placeholder:
        quant_meta = _quantize_placeholder(source_path, target_dir, quantization)

    manifest = {
        "version": version,
        "export_time": datetime.now(UTC).isoformat(),
        "source_path": str(source_path),
        "base_model": version_metadata.get("base_model"),
        "quantization": quantization,
        "status": "prepared",
        "checksums": checksums,
        "quantization_placeholder": quant_meta,
    }
    with open(target_dir / "export_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if include_card:
        # Augment metadata passed to card with checksum summary
        meta_for_card = dict(version_metadata)
        meta_for_card["checksum_count"] = len(checksums)
        card = generate_model_card(meta_for_card, quantization=quantization)
        with open(target_dir / "README.md", "w") as f:
            f.write(card)

    return target_dir


def push_to_hub(
    export_dir: Path,
    *,
    repo_id: str,
    private: bool,
    token: Optional[str] = None,
    create_card: bool = True,
) -> bool:
    """Push exported model to Hugging Face Hub (simplified stub).

    Real implementation would:
      - from huggingface_hub import HfApi, create_repo, upload_folder
      - authenticate via token or env HF_TOKEN
    """
    try:
        try:
            from huggingface_hub import HfApi  # type: ignore
        except ImportError:
            console.print("⚠️  huggingface_hub not installed. Install to enable push.")
            return False

        api = HfApi()
        if token:
            # validate token (no-op if invalid; rely on exceptions)
            pass
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True, token=token)
        api.upload_folder(
            folder_path=str(export_dir),
            repo_id=repo_id,
            commit_message="Add Toaripi model export",
            token=token,
        )
        console.print(f"✅ Pushed export to https://huggingface.co/{repo_id}")
        return True
    except Exception as e:  # pragma: no cover
        console.print(f"❌ Push failed: {e}")
        return False


__all__ = [
    "prepare_export",
    "push_to_hub",
    "generate_model_card",
]
