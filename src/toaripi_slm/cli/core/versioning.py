"""Model version registry helpers.

Centralizes logic for loading and resolving versioned model directories
so multiple commands (interact/export/etc.) share consistent behavior.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
import json

REGISTRY_PATH = Path("./models/hf/registry.json")


def load_registry() -> Dict[str, Any]:
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {"models": []}
    return {"models": []}


def list_models() -> List[Dict[str, Any]]:
    reg = load_registry()
    return reg.get("models", [])


def resolve_version_dir(version: Optional[str]) -> Optional[Path]:
    models = list_models()
    if not models:
        return None
    if version:
        for m in models:
            if m.get("version") == version:
                return Path(m.get("path"))
        return None
    # latest by created_at chronological order
    ordered = sorted(models, key=lambda m: m.get("created_at", ""))
    return Path(ordered[-1].get("path")) if ordered else None


def latest_version() -> Optional[str]:
    models = list_models()
    if not models:
        return None
    ordered = sorted(models, key=lambda m: m.get("created_at", ""))
    return ordered[-1].get("version") if ordered else None


__all__ = [
    "load_registry",
    "list_models",
    "resolve_version_dir",
    "latest_version",
]
