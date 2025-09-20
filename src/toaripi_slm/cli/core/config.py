"""Central CLI configuration defaults for generation/settings."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationDefaults:
    max_length: int = 200
    temperature: float = 0.7
    content_type: str = "story"


DEFAULTS = GenerationDefaults()

__all__ = ["GenerationDefaults", "DEFAULTS"]
