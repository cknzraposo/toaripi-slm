"""Core modular components for the Toaripi CLI interactive experience.

This package isolates concerns so the main `interact` command remains
lean and focused on orchestration:

Components:
    token_weights:  Token weight data structures + simulation provider
    display:        Rich-powered bilingual + weight visualization
    generator:      Lightweight wrapper around a HF model directory
    session:        Interactive session state + persistence helpers

All modules are intentionally dependency-light (only standard lib + rich + transformers/torch when loading models).
"""

from .token_weights import TokenWeight, TokenWeightProvider, SimulatedTokenWeightProvider
from .display import BilingualDisplay
from .generator import ToaripiGenerator
from .session import InteractiveSession

__all__ = [
    "TokenWeight",
    "TokenWeightProvider",
    "SimulatedTokenWeightProvider",
    "BilingualDisplay",
    "ToaripiGenerator",
    "InteractiveSession",
]
