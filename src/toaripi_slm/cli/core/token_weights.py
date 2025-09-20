"""Token weight data structures and providers.

In the absence of real attention extraction (which would require
modifying generate() calls with output_attentions=True and handling
model-specific architectures), we provide a simulation layer that
produces stable yet varied weights for visualization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import random

@dataclass
class TokenWeight:
    token: str
    weight: float  # 0–1 range

    def color_style(self) -> str:
        """Return a Rich color style based on weight bucket."""
        w = self.weight
        if w >= 0.8:
            return "bold red"
        if w >= 0.6:
            return "bold orange3"
        if w >= 0.4:
            return "bold yellow"
        if w >= 0.2:
            return "green"
        return "dim cyan"


class TokenWeightProvider:
    """Abstract provider; future real attention implementation will subclass."""

    def weights_for(self, tokens: Iterable[str], *, generated: bool) -> List[TokenWeight]:  # pragma: no cover - interface
        raise NotImplementedError


class SimulatedTokenWeightProvider(TokenWeightProvider):
    """Heuristic + random weight generator.

    Heuristics:
        - Slash-delimited bilingual pairs get high weights
        - Apparent function words receive lower weights
        - Longer tokens / tokens with certain suffixes slightly boosted
    """

    _FUNCTION_WORDS = {"the", "is", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for"}

    def weights_for(self, tokens: Iterable[str], *, generated: bool) -> List[TokenWeight]:
        out: List[TokenWeight] = []
        for t in tokens:
            base: float
            tl = t.lower()
            if not t:
                base = 0.0
            elif "/" in t:
                base = random.uniform(0.8, 1.0)
            elif tl in self._FUNCTION_WORDS:
                base = random.uniform(0.05, 0.3)
            elif len(t) > 7 or tl.endswith("ing") or tl.endswith("ed"):
                base = random.uniform(0.5, 0.85)
            else:
                base = random.uniform(0.25, 0.7)

            # Slight bump for generated tokens to create visual distinction
            if generated:
                base = min(1.0, base + 0.05)

            out.append(TokenWeight(t, round(base, 4)))
        return out
