"""Rich-powered bilingual + token weight display layer."""
from __future__ import annotations

from typing import List, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text

from .token_weights import TokenWeight, TokenWeightProvider


class BilingualDisplay:
    """Handles side-by-side rendering + toggles for weights and alignment."""

    def __init__(self, console: Console | None = None, provider: TokenWeightProvider | None = None):
        from rich.console import Console as _Console  # local import for optionality
        self.console = console or _Console()
        self._provider = provider
        self.show_weights = True
        self.align_tokens = False

    # --- configuration -------------------------------------------------
    def set_provider(self, provider: TokenWeightProvider):
        self._provider = provider

    def toggle_weights(self):
        self.show_weights = not self.show_weights
        state = "ON" if self.show_weights else "OFF"
        self.console.print(f"💡 Token weights display: [bold]{state}[/bold]")

    def toggle_alignment(self):
        self.align_tokens = not self.align_tokens
        state = "ON" if self.align_tokens else "OFF"
        self.console.print(f"🔗 Token alignment: [bold]{state}[/bold]")

    # --- rendering -----------------------------------------------------
    def _weighted_text(self, weights: List[TokenWeight]) -> Text:
        txt = Text()
        for tw in weights:
            if self.show_weights and tw.token:
                txt.append(tw.token, style=tw.color_style())
                txt.append(f"({tw.weight:.2f})", style="dim white")
            else:
                txt.append(tw.token)
            txt.append(" ")
        return txt

    def _align(self, left: List[TokenWeight], right: List[TokenWeight]) -> Tuple[List[TokenWeight], List[TokenWeight]]:
        if not self.align_tokens:
            return left, right
        max_len = max(len(left), len(right))
        while len(left) < max_len:
            left.append(TokenWeight("", 0.0))
        while len(right) < max_len:
            right.append(TokenWeight("", 0.0))
        return left, right

    def display(self, english: str, toaripi: str, content_type: str):
        if not self._provider:
            # Fallback: simple whitespace splitter with dummy provider
            from .token_weights import SimulatedTokenWeightProvider
            self._provider = SimulatedTokenWeightProvider()

        eng_tokens = self._provider.weights_for(english.split(), generated=False)
        tqo_tokens = self._provider.weights_for(toaripi.split(), generated=True)
        eng_tokens, tqo_tokens = self._align(eng_tokens, tqo_tokens)

        eng_panel = Panel(self._weighted_text(eng_tokens), title="🇺🇸 English", border_style="blue", padding=(1, 2))
        tqo_panel = Panel(self._weighted_text(tqo_tokens), title="🌺 Toaripi", border_style="green", padding=(1, 2))
        self.console.print(Columns([eng_panel, tqo_panel], equal=True, expand=True))
        if self.show_weights:
            self._legend()

    def _legend(self):
        legend = Text()
        legend.append("Token Weight Legend: ")
        legend.append("High (0.8+)", style="bold red"); legend.append(" | ")
        legend.append("Medium-High (0.6+)", style="bold orange3"); legend.append(" | ")
        legend.append("Medium (0.4+)", style="bold yellow"); legend.append(" | ")
        legend.append("Low (0.2+)", style="green"); legend.append(" | ")
        legend.append("Very Low (<0.2)", style="dim cyan")
        self.console.print(Panel(legend, title="🎨 Weight Color Guide", border_style="magenta", padding=(0, 1)))
