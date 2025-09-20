"""Interactive session state + persistence."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import json

from rich.console import Console

from .display import BilingualDisplay


@dataclass
class InteractiveSession:
    display: BilingualDisplay
    start: datetime = field(default_factory=datetime.now)
    conversation: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, *, user: str, english: str, toaripi: str, content_type: str):
        self.conversation.append(
            {
                "timestamp": datetime.now().isoformat(),
                "user_input": user,
                "english_content": english,
                "toaripi_content": toaripi,
                "content_type": content_type,
            }
        )

    def save(self, path: Path):
        data = {
            "session_start": self.start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "total_exchanges": len(self.conversation),
            "conversation": self.conversation,
            "display_settings": {
                "show_weights": self.display.show_weights,
                "align_tokens": self.display.align_tokens,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path
