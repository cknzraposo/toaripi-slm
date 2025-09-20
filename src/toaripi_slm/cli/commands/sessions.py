"""Session management commands for interactive chat logs."""
from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

SESSIONS_DIR = Path("./chat_sessions")


@click.group()
def sessions():  # noqa: D401
    """Manage saved interactive sessions."""
    pass


@sessions.command("list")
def list_sessions():
    """List saved session files."""
    if not SESSIONS_DIR.exists():
        console.print("No sessions saved yet.")
        return
    files = sorted(SESSIONS_DIR.glob("session_*.json")) + sorted(SESSIONS_DIR.glob("auto_session_*.json"))
    if not files:
        console.print("No sessions found.")
        return
    table = Table(title="Saved Sessions", show_lines=False, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Filename", style="green")
    table.add_column("Exchanges", justify="right")
    table.add_column("Start Time (UTC)")
    for idx, f in enumerate(files, 1):
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
            exchanges = data.get("total_exchanges") or len(data.get("conversation", []))
            start = data.get("session_start", "?")
        except Exception:
            exchanges = "-"
            start = "?"
        table.add_row(str(idx), f.name, str(exchanges), start)
    console.print(table)


@sessions.command("show")
@click.argument("filename")
def show_session(filename: str):
    """Show summary of a session file."""
    path = SESSIONS_DIR / filename
    if not path.exists():
        console.print(f"❌ Session file not found: {path}")
        return
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"❌ Failed reading session: {e}")
        return
    exchanges = data.get("total_exchanges") or len(data.get("conversation", []))
    panel = Panel(
        f"File: {path.name}\nStart: {data.get('session_start')}\nEnd: {data.get('session_end')}\nExchanges: {exchanges}",
        title="Session Summary",
        border_style="blue",
    )
    console.print(panel)
    # Optionally list first few exchanges
    convo = data.get("conversation") or data.get("conversation_history") or []
    for i, ex in enumerate(convo[:5], 1):
        console.print(f"[bold cyan]{i}. {ex.get('content_type','?')}[/bold cyan] {ex.get('user_input','')}")
        console.print(f"  [green]{ex.get('toaripi_content','')}[/green]")


@sessions.command("replay")
@click.argument("filename")
def replay_session(filename: str):
    """Replay a session to the console (first 10 exchanges)."""
    path = SESSIONS_DIR / filename
    if not path.exists():
        console.print(f"❌ Session file not found: {path}")
        return
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"❌ Failed reading session: {e}")
        return
    convo = data.get("conversation") or data.get("conversation_history") or []
    console.print(Panel(f"Replaying {len(convo)} exchanges (showing up to 10)", title="Replay", border_style="magenta"))
    for i, ex in enumerate(convo[:10], 1):
        console.print(f"[bold yellow]You:[/bold yellow] {ex.get('user_input','')}")
        console.print(f"[green]Toaripi:[/green] {ex.get('toaripi_content','')}")
        console.print()

__all__ = ["sessions"]
