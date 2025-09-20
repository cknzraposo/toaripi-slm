"""Model export command (HF -> edge formats placeholder).

Currently provides a stub for exporting a registered model version to a
GGUF (llama.cpp) directory. The actual conversion pipeline will be
implemented later (quantization tooling integration).
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import click
from rich.console import Console
from rich.panel import Panel

from ..core.versioning import resolve_version_dir, latest_version, load_registry
from ..core.exporter import prepare_export, push_to_hub

console = Console()


@click.command()
@click.option("--version", help="Model version to export (defaults to latest).")
@click.option("--format", "export_format", default="gguf", type=click.Choice(["gguf"], case_sensitive=False))
@click.option("--quant", default="q4_k_m", help="Quantization preset (placeholder or used in manifest).")
@click.option("--output-dir", type=click.Path(file_okay=False), default="models/gguf", help="Directory for exported model")
@click.option("--push", is_flag=True, help="Push exported model to Hugging Face Hub.")
@click.option("--repo-id", help="Hugging Face repo id (e.g. username/toaripi-educational-slm)")
@click.option("--private", is_flag=True, help="Create as private repo on the Hub.")
@click.option("--no-card", is_flag=True, help="Skip generating README model card.")
@click.option("--token", envvar="HF_TOKEN", help="Hugging Face auth token (or set HF_TOKEN env var)")
def export(version: str | None, export_format: str, quant: str, output_dir: str, push: bool, repo_id: str | None, private: bool, no_card: bool, token: str | None):
    """Export a trained model version (and optionally push to Hugging Face)."""
    target_version = version or latest_version()
    if not target_version:
        console.print("❌ No models registered. Train a model first.")
        return
    model_dir = resolve_version_dir(target_version)
    if not model_dir:
        console.print(f"❌ Version not found: {target_version}")
        return

    # Load metadata from registry entry
    registry = load_registry()
    meta = None
    for m in registry.get("models", []):
        if m.get("version") == target_version:
            meta = m
            break
    if not meta:
        console.print(f"⚠️  Metadata for version {target_version} not found; proceeding with minimal manifest.")
        meta = {"version": target_version, "path": str(model_dir), "base_model": "unknown", "created_at": datetime.utcnow().isoformat()}

    export_root = Path(output_dir)
    export_dir = prepare_export(meta, export_root=export_root, include_card=not no_card, quantization=quant)

    console.print(
        Panel(
            f"Prepared export for [cyan]{target_version}[/cyan] at [green]{export_dir}[/green].\n"
            f"Format: {export_format}  Quant: {quant}",
            title="Model Export",
            border_style="blue",
        )
    )

    if push:
        if not repo_id:
            console.print("❌ --repo-id required when using --push")
            return
        console.print(f"🚀 Pushing to Hugging Face Hub: {repo_id} (private={private})")
        success = push_to_hub(export_dir, repo_id=repo_id, private=private, token=token, create_card=not no_card)
        if not success:
            console.print("⚠️  Push failed. You can retry with the same export directory.")
        else:
            console.print("✅ Export push complete.")
