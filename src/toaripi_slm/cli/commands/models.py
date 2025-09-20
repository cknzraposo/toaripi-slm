"""Model version management commands for Toaripi SLM CLI."""

from pathlib import Path
import json
from typing import Dict, Any, Optional
import shutil
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

REGISTRY_FILE = Path("./models/hf/registry.json")

def load_registry() -> Dict[str, Any]:
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            console.print(f"❌ Failed to read registry: {e}")
    return {"models": []}

@click.group(help="Manage versioned Toaripi SLM models.")
def models():
    pass

@models.command("list", help="List all available model versions.")
def list_models():
    registry = load_registry()
    models = registry.get("models", [])
    if not models:
        console.print("⚠️  No models registered. Train one with: toaripi train")
        return
    table = Table(title="📦 Available Model Versions", header_style="bold magenta")
    table.add_column("Version", style="cyan")
    table.add_column("Base Model", style="green")
    table.add_column("Created", style="yellow")
    table.add_column("Checkpoint Dir", style="dim")
    for m in sorted(models, key=lambda x: x.get("created_at", "")):
        table.add_row(m.get("version", "?"), m.get("base_model", "?"), m.get("created_at", "?"), m.get("checkpoint_dir", "?"))
    console.print(table)

@models.command("info", help="Show detailed metadata for a specific version.")
@click.argument("version")
def model_info(version):
    registry = load_registry()
    target = None
    for m in registry.get("models", []):
        if m.get("version") == version:
            target = m
            break
    if not target:
        console.print(f"❌ Version not found: {version}")
        return
    info_file = Path(target.get("path", "")) / "model_info.json"
    if not info_file.exists():
        console.print(f"⚠️  model_info.json missing for {version}")
        return
    try:
        with open(info_file, "r") as f:
            info = json.load(f)
    except Exception as e:
        console.print(f"❌ Failed reading model_info.json: {e}")
        return
    panel_text = json.dumps(info, indent=2)
    console.print(Panel(panel_text, title=f"Model {version} Metadata", border_style="blue"))


def _materialize(version: str, *, overwrite: bool = False) -> bool:
    """Create a self-contained HF-style directory for the given version.

    Copies required tokenizer + config artifacts from the base model and, if
    present, adapter weights from checkpoint_dir. Skips files that already
    exist unless overwrite=True.
    """
    registry = load_registry()
    target_meta: Optional[Dict[str, Any]] = None
    for m in registry.get("models", []):
        if m.get("version") == version:
            target_meta = m
            break
    if not target_meta:
        console.print(f"❌ Version not found: {version}")
        return False
    version_path = Path(target_meta.get("path", ""))
    info_file = version_path / "model_info.json"
    if not info_file.exists():
        console.print(f"❌ model_info.json missing in {version_path}")
        return False
    try:
        with open(info_file, "r") as f:
            info = json.load(f)
    except Exception as e:
        console.print(f"❌ Failed to parse model_info.json: {e}")
        return False
    base = info.get("base_model")
    if not base:
        console.print("❌ base_model not specified in metadata")
        return False
    try:
        from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM  # noqa: F401
    except ImportError:
        console.print("⚠️  transformers not installed; cannot materialize.")
        return False
    # Fetch base model locally (will download to cache if not present)
    try:
        from transformers import AutoTokenizer, AutoConfig
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        config = AutoConfig.from_pretrained(base, trust_remote_code=True)
    except Exception as e:
        console.print(f"❌ Failed to load base assets: {e}")
        return False

    version_path.mkdir(parents=True, exist_ok=True)
    # Persist tokenizer + config
    if overwrite or not (version_path / "tokenizer.json").exists():
        try:
            tokenizer.save_pretrained(str(version_path))
        except Exception:
            pass  # pragma: no cover
    if overwrite or not (version_path / "config.json").exists():
        try:
            config.to_json_file(version_path / "config.json")
        except Exception:
            pass  # pragma: no cover

    # Copy adapter weights if available
    ckpt_dir = info.get("checkpoint_dir")
    if ckpt_dir:
        ckpt_path = Path(ckpt_dir)
        if ckpt_path.exists():
            for f in ckpt_path.glob("*adapter_model.bin"):
                dest = version_path / f.name
                if overwrite or not dest.exists():
                    try:
                        shutil.copy2(f, dest)
                    except Exception:
                        console.print(f"⚠️  Failed copying adapter {f.name}")

    console.print(f"✅ Materialized artifacts for {version} in {version_path}")
    return True


@models.command("materialize", help="Create missing HF artifacts for a version (config, tokenizer, adapters).")
@click.argument("version")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files.")
def materialize_cmd(version: str, overwrite: bool):
    ok = _materialize(version, overwrite=overwrite)
    if not ok:
        raise SystemExit(1)
