#!/usr/bin/env python3
"""
Models command for Toaripi SLM CLI.

Provides model management capabilities including listing, conversion, and deployment.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
from loguru import logger


def get_model_info(model_path: Path) -> Dict:
    """Extract information about a model"""
    info = {
        "path": str(model_path),
        "name": model_path.name,
        "type": "unknown",
        "size_mb": 0,
        "config": {},
        "files": [],
        "valid": False
    }
    
    try:
        # Check if it's a directory (HuggingFace format)
        if model_path.is_dir():
            config_file = model_path / "config.json"
            if config_file.exists():
                info["type"] = "huggingface"
                with open(config_file, 'r') as f:
                    info["config"] = json.load(f)
                
                # List model files
                for file in model_path.iterdir():
                    if file.is_file():
                        size = file.stat().st_size / (1024 * 1024)  # MB
                        info["files"].append({
                            "name": file.name,
                            "size_mb": round(size, 2)
                        })
                        info["size_mb"] += size
                
                info["valid"] = True
        
        # Check if it's a GGUF file
        elif model_path.suffix == ".gguf":
            info["type"] = "gguf"
            info["size_mb"] = model_path.stat().st_size / (1024 * 1024)
            info["files"] = [{"name": model_path.name, "size_mb": info["size_mb"]}]
            info["valid"] = True
        
        info["size_mb"] = round(info["size_mb"], 2)
        
    except Exception as e:
        info["error"] = str(e)
    
    return info


def find_all_models() -> List[Dict]:
    """Find all available models in the project"""
    models = []
    
    # Search in checkpoints directory
    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        for item in checkpoints_dir.iterdir():
            if item.is_dir():
                info = get_model_info(item)
                if info["valid"]:
                    info["location"] = "checkpoints"
                    models.append(info)
    
    # Search in models directory
    models_dir = Path("models")
    if models_dir.exists():
        # HuggingFace format models
        hf_dir = models_dir / "hf"
        if hf_dir.exists():
            for item in hf_dir.iterdir():
                if item.is_dir():
                    info = get_model_info(item)
                    if info["valid"]:
                        info["location"] = "models/hf"
                        models.append(info)
        
        # GGUF format models
        gguf_dir = models_dir / "gguf"
        if gguf_dir.exists():
            for item in gguf_dir.iterdir():
                if item.suffix == ".gguf":
                    info = get_model_info(item)
                    if info["valid"]:
                        info["location"] = "models/gguf"
                        models.append(info)
    
    return models


def print_model_table(models: List[Dict]):
    """Print a formatted table of models"""
    if not models:
        click.echo("📭 No models found")
        return
    
    click.echo("\n📚 Available Models:")
    click.echo("-" * 80)
    click.echo(f"{'Name':<25} {'Type':<12} {'Location':<15} {'Size (MB)':<10} {'Status'}")
    click.echo("-" * 80)
    
    for model in models:
        status = "✅ Ready" if model["valid"] else "❌ Invalid"
        click.echo(f"{model['name']:<25} {model['type']:<12} {model['location']:<15} {model['size_mb']:<10.1f} {status}")
    
    click.echo("-" * 80)
    click.echo(f"Total: {len(models)} models, {sum(m['size_mb'] for m in models):.1f} MB")


def convert_to_gguf(model_path: Path, output_path: Path, quantization: str = "q4_k_m") -> bool:
    """Convert a HuggingFace model to GGUF format"""
    
    try:
        # Check if llama.cpp tools are available
        # This is a placeholder - actual implementation would depend on
        # having llama.cpp installed and configured
        click.echo(f"🔄 Converting {model_path} to GGUF format...")
        click.echo(f"   Quantization: {quantization}")
        click.echo(f"   Output: {output_path}")
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # This would be the actual conversion command
        # For now, we'll simulate it
        click.echo("⚠️  GGUF conversion requires llama.cpp tools")
        click.echo("   Install: https://github.com/ggerganov/llama.cpp")
        click.echo("   This feature will be implemented when dependencies are available")
        
        return False
        
    except Exception as e:
        click.echo(f"❌ Conversion failed: {e}")
        return False


def copy_model(source: Path, destination: Path) -> bool:
    """Copy a model to a new location"""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
        
        return True
    except Exception as e:
        click.echo(f"❌ Copy failed: {e}")
        return False


def delete_model(model_path: Path, confirm: bool = True) -> bool:
    """Delete a model"""
    if confirm:
        if not click.confirm(f"Are you sure you want to delete {model_path}?"):
            return False
    
    try:
        if model_path.is_dir():
            shutil.rmtree(model_path)
        else:
            model_path.unlink()
        
        click.echo(f"✅ Deleted {model_path}")
        return True
    except Exception as e:
        click.echo(f"❌ Deletion failed: {e}")
        return False


@click.group(invoke_without_command=True)
@click.pass_context
def models(ctx):
    """
    Manage Toaripi SLM models.
    
    This command group provides tools for listing, converting, copying,
    and managing trained models in various formats.
    
    \b
    Commands:
        list      List all available models
        info      Show detailed information about a model
        convert   Convert models between formats
        copy      Copy a model to a new location
        delete    Delete a model
        deploy    Prepare a model for deployment
    """
    if ctx.invoked_subcommand is None:
        # Default: list models
        models_list = find_all_models()
        print_model_table(models_list)


@models.command()
@click.option('--format', type=click.Choice(['table', 'json']), default='table', help='Output format')
def list(format):
    """List all available models"""
    models_list = find_all_models()
    
    if format == 'json':
        click.echo(json.dumps(models_list, indent=2))
    else:
        print_model_table(models_list)


@models.command()
@click.argument('model_name')
def info(model_name):
    """Show detailed information about a specific model"""
    models_list = find_all_models()
    
    # Find the model
    target_model = None
    for model in models_list:
        if model["name"] == model_name or model_name in model["path"]:
            target_model = model
            break
    
    if not target_model:
        click.echo(f"❌ Model not found: {model_name}")
        click.echo("\n💡 Available models:")
        print_model_table(models_list)
        return
    
    # Print detailed info
    click.echo(f"\n📊 Model Information: {target_model['name']}")
    click.echo("=" * 50)
    click.echo(f"📁 Path: {target_model['path']}")
    click.echo(f"🏷️  Type: {target_model['type']}")
    click.echo(f"📍 Location: {target_model['location']}")
    click.echo(f"💾 Size: {target_model['size_mb']:.1f} MB")
    click.echo(f"✅ Status: {'Valid' if target_model['valid'] else 'Invalid'}")
    
    # Configuration info
    if target_model.get("config"):
        config = target_model["config"]
        click.echo(f"\n⚙️  Configuration:")
        if "model_type" in config:
            click.echo(f"   Model type: {config['model_type']}")
        if "vocab_size" in config:
            click.echo(f"   Vocabulary size: {config['vocab_size']:,}")
        if "hidden_size" in config:
            click.echo(f"   Hidden size: {config['hidden_size']}")
        if "num_attention_heads" in config:
            click.echo(f"   Attention heads: {config['num_attention_heads']}")
        if "num_hidden_layers" in config:
            click.echo(f"   Layers: {config['num_hidden_layers']}")
    
    # Files info
    if target_model.get("files"):
        click.echo(f"\n📄 Files:")
        for file_info in target_model["files"]:
            click.echo(f"   {file_info['name']}: {file_info['size_mb']:.1f} MB")


@models.command()
@click.argument('source_model')
@click.option('--to-gguf', is_flag=True, help='Convert to GGUF format')
@click.option('--quantization', default='q4_k_m', help='GGUF quantization level')
@click.option('--output', help='Output path for converted model')
def convert(source_model, to_gguf, quantization, output):
    """Convert a model between formats"""
    models_list = find_all_models()
    
    # Find source model
    source_path = None
    for model in models_list:
        if model["name"] == source_model or source_model in model["path"]:
            source_path = Path(model["path"])
            break
    
    if not source_path:
        click.echo(f"❌ Source model not found: {source_model}")
        return
    
    if to_gguf:
        if not output:
            output = f"models/gguf/{source_path.name}.gguf"
        
        output_path = Path(output)
        success = convert_to_gguf(source_path, output_path, quantization)
        
        if success:
            click.echo(f"✅ Model converted to GGUF: {output_path}")
        else:
            click.echo("❌ GGUF conversion failed")
    else:
        click.echo("❌ Only GGUF conversion is currently supported")
        click.echo("💡 Use --to-gguf flag")


@models.command()
@click.argument('source_model')
@click.argument('destination')
def copy(source_model, destination):
    """Copy a model to a new location"""
    models_list = find_all_models()
    
    # Find source model
    source_path = None
    for model in models_list:
        if model["name"] == source_model or source_model in model["path"]:
            source_path = Path(model["path"])
            break
    
    if not source_path:
        click.echo(f"❌ Source model not found: {source_model}")
        return
    
    dest_path = Path(destination)
    
    click.echo(f"🔄 Copying {source_path} to {dest_path}...")
    success = copy_model(source_path, dest_path)
    
    if success:
        click.echo(f"✅ Model copied successfully")
    else:
        click.echo("❌ Copy failed")


@models.command()
@click.argument('model_name')
@click.option('--force', is_flag=True, help='Skip confirmation prompt')
def delete(model_name, force):
    """Delete a model"""
    models_list = find_all_models()
    
    # Find the model
    target_path = None
    for model in models_list:
        if model["name"] == model_name or model_name in model["path"]:
            target_path = Path(model["path"])
            break
    
    if not target_path:
        click.echo(f"❌ Model not found: {model_name}")
        return
    
    success = delete_model(target_path, confirm=not force)
    
    if success:
        click.echo(f"✅ Model deleted: {target_path}")


@models.command()
@click.argument('model_name')
@click.option('--format', type=click.Choice(['hf', 'gguf', 'both']), default='both', help='Deployment format')
@click.option('--output-dir', default='deployment', help='Output directory for deployment files')
def deploy(model_name, format, output_dir):
    """Prepare a model for deployment"""
    models_list = find_all_models()
    
    # Find the model
    target_model = None
    for model in models_list:
        if model["name"] == model_name or model_name in model["path"]:
            target_model = model
            break
    
    if not target_model:
        click.echo(f"❌ Model not found: {model_name}")
        return
    
    source_path = Path(target_model["path"])
    deploy_dir = Path(output_dir)
    deploy_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"📦 Preparing {model_name} for deployment...")
    
    # Copy original model
    if format in ['hf', 'both'] and target_model["type"] == "huggingface":
        hf_dest = deploy_dir / "hf" / source_path.name
        click.echo(f"🔄 Copying HuggingFace format to {hf_dest}...")
        copy_model(source_path, hf_dest)
    
    # Convert to GGUF if needed
    if format in ['gguf', 'both']:
        gguf_dest = deploy_dir / "gguf" / f"{source_path.name}.gguf"
        click.echo(f"🔄 Converting to GGUF format...")
        convert_to_gguf(source_path, gguf_dest)
    
    # Create deployment info
    deploy_info = {
        "model_name": model_name,
        "source_path": str(source_path),
        "deployment_date": time.time(),
        "formats": [],
        "files": {}
    }
    
    if (deploy_dir / "hf").exists():
        deploy_info["formats"].append("huggingface")
        deploy_info["files"]["huggingface"] = str(deploy_dir / "hf" / source_path.name)
    
    if (deploy_dir / "gguf").exists():
        deploy_info["formats"].append("gguf")
        deploy_info["files"]["gguf"] = str(deploy_dir / "gguf" / f"{source_path.name}.gguf")
    
    # Save deployment info
    with open(deploy_dir / "deployment_info.json", 'w') as f:
        json.dump(deploy_info, f, indent=2)
    
    click.echo(f"✅ Deployment package created in: {deploy_dir}")
    click.echo(f"📋 Deployment info saved to: {deploy_dir / 'deployment_info.json'}")


if __name__ == '__main__':
    models()