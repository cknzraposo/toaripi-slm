#!/usr/bin/env python3
"""
Main CLI entry point for Toaripi SLM.

This provides a sleek, intuitive command-line interface for training, testing,
and interacting with Toaripi Small Language Models with guided user experience.
"""

import os
import sys
import platform
from pathlib import Path
from typing import Dict, List, Optional

import click
from loguru import logger

# Try to import our modules, handle gracefully if missing
try:
    from ..utils import setup_logging, load_config
except ImportError:
    def setup_logging(level="INFO", log_file=None):
        """Fallback logging setup"""
        import logging
        logging.basicConfig(level=getattr(logging, level))
    
    def load_config(path):
        """Fallback config loader"""
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)


# CLI Styling and Colors
class Colors:
    """Cross-platform color codes for CLI output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def disable(cls):
        """Disable colors for Windows compatibility"""
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ''
        cls.YELLOW = cls.RED = cls.ENDC = cls.BOLD = cls.UNDERLINE = ''


# Disable colors on Windows by default unless explicitly enabled
if platform.system() == "Windows" and not os.environ.get("FORCE_COLOR"):
    Colors.disable()


def print_banner():
    """Print the Toaripi SLM banner"""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
████████╗ ██████╗  █████╗ ██████╗ ██╗██████╗ ██╗    
╚══██╔══╝██╔═══██╗██╔══██╗██╔══██╗██║██╔══██╗██║    
   ██║   ██║   ██║███████║██████╔╝██║██████╔╝██║    
   ██║   ██║   ██║██╔══██║██╔══██╗██║██╔═══╝ ██║    
   ██║   ╚██████╔╝██║  ██║██║  ██║██║██║     ██║    
   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝    
{Colors.ENDC}
{Colors.GREEN}🌴 Toaripi Small Language Model CLI{Colors.ENDC}
{Colors.BLUE}Educational Content Generation for Language Preservation{Colors.ENDC}

{Colors.YELLOW}💡 Need help getting started? Run: {Colors.BOLD}toaripi setup --guided{Colors.ENDC}
    """
    click.echo(banner)


def check_system_requirements() -> Dict[str, bool]:
    """Check system requirements and return status"""
    checks = {}
    
    # Python version
    python_version = sys.version_info
    checks['python_version'] = python_version >= (3, 10)
    
    # Available memory (rough estimate)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        checks['memory'] = memory_gb >= 8
    except ImportError:
        checks['memory'] = None  # Can't determine
    
    # GPU availability
    try:
        import torch
        checks['gpu'] = torch.cuda.is_available()
    except ImportError:
        checks['gpu'] = False
    
    # Required directories
    current_dir = Path.cwd()
    checks['data_dir'] = (current_dir / 'data').exists()
    checks['config_dir'] = (current_dir / 'configs').exists()
    
    return checks


def print_system_status():
    """Print current system status"""
    checks = check_system_requirements()
    
    click.echo(f"\n{Colors.BOLD}🔧 System Status:{Colors.ENDC}")
    
    # Python version
    version_str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    status = "✅" if checks['python_version'] else "❌"
    click.echo(f"  {status} Python {version_str} {'(compatible)' if checks['python_version'] else '(needs 3.10+)'}")
    
    # Memory
    if checks['memory'] is not None:
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            status = "✅" if checks['memory'] else "⚠️"
            click.echo(f"  {status} Memory: {memory_gb:.1f}GB {'(sufficient)' if checks['memory'] else '(8GB+ recommended)'}")
        except ImportError:
            click.echo(f"  ❓ Memory: Unable to detect (install psutil for memory check)")
    
    # GPU
    status = "✅" if checks['gpu'] else "ℹ️"
    click.echo(f"  {status} GPU: {'CUDA available' if checks['gpu'] else 'CPU only (GPU recommended for training)'}")
    
    # Project structure
    status = "✅" if checks['data_dir'] else "⚠️"
    click.echo(f"  {status} Data directory: {'Found' if checks['data_dir'] else 'Missing (run: toaripi setup)'}")
    
    status = "✅" if checks['config_dir'] else "⚠️"
    click.echo(f"  {status} Config directory: {'Found' if checks['config_dir'] else 'Missing (run: toaripi setup)'}")


def get_available_models() -> List[str]:
    """Get list of available trained models"""
    models_dir = Path.cwd() / "models"
    if not models_dir.exists():
        return []
    
    models = []
    
    # Check for HuggingFace format models
    for model_dir in models_dir.glob("*/"):
        if (model_dir / "config.json").exists():
            models.append(str(model_dir.name))
    
    # Check for GGUF models
    gguf_dir = models_dir / "gguf"
    if gguf_dir.exists():
        for gguf_file in gguf_dir.glob("*.gguf"):
            models.append(f"gguf/{gguf_file.name}")
    
    return models


def validate_data_directory(data_dir: Path) -> bool:
    """Validate that data directory has required files"""
    required_files = ["train.csv", "validation.csv"]
    
    for file in required_files:
        if not (data_dir / file).exists():
            click.echo(f"❌ Missing required file: {data_dir / file}")
            return False
    
    click.echo(f"✅ Data directory validated: {data_dir}")
    return True


@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version information')
@click.option('--status', is_flag=True, help='Show system status')
@click.pass_context
def cli(ctx, version, status):
    """
    🌴 Toaripi Small Language Model CLI
    
    A comprehensive toolkit for training, testing, and deploying educational
    language models for the Toaripi language.
    
    \b
    Quick Start:
        toaripi setup --guided    # First-time setup with guidance
        toaripi train --help      # Training options
        toaripi interact          # Chat with your model
        toaripi status            # Check system status
    
    \b
    Examples:
        toaripi train --data data/processed --model microsoft/DialoGPT-medium
        toaripi test --model models/toaripi-model --data data/processed/test.csv
        toaripi interact --model models/toaripi-model
    """
    if ctx.invoked_subcommand is None:
        if version:
            try:
                from .. import __version__
                click.echo(f"Toaripi SLM version {__version__}")
            except ImportError:
                click.echo("Toaripi SLM version 0.1.0")
            return
        
        if status:
            print_system_status()
            return
        
        # Default behavior: show banner and help
        print_banner()
        click.echo(ctx.get_help())


# Import subcommands (handle import errors gracefully)
try:
    from .commands import setup, train, test, interact, models, troubleshoot
    
    # Register subcommands
    cli.add_command(setup.setup)
    cli.add_command(train.train)
    cli.add_command(test.test)
    cli.add_command(interact.interact)
    cli.add_command(models.models)
    cli.add_command(troubleshoot.troubleshoot)
    
except ImportError as e:
    # Fallback for development or when modules aren't fully available
    logger.warning(f"Some CLI commands not available: {e}")
    
    @cli.command()
    def setup():
        """Project setup (not available)"""
        click.echo("❌ Setup command not available - package may not be fully installed")
    
    @cli.command()
    def train():
        """Model training (not available)"""
        click.echo("❌ Train command not available - package may not be fully installed")
    
    @cli.command()
    def test():
        """Model testing (not available)"""
        click.echo("❌ Test command not available - package may not be fully installed")
    
    @cli.command()
    def interact():
        """Interactive mode (not available)"""
        click.echo("❌ Interact command not available - package may not be fully installed")
    
    @cli.group()
    def models():
        """Model management (not available)"""
        click.echo("❌ Models command not available - package may not be fully installed")
    
    @cli.command()
    def troubleshoot():
        """Troubleshooting (not available)"""
        click.echo("❌ Troubleshoot command not available - package may not be fully installed")


if __name__ == '__main__':
    cli()