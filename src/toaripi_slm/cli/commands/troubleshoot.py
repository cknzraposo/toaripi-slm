#!/usr/bin/env python3
"""
Troubleshooting utilities for Toaripi SLM CLI.

This module provides diagnostic and troubleshooting capabilities.
"""

import importlib
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click


def check_python_environment() -> Dict[str, any]:
    """Check Python environment and dependencies"""
    checks = {}
    
    # Python version
    python_version = sys.version_info
    checks['python_version'] = {
        'version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
        'compatible': python_version >= (3, 10),
        'recommended': python_version >= (3, 11)
    }
    
    # Virtual environment
    checks['virtual_env'] = {
        'active': hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix),
        'path': sys.prefix
    }
    
    # Platform info
    checks['platform'] = {
        'system': platform.system(),
        'machine': platform.machine(),
        'python_implementation': platform.python_implementation()
    }
    
    return checks


def check_dependencies() -> Dict[str, Dict]:
    """Check status of required dependencies"""
    required_deps = {
        'torch': 'PyTorch for deep learning',
        'transformers': 'Hugging Face transformers',
        'datasets': 'Hugging Face datasets',
        'accelerate': 'Hugging Face accelerate',
        'peft': 'Parameter-Efficient Fine-Tuning',
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'click': 'Command-line interface',
        'loguru': 'Logging',
        'fastapi': 'Web framework',
        'uvicorn': 'ASGI server'
    }
    
    dep_status = {}
    
    for name, description in required_deps.items():
        try:
            module = importlib.import_module(name)
            version = getattr(module, '__version__', 'unknown')
            dep_status[name] = {
                'installed': True,
                'version': version,
                'description': description
            }
        except ImportError:
            dep_status[name] = {
                'installed': False,
                'version': None,
                'description': description
            }
    
    return dep_status


def check_gpu_availability() -> Dict[str, any]:
    """Check GPU availability and configuration"""
    gpu_info = {
        'cuda_available': False,
        'cuda_version': None,
        'device_count': 0,
        'devices': []
    }
    
    try:
        import torch
        gpu_info['cuda_available'] = torch.cuda.is_available()
        
        if gpu_info['cuda_available']:
            gpu_info['cuda_version'] = torch.version.cuda
            gpu_info['device_count'] = torch.cuda.device_count()
            
            for i in range(gpu_info['device_count']):
                device_props = torch.cuda.get_device_properties(i)
                gpu_info['devices'].append({
                    'id': i,
                    'name': device_props.name,
                    'memory_total': device_props.total_memory / (1024**3),  # GB
                    'memory_free': torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Simplified
                })
    except ImportError:
        pass
    
    return gpu_info


def check_disk_space(directories: List[str]) -> Dict[str, Dict]:
    """Check available disk space for important directories"""
    space_info = {}
    
    for directory in directories:
        path = Path(directory)
        if path.exists():
            try:
                stat = shutil.disk_usage(path)
                space_info[directory] = {
                    'total_gb': stat.total / (1024**3),
                    'free_gb': stat.free / (1024**3),
                    'used_gb': (stat.total - stat.free) / (1024**3),
                    'free_percent': (stat.free / stat.total) * 100
                }
            except Exception:
                space_info[directory] = {'error': 'Cannot access directory'}
        else:
            space_info[directory] = {'error': 'Directory does not exist'}
    
    return space_info


def check_project_structure() -> Dict[str, bool]:
    """Check if project structure is properly set up"""
    required_dirs = [
        'data',
        'data/raw',
        'data/processed',
        'data/samples',
        'configs',
        'configs/training',
        'configs/data',
        'models',
        'checkpoints'
    ]
    
    structure_status = {}
    for directory in required_dirs:
        structure_status[directory] = Path(directory).exists()
    
    return structure_status


def diagnose_common_issues() -> List[Dict[str, str]]:
    """Diagnose common issues and provide solutions"""
    issues = []
    
    # Check Python version
    python_check = check_python_environment()
    if not python_check['python_version']['compatible']:
        issues.append({
            'type': 'error',
            'title': 'Incompatible Python Version',
            'problem': f"Python {python_check['python_version']['version']} detected, but 3.10+ required",
            'solution': 'Upgrade to Python 3.10 or later. Consider using pyenv or conda.'
        })
    
    # Check dependencies
    deps = check_dependencies()
    missing_deps = [name for name, info in deps.items() if not info['installed']]
    if missing_deps:
        issues.append({
            'type': 'error',
            'title': 'Missing Dependencies',
            'problem': f"Missing required packages: {', '.join(missing_deps)}",
            'solution': 'Install missing dependencies: pip install -r requirements.txt'
        })
    
    # Check disk space
    disk_info = check_disk_space(['.', 'models', 'data'])
    for path, info in disk_info.items():
        if 'free_gb' in info and info['free_gb'] < 5:
            issues.append({
                'type': 'warning',
                'title': 'Low Disk Space',
                'problem': f"Only {info['free_gb']:.1f}GB free in {path}",
                'solution': 'Free up disk space. Models can be 1-10GB+.'
            })
    
    # Check GPU
    gpu_info = check_gpu_availability()
    if not gpu_info['cuda_available']:
        issues.append({
            'type': 'info',
            'title': 'No GPU Detected',
            'problem': 'CUDA GPU not available, training will use CPU',
            'solution': 'For faster training, consider using a GPU-enabled environment or reduce model size.'
        })
    
    # Check project structure
    structure = check_project_structure()
    missing_dirs = [path for path, exists in structure.items() if not exists]
    if missing_dirs:
        issues.append({
            'type': 'warning',
            'title': 'Incomplete Project Structure',
            'problem': f"Missing directories: {', '.join(missing_dirs)}",
            'solution': 'Run: toaripi setup'
        })
    
    return issues


def print_system_report():
    """Print a comprehensive system report"""
    click.echo("🔍 Toaripi SLM System Diagnostic Report")
    click.echo("=" * 60)
    
    # Python environment
    python_info = check_python_environment()
    click.echo(f"\n🐍 Python Environment:")
    click.echo(f"   Version: {python_info['python_version']['version']}")
    status = "✅" if python_info['python_version']['compatible'] else "❌"
    click.echo(f"   Compatible: {status}")
    
    venv_status = "✅ Active" if python_info['virtual_env']['active'] else "⚠️ Not active"
    click.echo(f"   Virtual Environment: {venv_status}")
    click.echo(f"   Platform: {python_info['platform']['system']} {python_info['platform']['machine']}")
    
    # Dependencies
    click.echo(f"\n📦 Dependencies:")
    deps = check_dependencies()
    for name, info in deps.items():
        status = "✅" if info['installed'] else "❌"
        version = f"({info['version']})" if info['version'] else ""
        click.echo(f"   {status} {name} {version}")
    
    # GPU
    click.echo(f"\n🖥️  GPU Information:")
    gpu_info = check_gpu_availability()
    if gpu_info['cuda_available']:
        click.echo(f"   ✅ CUDA Available (v{gpu_info['cuda_version']})")
        click.echo(f"   🎮 Devices: {gpu_info['device_count']}")
        for device in gpu_info['devices']:
            click.echo(f"      {device['id']}: {device['name']} ({device['memory_total']:.1f}GB)")
    else:
        click.echo(f"   ℹ️ CUDA not available (CPU only)")
    
    # Disk space
    click.echo(f"\n💾 Disk Space:")
    disk_info = check_disk_space(['.', 'models', 'data'])
    for path, info in disk_info.items():
        if 'free_gb' in info:
            click.echo(f"   {path}: {info['free_gb']:.1f}GB free / {info['total_gb']:.1f}GB total")
        else:
            click.echo(f"   {path}: {info.get('error', 'Unknown error')}")
    
    # Project structure
    click.echo(f"\n📁 Project Structure:")
    structure = check_project_structure()
    for path, exists in structure.items():
        status = "✅" if exists else "❌"
        click.echo(f"   {status} {path}")
    
    # Issues and recommendations
    issues = diagnose_common_issues()
    if issues:
        click.echo(f"\n⚠️ Issues and Recommendations:")
        for issue in issues:
            icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(issue['type'], "•")
            click.echo(f"\n   {icon} {issue['title']}")
            click.echo(f"      Problem: {issue['problem']}")
            click.echo(f"      Solution: {issue['solution']}")
    else:
        click.echo(f"\n✅ No issues detected!")
    
    click.echo(f"\n" + "=" * 60)


@click.command()
@click.option('--report', is_flag=True, help='Generate full system report')
@click.option('--fix-permissions', is_flag=True, help='Attempt to fix common permission issues')
@click.option('--clean-cache', is_flag=True, help='Clean model and data caches')
def troubleshoot(report, fix_permissions, clean_cache):
    """
    Diagnose and fix common issues with Toaripi SLM setup.
    
    This command helps identify and resolve common problems with
    dependencies, permissions, disk space, and project configuration.
    
    \b
    Examples:
        toaripi troubleshoot --report          # Full diagnostic report
        toaripi troubleshoot --fix-permissions # Fix permission issues
        toaripi troubleshoot --clean-cache     # Clean caches
    """
    
    if report:
        print_system_report()
        return
    
    if fix_permissions:
        click.echo("🔧 Attempting to fix permission issues...")
        # This would implement permission fixes
        click.echo("⚠️ Permission fixing not yet implemented")
        return
    
    if clean_cache:
        click.echo("🧹 Cleaning caches...")
        cache_dirs = [
            Path.home() / ".cache" / "huggingface",
            Path("models") / "cache",
            Path("data") / "cache"
        ]
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                try:
                    shutil.rmtree(cache_dir)
                    click.echo(f"✅ Cleaned {cache_dir}")
                except Exception as e:
                    click.echo(f"❌ Failed to clean {cache_dir}: {e}")
        return
    
    # Default: quick diagnostic
    click.echo("🔍 Running quick diagnostic...")
    issues = diagnose_common_issues()
    
    if issues:
        click.echo("\n⚠️ Issues found:")
        for issue in issues:
            icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(issue['type'], "•")
            click.echo(f"\n{icon} {issue['title']}")
            click.echo(f"   {issue['problem']}")
            click.echo(f"   💡 {issue['solution']}")
    else:
        click.echo("✅ No issues detected!")
    
    click.echo(f"\n💡 For a detailed report, run: toaripi troubleshoot --report")


if __name__ == '__main__':
    troubleshoot()