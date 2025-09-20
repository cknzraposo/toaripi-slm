"""
System diagnostic and configuration commands.
"""

import click
import platform
import sys
import psutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree
from rich.prompt import Confirm, Prompt

from ...context import get_context

console = Console()

@click.group()
def system():
    """System diagnostics, configuration, and maintenance commands."""
    pass

@system.command()
@click.option("--detailed", "-d", is_flag=True, help="Show detailed system information")
@click.option("--json-output", type=Path, help="Save status to JSON file")
def status(detailed, json_output):
    """Check system status and environment setup."""
    ctx = get_context()
    
    console.print("🔍 [bold blue]System Status Check[/bold blue]\n")
    
    # Gather system information
    status_data = gather_system_status()
    
    # Display basic status
    display_basic_status(status_data)
    
    if detailed:
        display_detailed_status(status_data)
    
    # Save to JSON if requested
    if json_output:
        save_status_report(status_data, json_output)
        console.print(f"📄 Status report saved to: [cyan]{json_output}[/cyan]")

def gather_system_status() -> Dict[str, Any]:
    """Gather comprehensive system status information."""
    
    status = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        },
        "python": {
            "version": sys.version,
            "version_info": sys.version_info[:3],
            "executable": sys.executable,
            "platform": sys.platform
        },
        "hardware": {},
        "directories": {},
        "dependencies": {},
        "toaripi_slm": {}
    }
    
    # Hardware information
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        status["hardware"] = {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_usage_percent": memory.percent,
            "disk_total_gb": disk.total / (1024**3),
            "disk_free_gb": disk.free / (1024**3),
            "disk_usage_percent": (disk.used / disk.total) * 100
        }
    except Exception as e:
        status["hardware"]["error"] = str(e)
    
    # Directory status
    ctx = get_context()
    directories = {
        "config": ctx.config_dir,
        "data": ctx.data_dir,
        "models": ctx.models_dir,
        "cache": ctx.cache_dir,
        "sessions": ctx.sessions_dir
    }
    
    for name, path in directories.items():
        status["directories"][name] = {
            "path": str(path),
            "exists": path.exists(),
            "writable": path.is_dir() and os.access(path, os.W_OK) if path.exists() else False
        }
    
    # Dependencies
    dependencies = [
        "torch", "transformers", "datasets", "accelerate", "peft",
        "yaml", "pandas", "numpy", "rich", "click", "psutil"
    ]
    
    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, "__version__", "unknown")
            status["dependencies"][dep] = {
                "available": True,
                "version": version
            }
        except ImportError:
            status["dependencies"][dep] = {
                "available": False,
                "version": None
            }
    
    # Toaripi SLM specific status
    status["toaripi_slm"] = {
        "models_found": count_trained_models(),
        "sample_data_available": check_sample_data(),
        "config_files_present": check_config_files()
    }
    
    return status

def display_basic_status(status: Dict[str, Any]):
    """Display basic system status."""
    
    # System overview
    platform_info = status["platform"]
    python_info = status["python"]
    hardware_info = status.get("hardware", {})
    
    overview_panel = Panel(
        f"""
        [bold cyan]System Overview[/bold cyan]
        
        OS: {platform_info["system"]} {platform_info["release"]}
        Python: {python_info["version_info"][0]}.{python_info["version_info"][1]}.{python_info["version_info"][2]}
        CPU Cores: {hardware_info.get("cpu_count", "Unknown")}
        Memory: {hardware_info.get("memory_total_gb", 0):.1f} GB total, {hardware_info.get("memory_available_gb", 0):.1f} GB available
        """,
        title="💻 System Info",
        border_style="blue"
    )
    console.print(overview_panel)
    
    # Health status
    health_issues = []
    
    # Check Python version
    if python_info["version_info"][0] < 3 or (python_info["version_info"][0] == 3 and python_info["version_info"][1] < 10):
        health_issues.append("Python 3.10+ required")
    
    # Check memory
    if hardware_info.get("memory_total_gb", 0) < 4:
        health_issues.append("Low memory (recommend 8GB+)")
    
    # Check dependencies
    missing_deps = [dep for dep, info in status["dependencies"].items() 
                   if not info.get("available", False)]
    if missing_deps:
        health_issues.append(f"Missing dependencies: {', '.join(missing_deps)}")
    
    # Check directories
    missing_dirs = [name for name, info in status["directories"].items() 
                   if not info.get("exists", False)]
    if missing_dirs:
        health_issues.append(f"Missing directories: {', '.join(missing_dirs)}")
    
    # Display health status
    if health_issues:
        health_status = "⚠️  Issues Found"
        health_color = "yellow"
        health_details = "\n".join(f"• {issue}" for issue in health_issues)
    else:
        health_status = "✅ System Healthy"
        health_color = "green" 
        health_details = "All systems operational"
    
    health_panel = Panel(
        f"""
        [bold {health_color}]{health_status}[/bold {health_color}]
        
        {health_details}
        """,
        title="🏥 Health Status",
        border_style=health_color
    )
    console.print(health_panel)

def display_detailed_status(status: Dict[str, Any]):
    """Display detailed system status information."""
    
    # Dependencies table
    deps_table = Table(show_header=True, header_style="bold magenta")
    deps_table.add_column("Package", style="cyan")
    deps_table.add_column("Status", style="green")
    deps_table.add_column("Version", style="yellow")
    
    for dep, info in status["dependencies"].items():
        status_icon = "✅" if info["available"] else "❌"
        version = info.get("version", "N/A")
        deps_table.add_row(dep, status_icon, version)
    
    console.print("\n📦 Dependencies:")
    console.print(deps_table)
    
    # Directories table
    dirs_table = Table(show_header=True, header_style="bold magenta")
    dirs_table.add_column("Directory", style="cyan")
    dirs_table.add_column("Path", style="blue")
    dirs_table.add_column("Exists", style="green")
    dirs_table.add_column("Writable", style="yellow")
    
    for name, info in status["directories"].items():
        exists_icon = "✅" if info["exists"] else "❌"
        writable_icon = "✅" if info.get("writable", False) else "❌"
        dirs_table.add_row(name.title(), info["path"], exists_icon, writable_icon)
    
    console.print("\n📁 Directories:")
    console.print(dirs_table)
    
    # Hardware details
    hardware = status.get("hardware", {})
    if hardware and "error" not in hardware:
        hw_table = Table(show_header=True, header_style="bold magenta")
        hw_table.add_column("Component", style="cyan")
        hw_table.add_column("Details", style="green")
        
        hw_table.add_row("CPU Cores", str(hardware.get("cpu_count", "Unknown")))
        if hardware.get("cpu_freq"):
            hw_table.add_row("CPU Frequency", f"{hardware['cpu_freq']:.0f} MHz")
        hw_table.add_row("Memory Total", f"{hardware.get('memory_total_gb', 0):.1f} GB")
        hw_table.add_row("Memory Available", f"{hardware.get('memory_available_gb', 0):.1f} GB")
        hw_table.add_row("Memory Usage", f"{hardware.get('memory_usage_percent', 0):.1f}%")
        hw_table.add_row("Disk Total", f"{hardware.get('disk_total_gb', 0):.1f} GB")
        hw_table.add_row("Disk Free", f"{hardware.get('disk_free_gb', 0):.1f} GB")
        hw_table.add_row("Disk Usage", f"{hardware.get('disk_usage_percent', 0):.1f}%")
        
        console.print("\n🖥️  Hardware:")
        console.print(hw_table)

@system.command()
@click.option("--detailed", "-d", is_flag=True, help="Show detailed diagnostic information")
@click.option("--fix", "-f", is_flag=True, help="Attempt to fix common issues automatically")
@click.option("--export-report", type=Path, help="Export diagnostic report to file")
def doctor(detailed, fix, export_report):
    """Comprehensive system health check and troubleshooting."""
    ctx = get_context()
    
    console.print("🩺 [bold blue]Toaripi SLM System Doctor[/bold blue]\n")
    
    # Run diagnostics
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Diagnostic steps
        diagnostics = run_comprehensive_diagnostics(progress, fix)
    
    # Display results
    display_diagnostic_results(diagnostics, detailed)
    
    # Export report if requested
    if export_report:
        export_diagnostic_report(diagnostics, export_report)

def run_comprehensive_diagnostics(progress: Progress, auto_fix: bool = False) -> Dict[str, Any]:
    """Run comprehensive system diagnostics."""
    
    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "tests": {},
        "fixes_applied": [],
        "recommendations": []
    }
    
    # Test 1: Python environment
    task = progress.add_task("Checking Python environment...", total=None)
    diagnostics["tests"]["python_environment"] = check_python_environment()
    progress.update(task, completed=100)
    
    # Test 2: Dependencies
    task = progress.add_task("Validating dependencies...", total=None)
    diagnostics["tests"]["dependencies"] = check_dependencies(auto_fix)
    if auto_fix and diagnostics["tests"]["dependencies"].get("fixes_applied"):
        diagnostics["fixes_applied"].extend(diagnostics["tests"]["dependencies"]["fixes_applied"])
    progress.update(task, completed=100)
    
    # Test 3: Directory structure
    task = progress.add_task("Verifying directory structure...", total=None)
    diagnostics["tests"]["directories"] = check_directory_structure(auto_fix)
    if auto_fix and diagnostics["tests"]["directories"].get("fixes_applied"):
        diagnostics["fixes_applied"].extend(diagnostics["tests"]["directories"]["fixes_applied"])
    progress.update(task, completed=100)
    
    # Test 4: Data availability
    task = progress.add_task("Checking training data...", total=None)
    diagnostics["tests"]["data"] = check_training_data()
    progress.update(task, completed=100)
    
    # Test 5: Model status
    task = progress.add_task("Checking trained models...", total=None)
    diagnostics["tests"]["models"] = check_model_status()
    progress.update(task, completed=100)
    
    # Test 6: Hardware resources
    task = progress.add_task("Analyzing hardware resources...", total=None)
    diagnostics["tests"]["hardware"] = check_hardware_resources()
    progress.update(task, completed=100)
    
    # Test 7: Configuration files
    task = progress.add_task("Validating configuration...", total=None)
    diagnostics["tests"]["configuration"] = check_configuration_files(auto_fix)
    if auto_fix and diagnostics["tests"]["configuration"].get("fixes_applied"):
        diagnostics["fixes_applied"].extend(diagnostics["tests"]["configuration"]["fixes_applied"])
    progress.update(task, completed=100)
    
    # Generate recommendations
    diagnostics["recommendations"] = generate_recommendations(diagnostics["tests"])
    
    return diagnostics

def display_diagnostic_results(diagnostics: Dict[str, Any], detailed: bool = False):
    """Display diagnostic results."""
    
    # Summary
    tests = diagnostics["tests"]
    total_tests = len(tests)
    passed_tests = sum(1 for test in tests.values() if test.get("status") == "pass")
    
    summary_panel = Panel(
        f"""
        [bold cyan]Diagnostic Summary[/bold cyan]
        
        Tests Run: {total_tests}
        Tests Passed: {passed_tests}
        Tests Failed: {total_tests - passed_tests}
        
        Overall Status: {"✅ HEALTHY" if passed_tests == total_tests else "⚠️  ISSUES FOUND"}
        """,
        title="📊 Results",
        border_style="green" if passed_tests == total_tests else "yellow"
    )
    console.print(summary_panel)
    
    # Test results table
    results_table = Table(show_header=True, header_style="bold magenta")
    results_table.add_column("Test", style="cyan")
    results_table.add_column("Status", style="green")
    results_table.add_column("Score", style="yellow")
    results_table.add_column("Issues", style="red")
    
    for test_name, result in tests.items():
        status = "✅ PASS" if result.get("status") == "pass" else "❌ FAIL"
        score = f"{result.get('score', 0)}/100"
        issues = str(len(result.get('issues', [])))
        
        results_table.add_row(
            test_name.replace("_", " ").title(),
            status,
            score,
            issues
        )
    
    console.print("\n📋 Test Results:")
    console.print(results_table)
    
    # Show fixes applied
    if diagnostics.get("fixes_applied"):
        console.print("\n🔧 [bold green]Fixes Applied:[/bold green]")
        for fix in diagnostics["fixes_applied"]:
            console.print(f"   • {fix}")
    
    # Show recommendations
    if diagnostics.get("recommendations"):
        console.print("\n💡 [bold blue]Recommendations:[/bold blue]")
        for rec in diagnostics["recommendations"][:5]:  # Show top 5
            console.print(f"   • {rec}")
    
    # Detailed results
    if detailed:
        console.print("\n📝 [bold blue]Detailed Results:[/bold blue]")
        for test_name, result in tests.items():
            if result.get("issues"):
                console.print(f"\n[bold red]{test_name.replace('_', ' ').title()} Issues:[/bold red]")
                for issue in result["issues"]:
                    console.print(f"   • {issue}")

@system.command()
@click.option("--profile", type=click.Choice(["beginner", "intermediate", "advanced", "expert"]),
              help="Set user profile")
@click.option("--list", "list_config", is_flag=True, help="List current configuration")
def config(profile, list_config):
    """Manage system configuration and user preferences."""
    ctx = get_context()
    
    console.print("⚙️  [bold blue]System Configuration[/bold blue]\n")
    
    if list_config:
        display_current_config(ctx)
        return
    
    if profile:
        ctx.profile = profile
        ctx.save_preferences()
        console.print(f"✅ Profile set to: [green]{profile}[/green]")
        
        # Show profile details
        profile_config = ctx.get_profile_config()
        config_table = Table(show_header=True, header_style="bold magenta")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")
        
        for key, value in profile_config.items():
            config_table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(f"\n📋 {profile.title()} Profile Settings:")
        console.print(config_table)
    else:
        # Interactive configuration
        interactive_configuration(ctx)

def display_current_config(ctx):
    """Display current configuration."""
    
    config_table = Table(show_header=True, header_style="bold magenta")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_table.add_row("Profile", ctx.profile)
    config_table.add_row("Verbose", str(ctx.verbose))
    config_table.add_row("Debug", str(ctx.debug))
    config_table.add_row("Config Directory", str(ctx.config_dir))
    config_table.add_row("Data Directory", str(ctx.data_dir))
    config_table.add_row("Models Directory", str(ctx.models_dir))
    
    console.print("📋 Current Configuration:")
    console.print(config_table)

def interactive_configuration(ctx):
    """Interactive configuration setup."""
    console.print("🔧 [bold blue]Interactive Configuration[/bold blue]\n")
    
    # Profile selection
    profiles = ["beginner", "intermediate", "advanced", "expert"]
    current_profile = ctx.profile
    
    console.print(f"Current profile: [green]{current_profile}[/green]")
    
    profile_descriptions = {
        "beginner": "Guided experience, auto-fixes, detailed help",
        "intermediate": "Balanced guidance, some automation",
        "advanced": "Minimal guidance, manual control",
        "expert": "No guidance, full manual control"
    }
    
    console.print("\nAvailable profiles:")
    for profile in profiles:
        status = " (current)" if profile == current_profile else ""
        console.print(f"  • [cyan]{profile}[/cyan] - {profile_descriptions[profile]}{status}")
    
    new_profile = Prompt.ask(
        "\nSelect profile",
        choices=profiles,
        default=current_profile
    )
    
    if new_profile != current_profile:
        ctx.profile = new_profile
        ctx.save_preferences()
        console.print(f"✅ Profile updated to: [green]{new_profile}[/green]")

# Helper functions

def check_python_environment() -> Dict[str, Any]:
    """Check Python environment."""
    result = {
        "status": "pass",
        "score": 100,
        "issues": []
    }
    
    # Check Python version
    if sys.version_info < (3, 10):
        result["issues"].append(f"Python 3.10+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        result["status"] = "fail"
        result["score"] -= 50
    
    # Check virtual environment
    if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
        result["issues"].append("Virtual environment recommended")
        result["score"] -= 10
    
    return result

def check_dependencies(auto_fix: bool = False) -> Dict[str, Any]:
    """Check and optionally fix dependencies."""
    result = {
        "status": "pass",
        "score": 100,
        "issues": [],
        "fixes_applied": []
    }
    
    required_deps = [
        "torch", "transformers", "datasets", "accelerate", "peft",
        "yaml", "pandas", "numpy", "rich", "click"
    ]
    
    missing_deps = []
    for dep in required_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        result["issues"].append(f"Missing dependencies: {', '.join(missing_deps)}")
        result["status"] = "fail"
        result["score"] = max(0, 100 - len(missing_deps) * 10)
        
        if auto_fix:
            # This would install missing dependencies
            result["fixes_applied"].append(f"Would install: {', '.join(missing_deps)}")
    
    return result

def check_directory_structure(auto_fix: bool = False) -> Dict[str, Any]:
    """Check and optionally fix directory structure."""
    result = {
        "status": "pass",
        "score": 100,
        "issues": [],
        "fixes_applied": []
    }
    
    ctx = get_context()
    required_dirs = [
        ctx.config_dir,
        ctx.data_dir,
        ctx.models_dir,
        ctx.cache_dir,
        ctx.sessions_dir
    ]
    
    missing_dirs = [d for d in required_dirs if not d.exists()]
    
    if missing_dirs:
        result["issues"].append(f"Missing directories: {len(missing_dirs)}")
        result["score"] = max(0, 100 - len(missing_dirs) * 15)
        
        if auto_fix:
            for directory in missing_dirs:
                directory.mkdir(parents=True, exist_ok=True)
                result["fixes_applied"].append(f"Created directory: {directory}")
            result["status"] = "pass"
            result["score"] = 100
        else:
            result["status"] = "fail"
    
    return result

def check_training_data() -> Dict[str, Any]:
    """Check availability of training data."""
    result = {
        "status": "pass",
        "score": 100,
        "issues": []
    }
    
    ctx = get_context()
    sample_data = ctx.data_dir / "samples" / "sample_parallel.csv"
    
    if not sample_data.exists():
        result["issues"].append("No sample training data found")
        result["status"] = "fail"
        result["score"] = 0
    
    return result

def check_model_status() -> Dict[str, Any]:
    """Check status of trained models."""
    result = {
        "status": "pass",
        "score": 100,
        "issues": []
    }
    
    model_count = count_trained_models()
    
    if model_count == 0:
        result["issues"].append("No trained models found")
        result["score"] = 50
    
    return result

def check_hardware_resources() -> Dict[str, Any]:
    """Check hardware resource adequacy."""
    result = {
        "status": "pass",
        "score": 100,
        "issues": []
    }
    
    try:
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        if memory_gb < 4:
            result["issues"].append("Low memory (recommend 8GB+)")
            result["score"] -= 30
        elif memory_gb < 8:
            result["issues"].append("Moderate memory (8GB+ recommended)")
            result["score"] -= 10
        
        cpu_count = psutil.cpu_count()
        if cpu_count < 2:
            result["issues"].append("Low CPU count (2+ cores recommended)")
            result["score"] -= 20
        
    except Exception as e:
        result["issues"].append(f"Hardware check failed: {e}")
        result["score"] -= 50
    
    if result["issues"]:
        result["status"] = "fail" if result["score"] < 70 else "pass"
    
    return result

def check_configuration_files(auto_fix: bool = False) -> Dict[str, Any]:
    """Check and optionally create configuration files."""
    result = {
        "status": "pass",
        "score": 100,
        "issues": [],
        "fixes_applied": []
    }
    
    ctx = get_context()
    
    # Check for essential config files
    config_files = [
        (ctx.config_dir / "training" / "base_config.yaml", "basic training config"),
        (ctx.config_dir / "data" / "preprocessing_config.yaml", "data preprocessing config")
    ]
    
    missing_configs = []
    for config_path, description in config_files:
        if not config_path.exists():
            missing_configs.append((config_path, description))
    
    if missing_configs:
        result["issues"].append(f"Missing config files: {len(missing_configs)}")
        result["score"] = max(0, 100 - len(missing_configs) * 20)
        
        if auto_fix:
            for config_path, description in missing_configs:
                create_default_config(config_path, description)
                result["fixes_applied"].append(f"Created {description}")
            result["status"] = "pass"
            result["score"] = 100
        else:
            result["status"] = "fail"
    
    return result

def create_default_config(config_path: Path, description: str):
    """Create a default configuration file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    if "training" in str(config_path):
        config_content = """# Basic training configuration
model:
  name: "microsoft/DialoGPT-medium"
  cache_dir: "./models/cache"

training:
  epochs: 3
  learning_rate: 1e-4
  batch_size: 2
  gradient_accumulation_steps: 4

lora:
  enabled: true
  r: 16
  lora_alpha: 32

output:
  checkpoint_dir: "./models/checkpoints"
"""
    else:
        config_content = """# Data preprocessing configuration
data_sources:
  english:
    type: "local_csv"
    path: "./data/raw/english.csv"
  toaripi:
    type: "local_csv"
    path: "./data/raw/toaripi.csv"

preprocessing:
  text_cleaning:
    min_length: 10
    max_length: 512
    remove_duplicates: true

output:
  format: "csv"
  encoding: "utf-8"
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)

def generate_recommendations(test_results: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on test results."""
    recommendations = []
    
    for test_name, result in test_results.items():
        if result.get("status") == "fail":
            if test_name == "python_environment":
                recommendations.append("Upgrade to Python 3.10 or newer")
            elif test_name == "dependencies":
                recommendations.append("Install missing dependencies: pip install -r requirements.txt")
            elif test_name == "directories":
                recommendations.append("Run: toaripi system doctor --fix to create missing directories")
            elif test_name == "data":
                recommendations.append("Run: toaripi workflow quickstart to create sample data")
            elif test_name == "models":
                recommendations.append("Train your first model: toaripi model train --interactive")
            elif test_name == "hardware":
                recommendations.append("Consider upgrading memory to 8GB+ for better performance")
            elif test_name == "configuration":
                recommendations.append("Create missing config files with: toaripi system doctor --fix")
    
    # General recommendations
    if not recommendations:
        recommendations.append("System looks good! Try: toaripi workflow quickstart")
    
    return recommendations

def count_trained_models() -> int:
    """Count number of trained models."""
    ctx = get_context()
    models_dir = ctx.models_dir / "hf"
    if not models_dir.exists():
        return 0
    return len([d for d in models_dir.iterdir() if d.is_dir()])

def check_sample_data() -> bool:
    """Check if sample data is available."""
    ctx = get_context()
    sample_data = ctx.data_dir / "samples" / "sample_parallel.csv"
    return sample_data.exists()

def check_config_files() -> bool:
    """Check if essential config files exist."""
    ctx = get_context()
    essential_configs = [
        ctx.config_dir / "training" / "base_config.yaml",
        ctx.config_dir / "data" / "preprocessing_config.yaml"
    ]
    return all(config.exists() for config in essential_configs)

def save_status_report(status: Dict[str, Any], output_path: Path):
    """Save status report to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(status, f, indent=2)

def export_diagnostic_report(diagnostics: Dict[str, Any], output_path: Path):
    """Export diagnostic report to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    
    console.print(f"📄 Diagnostic report exported to: [cyan]{output_path}[/cyan]")

# Import os module for directory checks
import os