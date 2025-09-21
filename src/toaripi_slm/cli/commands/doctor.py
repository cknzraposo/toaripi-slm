"""
Doctor command for the Toaripi SLM CLI.

Provides comprehensive system diagnostics, troubleshooting guidance,
and health checks for the Toaripi SLM environment.
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree
from rich.text import Text
from rich import print as rprint

console = Console()

class SystemDoctor:
    """Comprehensive system health checker for Toaripi SLM."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.recommendations = []
    
    def check_python_environment(self) -> Dict[str, Any]:
        """Check Python version and environment."""
        
        results = {
            "python_version": sys.version_info,
            "python_executable": sys.executable,
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
            "issues": []
        }
        
        # Check Python version
        if sys.version_info < (3, 10):
            results["issues"].append("Python 3.10+ required, found {}.{}".format(
                sys.version_info.major, sys.version_info.minor
            ))
            self.issues.append("Upgrade Python to 3.10 or later")
        
        # Check virtual environment
        if not results["virtual_env"]:
            self.warnings.append("Not running in virtual environment")
            self.recommendations.append("Create and activate a virtual environment")
        
        return results
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check required and optional dependencies."""
        
        required_deps = [
            "torch", "transformers", "datasets", "accelerate", 
            "peft", "yaml", "pandas", "numpy", "click", "rich"
        ]
        
        optional_deps = [
            "wandb", "tensorboard", "jupyter", "matplotlib", 
            "seaborn", "scikit-learn", "llama-cpp-python"
        ]
        
        results = {
            "required": {},
            "optional": {},
            "missing_required": [],
            "missing_optional": []
        }
        
        # Check required dependencies
        for dep in required_deps:
            try:
                module = __import__(dep.replace("-", "_"))
                version = getattr(module, "__version__", "unknown")
                results["required"][dep] = {"installed": True, "version": version}
            except ImportError:
                results["required"][dep] = {"installed": False, "version": None}
                results["missing_required"].append(dep)
                self.issues.append(f"Missing required dependency: {dep}")
        
        # Check optional dependencies
        for dep in optional_deps:
            try:
                module = __import__(dep.replace("-", "_"))
                version = getattr(module, "__version__", "unknown")
                results["optional"][dep] = {"installed": True, "version": version}
            except ImportError:
                results["optional"][dep] = {"installed": False, "version": None}
                results["missing_optional"].append(dep)
        
        return results
    
    def check_gpu_availability(self) -> Dict[str, Any]:
        """Check GPU availability and CUDA setup."""
        
        results = {
            "cuda_available": False,
            "cuda_version": None,
            "gpu_count": 0,
            "gpu_names": [],
            "gpu_memory": [],
            "recommendations": []
        }
        
        try:
            import torch
            results["cuda_available"] = torch.cuda.is_available()
            
            if results["cuda_available"]:
                results["cuda_version"] = torch.version.cuda
                results["gpu_count"] = torch.cuda.device_count()
                
                for i in range(results["gpu_count"]):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    results["gpu_names"].append(gpu_name)
                    results["gpu_memory"].append(f"{gpu_memory:.1f}GB")
                
                # GPU recommendations
                total_memory = sum([
                    torch.cuda.get_device_properties(i).total_memory 
                    for i in range(results["gpu_count"])
                ]) / (1024**3)
                
                if total_memory < 4:
                    self.warnings.append("Limited GPU memory may affect training")
                    results["recommendations"].append("Consider using CPU training or model quantization")
                elif total_memory < 8:
                    results["recommendations"].append("Consider using LoRA fine-tuning for efficiency")
                
            else:
                self.warnings.append("CUDA not available - will use CPU training")
                results["recommendations"].append("Install CUDA for faster training")
                
        except ImportError:
            self.issues.append("PyTorch not installed - cannot check GPU")
        
        return results
    
    def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        
        results = {
            "total_space": 0,
            "available_space": 0,
            "used_space": 0,
            "usage_percent": 0,
            "warnings": []
        }
        
        try:
            if platform.system() == "Windows":
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                total_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p("."), 
                    ctypes.pointer(free_bytes), 
                    ctypes.pointer(total_bytes), 
                    None
                )
                results["available_space"] = free_bytes.value / (1024**3)
                results["total_space"] = total_bytes.value / (1024**3)
            else:
                statvfs = os.statvfs(".")
                results["available_space"] = (statvfs.f_frsize * statvfs.f_avail) / (1024**3)
                results["total_space"] = (statvfs.f_frsize * statvfs.f_blocks) / (1024**3)
            
            results["used_space"] = results["total_space"] - results["available_space"]
            results["usage_percent"] = (results["used_space"] / results["total_space"]) * 100
            
            # Check space requirements
            if results["available_space"] < 5:
                self.issues.append("Insufficient disk space (< 5GB available)")
                results["warnings"].append("At least 5GB recommended for model training")
            elif results["available_space"] < 10:
                self.warnings.append("Limited disk space may affect large model training")
                results["warnings"].append("Consider freeing up space or using smaller models")
                
        except Exception as e:
            self.warnings.append(f"Could not check disk space: {e}")
        
        return results
    
    def check_project_structure(self) -> Dict[str, Any]:
        """Check project directory structure and files."""
        
        expected_structure = {
            "configs": ["training/base_config.yaml", "training/lora_config.yaml"],
            "data": ["processed/", "samples/"],
            "models": ["cache/", "hf/", "gguf/"],
            "src/toaripi_slm": ["core/", "data/", "inference/", "utils/"],
            "scripts": [],
            "tests": []
        }
        
        results = {
            "project_root_found": False,
            "structure_status": {},
            "missing_dirs": [],
            "missing_files": [],
            "config_files": {}
        }
        
        # Check if we're in a Toaripi SLM project
        if Path("setup.py").exists() or Path("pyproject.toml").exists():
            results["project_root_found"] = True
        
        # Check directory structure
        for dir_path, expected_files in expected_structure.items():
            dir_exists = Path(dir_path).exists()
            results["structure_status"][dir_path] = dir_exists
            
            if not dir_exists:
                results["missing_dirs"].append(dir_path)
                self.warnings.append(f"Missing directory: {dir_path}")
            
            # Check expected files within directories
            for file_path in expected_files:
                full_path = Path(dir_path) / file_path
                if not full_path.exists():
                    results["missing_files"].append(str(full_path))
        
        # Check configuration files
        config_files = [
            "configs/training/base_config.yaml",
            "configs/training/lora_config.yaml",
            "configs/data/preprocessing_config.yaml"
        ]
        
        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path, "r") as f:
                        config_data = yaml.safe_load(f)
                    results["config_files"][config_file] = {"valid": True, "keys": list(config_data.keys())}
                except Exception as e:
                    results["config_files"][config_file] = {"valid": False, "error": str(e)}
                    self.issues.append(f"Invalid config file: {config_file}")
            else:
                results["config_files"][config_file] = {"valid": False, "error": "File not found"}
        
        return results
    
    def check_data_availability(self) -> Dict[str, Any]:
        """Check training data availability and quality."""
        
        results = {
            "raw_data_exists": False,
            "processed_data_exists": False,
            "data_files": {},
            "data_quality": {}
        }
        
        # Check raw data
        raw_data_dir = Path("data/raw")
        if raw_data_dir.exists() and any(raw_data_dir.iterdir()):
            results["raw_data_exists"] = True
        
        # Check processed data
        processed_data_dir = Path("data/processed")
        if processed_data_dir.exists():
            results["processed_data_exists"] = True
            
            # Check specific data files
            data_files = ["train.csv", "validation.csv", "test.csv"]
            for file_name in data_files:
                file_path = processed_data_dir / file_name
                if file_path.exists():
                    try:
                        import pandas as pd
                        df = pd.read_csv(file_path)
                        results["data_files"][file_name] = {
                            "exists": True,
                            "rows": len(df),
                            "columns": list(df.columns)
                        }
                        
                        # Basic quality checks
                        if "english" in df.columns and "toaripi" in df.columns:
                            empty_english = df["english"].isna().sum()
                            empty_toaripi = df["toaripi"].isna().sum()
                            results["data_quality"][file_name] = {
                                "empty_english": empty_english,
                                "empty_toaripi": empty_toaripi,
                                "quality_score": 1 - (empty_english + empty_toaripi) / (len(df) * 2)
                            }
                    except Exception as e:
                        results["data_files"][file_name] = {"exists": True, "error": str(e)}
                        self.issues.append(f"Cannot read data file: {file_name}")
                else:
                    results["data_files"][file_name] = {"exists": False}
                    self.warnings.append(f"Missing data file: {file_name}")
        
        if not results["processed_data_exists"]:
            self.issues.append("No processed training data found")
            self.recommendations.append("Run data preprocessing: toaripi-prepare-data")
        
        return results
    
    def generate_health_score(self) -> Tuple[float, str]:
        """Generate an overall health score."""
        
        # Calculate score based on issues and warnings
        base_score = 100.0
        
        # Deduct points for issues and warnings
        base_score -= len(self.issues) * 15  # Major issues
        base_score -= len(self.warnings) * 5  # Minor warnings
        
        # Ensure score is between 0 and 100
        score = max(0.0, min(100.0, base_score))
        
        # Determine health status
        if score >= 90:
            status = "Excellent"
        elif score >= 75:
            status = "Good"
        elif score >= 60:
            status = "Fair"
        elif score >= 40:
            status = "Poor"
        else:
            status = "Critical"
        
        return score, status

def display_system_info(info: Dict[str, Any]):
    """Display system information in a formatted table."""
    
    table = Table(title="🖥️  System Information", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Status", style="yellow")
    
    # Python info
    python_info = info["python"]
    python_version = f"{python_info['python_version'][0]}.{python_info['python_version'][1]}.{python_info['python_version'][2]}"
    python_status = "✅ OK" if python_info['python_version'] >= (3, 10) else "❌ Upgrade needed"
    table.add_row("Python Version", python_version, python_status)
    
    venv_status = "✅ Active" if python_info["virtual_env"] else "⚠️ Not active"
    table.add_row("Virtual Environment", python_info["virtual_env"] or "None", venv_status)
    
    table.add_row("Platform", python_info["platform"], "✅ Supported")
    table.add_row("Architecture", python_info["architecture"], "✅ Compatible")
    
    # GPU info
    gpu_info = info["gpu"]
    if gpu_info["cuda_available"]:
        gpu_status = f"✅ {gpu_info['gpu_count']} GPU(s)"
        gpu_details = f"CUDA {gpu_info['cuda_version']}"
        for i, (name, memory) in enumerate(zip(gpu_info["gpu_names"], gpu_info["gpu_memory"])):
            table.add_row(f"GPU {i}", f"{name} ({memory})", gpu_status if i == 0 else "")
    else:
        table.add_row("GPU", "None detected", "⚠️ CPU only")
    
    # Disk space
    disk_info = info["disk"]
    disk_status = "✅ Sufficient" if disk_info["available_space"] > 10 else "⚠️ Limited"
    table.add_row("Available Disk", f"{disk_info['available_space']:.1f}GB", disk_status)
    
    console.print(table)

def display_dependencies(deps: Dict[str, Any]):
    """Display dependency status."""
    
    # Required dependencies
    req_table = Table(title="📦 Required Dependencies", show_header=True, header_style="bold magenta")
    req_table.add_column("Package", style="cyan")
    req_table.add_column("Status", style="green")
    req_table.add_column("Version", style="dim")
    
    for dep, info in deps["required"].items():
        status = "✅ Installed" if info["installed"] else "❌ Missing"
        version = info["version"] if info["installed"] else "N/A"
        req_table.add_row(dep, status, version)
    
    console.print(req_table)
    
    # Optional dependencies (only if any are installed)
    installed_optional = {k: v for k, v in deps["optional"].items() if v["installed"]}
    if installed_optional:
        opt_table = Table(title="🔧 Optional Dependencies", show_header=True, header_style="bold blue")
        opt_table.add_column("Package", style="cyan")
        opt_table.add_column("Status", style="green")
        opt_table.add_column("Version", style="dim")
        
        for dep, info in installed_optional.items():
            opt_table.add_row(dep, "✅ Installed", info["version"])
        
        console.print(opt_table)

def display_project_structure(structure: Dict[str, Any]):
    """Display project structure status."""
    
    tree = Tree("📁 Project Structure")
    
    for dir_path, exists in structure["structure_status"].items():
        status_icon = "✅" if exists else "❌"
        dir_node = tree.add(f"{status_icon} {dir_path}")
        
        # Add details for important directories
        if dir_path == "data" and exists:
            data_info = structure.get("data_files", {})
            for file_name, file_info in data_info.items():
                if file_info.get("exists"):
                    file_node = dir_node.add(f"✅ {file_name}")
                    if "rows" in file_info:
                        file_node.add(f"📊 {file_info['rows']} rows")
                else:
                    dir_node.add(f"❌ {file_name}")
    
    console.print(tree)

@click.command()
@click.option("--detailed", "-d", is_flag=True, help="Show detailed diagnostic information")
@click.option("--fix", "-f", is_flag=True, help="Attempt to fix common issues automatically")
@click.option("--export-report", type=click.Path(), help="Export diagnostic report to file")
def doctor(detailed, fix, export_report):
    """
    Comprehensive system health check and troubleshooting.
    
    This command analyzes your Toaripi SLM environment and provides:
    - System compatibility checks
    - Dependency validation
    - Project structure verification
    - Performance recommendations
    - Troubleshooting guidance
    """
    
    console.print("🩺 [bold blue]Toaripi SLM System Doctor[/bold blue]\n")
    console.print("Running comprehensive system diagnostics...\n")
    
    doctor = SystemDoctor()
    
    # Run all diagnostic checks
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # System environment
        task1 = progress.add_task("Checking Python environment...", total=None)
        python_info = doctor.check_python_environment()
        progress.update(task1, completed=True)
        
        # Dependencies
        task2 = progress.add_task("Checking dependencies...", total=None)
        deps_info = doctor.check_dependencies()
        progress.update(task2, completed=True)
        
        # GPU availability
        task3 = progress.add_task("Checking GPU availability...", total=None)
        gpu_info = doctor.check_gpu_availability()
        progress.update(task3, completed=True)
        
        # Disk space
        task4 = progress.add_task("Checking disk space...", total=None)
        disk_info = doctor.check_disk_space()
        progress.update(task4, completed=True)
        
        # Project structure
        task5 = progress.add_task("Checking project structure...", total=None)
        structure_info = doctor.check_project_structure()
        progress.update(task5, completed=True)
        
        # Data availability
        task6 = progress.add_task("Checking data availability...", total=None)
        data_info = doctor.check_data_availability()
        progress.update(task6, completed=True)
    
    console.print()
    
    # Generate health score
    health_score, health_status = doctor.generate_health_score()
    
    # Display health summary
    health_color = "green" if health_score >= 75 else "yellow" if health_score >= 50 else "red"
    console.print(Panel(
        f"Overall Health Score: [bold {health_color}]{health_score:.0f}/100[/bold {health_color}] ({health_status})",
        title="🏥 Health Summary",
        border_style=health_color
    ))
    
    console.print()
    
    # Display detailed information
    all_info = {
        "python": python_info,
        "gpu": gpu_info,
        "disk": disk_info
    }
    
    display_system_info(all_info)
    console.print()
    
    display_dependencies(deps_info)
    console.print()
    
    if detailed:
        display_project_structure(structure_info)
        console.print()
    
    # Display issues and recommendations
    if doctor.issues:
        console.print("❌ [bold red]Critical Issues:[/bold red]")
        for issue in doctor.issues:
            console.print(f"  • {issue}")
        console.print()
    
    if doctor.warnings:
        console.print("⚠️  [bold yellow]Warnings:[/bold yellow]")
        for warning in doctor.warnings:
            console.print(f"  • {warning}")
        console.print()
    
    if doctor.recommendations:
        console.print("💡 [bold blue]Recommendations:[/bold blue]")
        for rec in doctor.recommendations:
            console.print(f"  • {rec}")
        console.print()
    
    # Auto-fix option
    if fix and doctor.issues:
        console.print("🔧 [bold]Attempting to fix issues...[/bold]")
        
        # Placeholder for auto-fix functionality
        if deps_info["missing_required"]:
            console.print("  📦 Installing missing dependencies...")
            console.print("  Run: [cyan]pip install -r requirements.txt[/cyan]")
        
        console.print("  Auto-fix functionality will be implemented in future versions.")
    
    # Export report
    if export_report:
        report_data = {
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "health_score": health_score,
            "health_status": health_status,
            "system_info": all_info,
            "dependencies": deps_info,
            "project_structure": structure_info,
            "data_info": data_info,
            "issues": doctor.issues,
            "warnings": doctor.warnings,
            "recommendations": doctor.recommendations
        }
        
        import json
        with open(export_report, "w") as f:
            json.dump(report_data, f, indent=2)
        
        console.print(f"📄 Diagnostic report exported to: {export_report}")
    
    # Final recommendations
    console.print("\n🎯 [bold]Next Steps:[/bold]")
    
    if health_score >= 90:
        console.print("  🎉 Your system is ready for Toaripi SLM!")
        console.print("  Try: [cyan]toaripi train --interactive[/cyan]")
    elif health_score >= 75:
        console.print("  ✅ System is mostly ready, address warnings if needed")
        console.print("  Try: [cyan]toaripi status --detailed[/cyan]")
    else:
        console.print("  🔧 Address critical issues before proceeding")
        console.print("  Check: [cyan]pip install -r requirements.txt[/cyan]")
        console.print("  Review: Project setup documentation")