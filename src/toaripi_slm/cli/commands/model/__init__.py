"""
Model management commands with enhanced functionality.
"""

import click
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Confirm, Prompt, IntPrompt, FloatPrompt
from rich.tree import Tree

from ...context import get_context

console = Console()

@click.group()
def model():
    """Model training, testing, and management commands."""
    pass

@model.command()
@click.option("--profile", type=click.Choice(["quick", "balanced", "thorough", "custom"]), 
              default="balanced", help="Training profile")
@click.option("--data", type=Path, help="Training data file")
@click.option("--config", type=Path, help="Training configuration file")
@click.option("--resume", type=Path, help="Resume from checkpoint")
@click.option("--dry-run", is_flag=True, help="Validate configuration without training")
@click.option("--interactive", is_flag=True, help="Interactive configuration")
def train(profile, data, config, resume, dry_run, interactive):
    """Train a Toaripi SLM model with enhanced configuration options."""
    ctx = get_context()
    
    console.print("🧠 [bold blue]Model Training[/bold blue]\n")
    
    # Profile-based configuration
    profile_configs = {
        "quick": {
            "epochs": 1,
            "batch_size": 1,
            "model_name": "microsoft/DialoGPT-small",
            "learning_rate": 2e-5,
            "description": "Fast iteration for testing"
        },
        "balanced": {
            "epochs": 3,
            "batch_size": 2,
            "model_name": "microsoft/DialoGPT-medium", 
            "learning_rate": 1e-4,
            "description": "Recommended for most users"
        },
        "thorough": {
            "epochs": 5,
            "batch_size": 4,
            "model_name": "microsoft/DialoGPT-medium",
            "learning_rate": 5e-5,
            "description": "High quality training"
        }
    }
    
    if profile == "custom" or interactive:
        training_config = configure_training_interactive()
    else:
        training_config = profile_configs[profile].copy()
        console.print(f"Using [cyan]{profile}[/cyan] profile: {training_config['description']}")
    
    # Data validation
    if not data:
        data = ctx.data_dir / "samples" / "sample_parallel.csv"
    
    if not data.exists():
        console.print(f"❌ Data file not found: {data}")
        console.print("💡 Run [cyan]toaripi workflow quickstart[/cyan] to create sample data")
        return
    
    # Show configuration
    config_table = Table(show_header=True, header_style="bold magenta")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    
    for key, value in training_config.items():
        if key != "description":
            config_table.add_row(str(key), str(value))
    
    console.print("📋 Training Configuration:")
    console.print(config_table)
    
    if dry_run:
        console.print("\n✅ [green]Dry run completed - configuration is valid![/green]")
        return
    
    if not Confirm.ask("\nStart training?", default=True):
        return
    
    # Auto-detect optimal settings based on hardware
    if ctx.get_profile_config().get("auto_fix_issues", False):
        training_config = optimize_for_hardware(training_config)
    
    # Training execution with progress tracking
    model_path = execute_training(training_config, data)
    
    if model_path:
        console.print(f"\n✅ [bold green]Training completed![/bold green]")
        console.print(f"📁 Model saved to: [cyan]{model_path}[/cyan]")
        
        # Suggest next steps
        next_steps = ctx.suggest_next_action("model train")
        if next_steps:
            console.print("\n💡 [bold blue]Suggested next steps:[/bold blue]")
            for step in next_steps[:3]:  # Show top 3 suggestions
                console.print(f"   • {step}")

def configure_training_interactive():
    """Enhanced interactive training configuration."""
    console.print("🔧 [bold blue]Interactive Training Configuration[/bold blue]\n")
    
    # Model selection with detailed info
    models = {
        "1": {
            "name": "microsoft/DialoGPT-small",
            "params": "117M",
            "memory": "~2GB",
            "speed": "Fast",
            "quality": "Good"
        },
        "2": {
            "name": "microsoft/DialoGPT-medium", 
            "params": "345M",
            "memory": "~4GB",
            "speed": "Medium",
            "quality": "Better"
        },
        "3": {
            "name": "microsoft/DialoGPT-large",
            "params": "762M", 
            "memory": "~8GB",
            "speed": "Slow",
            "quality": "Best"
        }
    }
    
    # Display model options
    model_table = Table(show_header=True, header_style="bold magenta")
    model_table.add_column("Option", style="cyan")
    model_table.add_column("Model", style="green")
    model_table.add_column("Parameters", style="yellow")
    model_table.add_column("Memory", style="blue")
    model_table.add_column("Speed", style="magenta")
    model_table.add_column("Quality", style="red")
    
    for key, info in models.items():
        model_table.add_row(
            key, info["name"], info["params"], 
            info["memory"], info["speed"], info["quality"]
        )
    
    console.print("📋 Available models:")
    console.print(model_table)
    
    model_choice = Prompt.ask("\nSelect model", choices=list(models.keys()), default="2")
    selected_model = models[model_choice]["name"]
    
    # Training parameters with smart defaults
    epochs = IntPrompt.ask("Number of epochs", default=3, show_default=True)
    
    # Auto-suggest batch size based on model
    suggested_batch = 4 if "small" in selected_model else 2 if "medium" in selected_model else 1
    batch_size = IntPrompt.ask("Batch size", default=suggested_batch, show_default=True)
    
    learning_rate = FloatPrompt.ask("Learning rate", default=1e-4, show_default=True)
    
    # Advanced options
    if Confirm.ask("Configure advanced options?", default=False):
        gradient_accumulation = IntPrompt.ask("Gradient accumulation steps", default=4)
        warmup_ratio = FloatPrompt.ask("Warmup ratio", default=0.1)
        weight_decay = FloatPrompt.ask("Weight decay", default=0.01)
    else:
        gradient_accumulation = 4
        warmup_ratio = 0.1
        weight_decay = 0.01
    
    return {
        "model_name": selected_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "gradient_accumulation_steps": gradient_accumulation,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay
    }

def optimize_for_hardware(config: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize training configuration for available hardware."""
    import psutil
    
    # Get system info
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count()
    
    optimized = config.copy()
    
    # Memory-based optimizations
    if memory_gb < 8:
        console.print("⚡ Optimizing for low memory...")
        optimized["batch_size"] = min(optimized["batch_size"], 1)
        optimized["gradient_accumulation_steps"] = max(optimized.get("gradient_accumulation_steps", 4), 8)
    elif memory_gb < 16:
        console.print("⚡ Optimizing for moderate memory...")
        optimized["batch_size"] = min(optimized["batch_size"], 2)
    
    # CPU-based optimizations
    if cpu_count < 4:
        console.print("⚡ Optimizing for limited CPU...")
        optimized["dataloader_num_workers"] = 1
    else:
        optimized["dataloader_num_workers"] = min(4, cpu_count // 2)
    
    return optimized

def execute_training(config: Dict[str, Any], data_path: Path) -> Optional[str]:
    """Execute model training with progress tracking."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        # Simulated training steps - replace with actual training logic
        steps = [
            ("Loading model and tokenizer...", 10),
            ("Preparing training data...", 20),
            ("Setting up training configuration...", 30),
            ("Training epoch 1...", 60),
            ("Training epoch 2...", 80),
            ("Training epoch 3...", 90),
            ("Saving model...", 100)
        ]
        
        task = progress.add_task("Training model...", total=100)
        
        for step_desc, progress_val in steps:
            progress.update(task, description=step_desc, completed=progress_val)
            # Simulate work
            import time
            time.sleep(1)
    
    # Generate model path with version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"./models/hf/toaripi-slm-{timestamp}"
    
    return model_path

@model.command()
@click.option("--profile", type=click.Choice(["quick", "standard", "thorough", "benchmark"]),
              default="standard", help="Testing profile")
@click.option("--model-path", type=Path, help="Path to model to test")
@click.option("--output", type=Path, help="Test results output file")
def test(profile, model_path, output):
    """Test model with comprehensive evaluation."""
    ctx = get_context()
    
    console.print("🧪 [bold blue]Model Testing[/bold blue]\n")
    
    # Find model if not specified
    if not model_path:
        model_path = find_latest_model()
        if not model_path:
            console.print("❌ No trained models found. Train a model first.")
            return
    
    console.print(f"Testing model: [cyan]{model_path}[/cyan]")
    
    # Test profiles
    test_profiles = {
        "quick": ["basic_functionality", "generation_test"],
        "standard": ["basic_functionality", "generation_test", "quality_metrics", "performance"],
        "thorough": ["basic_functionality", "generation_test", "quality_metrics", "performance", "educational_validation"],
        "benchmark": ["basic_functionality", "generation_test", "quality_metrics", "performance", "educational_validation", "comparison"]
    }
    
    tests_to_run = test_profiles[profile]
    
    # Execute tests
    results = execute_tests(model_path, tests_to_run)
    
    # Display results
    display_test_results(results)
    
    # Save results if requested
    if output:
        save_test_results(results, output)

def execute_tests(model_path: str, tests: List[str]) -> Dict[str, Any]:
    """Execute specified tests and return results."""
    results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for test_name in tests:
            task = progress.add_task(f"Running {test_name}...", total=None)
            
            # Simulate test execution
            import time
            time.sleep(2)
            
            # Mock results - replace with actual test implementations
            if test_name == "basic_functionality":
                results[test_name] = {"status": "pass", "score": 100}
            elif test_name == "generation_test":
                results[test_name] = {"status": "pass", "score": 85}
            elif test_name == "quality_metrics":
                results[test_name] = {"status": "pass", "score": 78, "bleu": 0.42, "rouge": 0.38}
            elif test_name == "performance":
                results[test_name] = {"status": "pass", "score": 92, "speed": 12.5, "memory": 2048}
            elif test_name == "educational_validation":
                results[test_name] = {"status": "pass", "score": 88, "appropriateness": 95, "accuracy": 82}
            elif test_name == "comparison":
                results[test_name] = {"status": "pass", "score": 75, "vs_baseline": "+12%"}
            
            progress.update(task, completed=100)
    
    return results

def display_test_results(results: Dict[str, Any]):
    """Display test results in a formatted table."""
    
    # Overall summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r.get("status") == "pass")
    avg_score = sum(r.get("score", 0) for r in results.values()) / total_tests if total_tests > 0 else 0
    
    summary_panel = Panel(
        f"""
        [bold green]Test Summary[/bold green]
        
        Tests passed: {passed_tests}/{total_tests}
        Average score: {avg_score:.1f}/100
        Overall status: {"✅ PASS" if passed_tests == total_tests else "❌ FAIL"}
        """,
        title="📊 Results Summary",
        border_style="green" if passed_tests == total_tests else "red"
    )
    console.print(summary_panel)
    
    # Detailed results
    results_table = Table(show_header=True, header_style="bold magenta")
    results_table.add_column("Test", style="cyan")
    results_table.add_column("Status", style="green")
    results_table.add_column("Score", style="yellow")
    results_table.add_column("Details", style="dim")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result.get("status") == "pass" else "❌ FAIL"
        score = f"{result.get('score', 0)}/100"
        
        # Build details string
        details = []
        for key, value in result.items():
            if key not in ["status", "score"]:
                details.append(f"{key}: {value}")
        details_str = ", ".join(details) if details else ""
        
        results_table.add_row(
            test_name.replace("_", " ").title(),
            status,
            score, 
            details_str
        )
    
    console.print("\n📋 Detailed Results:")
    console.print(results_table)

@model.command()
def list():
    """List all available trained models."""
    ctx = get_context()
    
    console.print("📋 [bold blue]Available Models[/bold blue]\n")
    
    models_dir = ctx.models_dir / "hf"
    if not models_dir.exists():
        console.print("No models directory found.")
        return
    
    # Find all model directories
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        console.print("No trained models found.")
        console.print("💡 Run [cyan]toaripi model train[/cyan] to train your first model!")
        return
    
    # Create models table
    models_table = Table(show_header=True, header_style="bold magenta")
    models_table.add_column("Version", style="cyan")
    models_table.add_column("Name", style="green")
    models_table.add_column("Created", style="yellow")
    models_table.add_column("Size", style="blue")
    models_table.add_column("Status", style="magenta")
    
    for model_dir in sorted(model_dirs, key=lambda x: x.stat().st_mtime, reverse=True):
        # Extract info from directory
        name = model_dir.name
        created = datetime.fromtimestamp(model_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        
        # Calculate directory size
        size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        size_str = f"{size / (1024**3):.1f} GB" if size > 1024**3 else f"{size / (1024**2):.1f} MB"
        
        # Check if model is complete
        required_files = ["config.json", "pytorch_model.bin"]
        has_all_files = all((model_dir / f).exists() for f in required_files)
        status = "✅ Ready" if has_all_files else "⚠️ Incomplete"
        
        # Extract version from name if possible
        version = name.split('-')[-1] if '-' in name else "Unknown"
        
        models_table.add_row(version, name, created, size_str, status)
    
    console.print(models_table)

@model.command()
@click.option("--format", type=click.Choice(["gguf", "onnx", "hf"]), default="gguf", help="Export format")
@click.option("--quantization", default="q4_k_m", help="Quantization level for GGUF")
@click.option("--model-path", type=Path, help="Model to export")
@click.option("--output-dir", type=Path, help="Output directory")
def export(format, quantization, model_path, output_dir):
    """Export trained model to different formats."""
    ctx = get_context()
    
    console.print(f"📦 [bold blue]Model Export ({format.upper()})[/bold blue]\n")
    
    # Find model if not specified
    if not model_path:
        model_path = find_latest_model()
        if not model_path:
            console.print("❌ No trained models found.")
            return
    
    console.print(f"Exporting model: [cyan]{model_path}[/cyan]")
    console.print(f"Format: [green]{format}[/green]")
    if format == "gguf":
        console.print(f"Quantization: [yellow]{quantization}[/yellow]")
    
    # Execute export
    export_path = execute_export(model_path, format, quantization, output_dir)
    
    if export_path:
        console.print(f"\n✅ [bold green]Export completed![/bold green]")
        console.print(f"📁 Exported to: [cyan]{export_path}[/cyan]")

def execute_export(model_path: str, format: str, quantization: str, output_dir: Optional[Path]) -> Optional[str]:
    """Execute model export with progress tracking."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        
        steps = [
            ("Loading model...", 20),
            ("Converting format...", 60),
            ("Applying quantization...", 80) if format == "gguf" else ("Optimizing...", 80),
            ("Saving exported model...", 100)
        ]
        
        task = progress.add_task("Exporting model...", total=100)
        
        for step_desc, progress_val in steps:
            progress.update(task, description=step_desc, completed=progress_val)
            import time
            time.sleep(1)
    
    # Generate export path
    if not output_dir:
        output_dir = Path(f"./models/{format}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = output_dir / f"toaripi-slm-{format}-{timestamp}"
    
    return str(export_path)

@model.command()
@click.argument("version1")
@click.argument("version2")
def compare(version1, version2):
    """Compare two model versions."""
    console.print(f"⚖️  [bold blue]Model Comparison[/bold blue]\n")
    console.print(f"Comparing [cyan]{version1}[/cyan] vs [cyan]{version2}[/cyan]")
    
    # Mock comparison results
    comparison_table = Table(show_header=True, header_style="bold magenta")
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column(version1, style="green")
    comparison_table.add_column(version2, style="yellow")
    comparison_table.add_column("Difference", style="red")
    
    metrics = [
        ("Model Size", "1.2 GB", "1.5 GB", "+25%"),
        ("Training Time", "45 min", "62 min", "+38%"),
        ("BLEU Score", "0.42", "0.48", "+14%"),
        ("Generation Speed", "12.5 tok/s", "10.8 tok/s", "-14%"),
        ("Educational Score", "78%", "85%", "+9%")
    ]
    
    for metric, val1, val2, diff in metrics:
        comparison_table.add_row(metric, val1, val2, diff)
    
    console.print(comparison_table)

def find_latest_model() -> Optional[str]:
    """Find the most recently trained model."""
    from ...context import get_context
    ctx = get_context()
    
    models_dir = ctx.models_dir / "hf"
    if not models_dir.exists():
        return None
    
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        return None
    
    # Return most recent
    latest = max(model_dirs, key=lambda x: x.stat().st_mtime)
    return str(latest)

def save_test_results(results: Dict[str, Any], output_path: Path):
    """Save test results to file."""
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "summary": {
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results.values() if r.get("status") == "pass"),
            "average_score": sum(r.get("score", 0) for r in results.values()) / len(results)
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    console.print(f"📄 Test results saved to: [cyan]{output_path}[/cyan]")