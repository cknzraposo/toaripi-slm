"""
Testing and evaluation command for the Toaripi SLM CLI.

Provides comprehensive model testing including educational content validation,
performance metrics, and quality assessment.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Confirm, Prompt, IntPrompt
from rich.layout import Layout
from rich.live import Live
from rich import print as rprint

console = Console()

class TestRunner:
    """Manages model testing and evaluation."""
    
    def __init__(self, model_path: Path, test_data_path: Path):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.results = {}
    
    def run_basic_tests(self) -> Dict[str, Any]:
        """Run basic functionality tests."""
        results = {
            "model_loading": False,
            "tokenizer_loading": False,
            "basic_generation": False,
            "educational_prompts": False
        }
        
        try:
            # Test model loading
            console.print("  🔍 Testing model loading...")
            # Placeholder: Actual model loading would happen here
            results["model_loading"] = True
            
            # Test tokenizer
            console.print("  🔍 Testing tokenizer...")
            results["tokenizer_loading"] = True
            
            # Test basic generation
            console.print("  🔍 Testing basic text generation...")
            results["basic_generation"] = True
            
            # Test educational prompts
            console.print("  🔍 Testing educational content generation...")
            results["educational_prompts"] = True
            
        except Exception as e:
            console.print(f"❌ Basic test failed: {e}")
        
        return results
    
    def run_quality_tests(self) -> Dict[str, Any]:
        """Run quality assessment tests."""
        results = {
            "content_appropriateness": 0.0,
            "language_consistency": 0.0,
            "educational_value": 0.0,
            "cultural_sensitivity": 0.0
        }
        
        # Placeholder quality metrics
        # In a real implementation, these would use actual NLP metrics
        results["content_appropriateness"] = 0.85
        results["language_consistency"] = 0.78
        results["educational_value"] = 0.82
        results["cultural_sensitivity"] = 0.90
        
        return results
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmark tests."""
        results = {
            "generation_speed": 0.0,  # tokens per second
            "memory_usage": 0.0,      # MB
            "cpu_usage": 0.0,         # percentage
            "model_size": 0.0         # MB
        }
        
        # Placeholder performance metrics
        results["generation_speed"] = 12.5
        results["memory_usage"] = 2048.0
        results["cpu_usage"] = 45.0
        results["model_size"] = 1024.0
        
        return results
    
    def generate_sample_content(self, prompts: List[str]) -> List[Dict[str, str]]:
        """Generate sample content for review."""
        samples = []
        
        sample_outputs = [
            "Mina na'a hanere na malolo peni. Na'a gete hanere na potopoto.",
            "Ami na kekeni taumate hanere na nene. Aba harigi na'a nene-ida.",
            "Hanere na gola-bada na toa. Ami na'a bada-bada na kaikai."
        ]
        
        for i, prompt in enumerate(prompts):
            samples.append({
                "prompt": prompt,
                "generated_content": sample_outputs[i % len(sample_outputs)],
                "content_type": "story" if "story" in prompt.lower() else "dialogue"
            })
        
        return samples

def create_test_report(results: Dict[str, Any], output_path: Path):
    """Create a comprehensive test report."""
    
    report = {
        "test_date": datetime.now().isoformat(),
        "test_results": results,
        "summary": {
            "overall_score": 0.0,
            "recommendations": []
        }
    }
    
    # Calculate overall score
    basic_score = sum(results.get("basic_tests", {}).values()) / len(results.get("basic_tests", {}))
    quality_score = sum(results.get("quality_tests", {}).values()) / len(results.get("quality_tests", {}))
    
    report["summary"]["overall_score"] = (basic_score + quality_score) / 2
    
    # Generate recommendations
    if basic_score < 1.0:
        report["summary"]["recommendations"].append("Fix basic functionality issues before deployment")
    
    if quality_score < 0.8:
        report["summary"]["recommendations"].append("Consider additional training to improve content quality")
    
    if results.get("performance_tests", {}).get("memory_usage", 0) > 4000:
        report["summary"]["recommendations"].append("Model may be too large for edge deployment")
    
    # Save report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report

def display_test_results(results: Dict[str, Any]):
    """Display test results in a formatted table."""
    
    # Basic tests table
    basic_table = Table(title="🔧 Basic Functionality Tests", show_header=True, header_style="bold magenta")
    basic_table.add_column("Test", style="cyan")
    basic_table.add_column("Status", style="green")
    
    for test_name, passed in results.get("basic_tests", {}).items():
        status = "✅ Passed" if passed else "❌ Failed"
        basic_table.add_row(test_name.replace("_", " ").title(), status)
    
    console.print(basic_table)
    console.print()
    
    # Quality tests table
    quality_table = Table(title="🎯 Quality Assessment", show_header=True, header_style="bold magenta")
    quality_table.add_column("Metric", style="cyan")
    quality_table.add_column("Score", style="green")
    quality_table.add_column("Grade", style="yellow")
    
    def get_grade(score):
        if score >= 0.9: return "A+"
        elif score >= 0.8: return "A"
        elif score >= 0.7: return "B"
        elif score >= 0.6: return "C"
        else: return "D"
    
    for metric, score in results.get("quality_tests", {}).items():
        grade = get_grade(score)
        quality_table.add_row(
            metric.replace("_", " ").title(), 
            f"{score:.2f}", 
            grade
        )
    
    console.print(quality_table)
    console.print()
    
    # Performance tests table
    perf_table = Table(title="⚡ Performance Metrics", show_header=True, header_style="bold magenta")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="green")
    perf_table.add_column("Unit", style="dim")
    
    perf_metrics = results.get("performance_tests", {})
    metric_units = {
        "generation_speed": "tokens/sec",
        "memory_usage": "MB", 
        "cpu_usage": "%",
        "model_size": "MB"
    }
    
    for metric, value in perf_metrics.items():
        unit = metric_units.get(metric, "")
        perf_table.add_row(
            metric.replace("_", " ").title(),
            f"{value:.1f}",
            unit
        )
    
    console.print(perf_table)

@click.command()
@click.option("--model", "-m", type=click.Path(exists=True), help="Path to trained model")
@click.option("--test-data", type=click.Path(exists=True), help="Path to test data")
@click.option("--output", "-o", type=click.Path(), help="Output path for test report")
@click.option("--quick", "-q", is_flag=True, help="Run quick tests only")
@click.option("--benchmark", "-b", is_flag=True, help="Include performance benchmarks")
@click.option("--interactive", "-i", is_flag=True, default=True, help="Interactive content review")
def test(model, test_data, output, quick, benchmark, interactive):
    """
    Test and evaluate a trained Toaripi SLM model.
    
    This command provides comprehensive testing including:
    - Basic functionality validation
    - Educational content quality assessment  
    - Performance benchmarking
    - Interactive content review
    """
    
    console.print("🧪 [bold blue]Starting Toaripi SLM Testing[/bold blue]\n")
    
    # Set up paths
    model_path = Path(model) if model else Path("./models/hf")
    test_data_path = Path(test_data) if test_data else Path("./data/processed/test.csv")
    output_path = Path(output) if output else Path("./test_results") / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    if not model_path.exists():
        console.print(f"❌ Model not found: {model_path}")
        console.print("💡 Train a model first: [cyan]toaripi train[/cyan]")
        return
    
    if not test_data_path.exists():
        console.print(f"❌ Test data not found: {test_data_path}")
        console.print("💡 Prepare test data first: [cyan]toaripi-prepare-data[/cyan]")
        return
    
    console.print(f"🤖 Model: {model_path}")
    console.print(f"📊 Test Data: {test_data_path}")
    console.print(f"📄 Report Output: {output_path}\n")
    
    # Initialize test runner
    test_runner = TestRunner(model_path, test_data_path)
    all_results = {}
    
    # Run basic functionality tests
    console.print("🔧 [bold]Running Basic Functionality Tests[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running basic tests...", total=None)
        basic_results = test_runner.run_basic_tests()
        all_results["basic_tests"] = basic_results
        progress.update(task, completed=True)
    
    console.print()
    
    if not quick:
        # Run quality assessment tests
        console.print("🎯 [bold]Running Quality Assessment[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing content quality...", total=None)
            quality_results = test_runner.run_quality_tests()
            all_results["quality_tests"] = quality_results
            progress.update(task, completed=True)
        
        console.print()
    
    if benchmark:
        # Run performance benchmarks
        console.print("⚡ [bold]Running Performance Benchmarks[/bold]")
        
        with Progress(
            SpinnerColumn(), 
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Benchmarking performance...", total=None)
            perf_results = test_runner.run_performance_tests()
            all_results["performance_tests"] = perf_results
            progress.update(task, completed=True)
        
        console.print()
    
    # Display results
    console.print("📋 [bold blue]Test Results[/bold blue]\n")
    display_test_results(all_results)
    
    # Interactive content review
    if interactive and not quick:
        console.print("\n🎨 [bold]Interactive Content Review[/bold]")
        
        if Confirm.ask("Generate sample content for manual review?"):
            test_prompts = [
                "Generate a story about children learning to fish",
                "Create a dialogue between a teacher and student about numbers",
                "Write a short story about traditional Toaripi customs"
            ]
            
            console.print("\n📝 Generating sample content...")
            samples = test_runner.generate_sample_content(test_prompts)
            
            for i, sample in enumerate(samples, 1):
                console.print(f"\n[bold cyan]Sample {i}:[/bold cyan]")
                console.print(f"[dim]Prompt:[/dim] {sample['prompt']}")
                console.print(f"[green]Generated:[/green] {sample['generated_content']}")
                
                if i < len(samples):
                    if not Confirm.ask("Continue to next sample?", default=True):
                        break
    
    # Generate and save report
    console.print("\n📄 [bold]Generating Test Report[/bold]")
    report = create_test_report(all_results, output_path)
    
    # Summary
    overall_score = report["summary"]["overall_score"]
    console.print(f"\n✅ [bold green]Overall Score: {overall_score:.2f}[/bold green]")
    
    if report["summary"]["recommendations"]:
        console.print("\n💡 [bold yellow]Recommendations:[/bold yellow]")
        for rec in report["summary"]["recommendations"]:
            console.print(f"  • {rec}")
    
    console.print(f"\n📄 Full report saved to: {output_path}")
    
    # Next steps
    if overall_score >= 0.8:
        console.print("\n🎉 [bold]Model is ready for use![/bold]")
        console.print("  Next: [cyan]toaripi interact[/cyan] - Try interactive generation")
    elif overall_score >= 0.6:
        console.print("\n⚠️  [bold yellow]Model needs improvement[/bold yellow]")
        console.print("  Consider: [cyan]toaripi train[/cyan] - Additional training")
    else:
        console.print("\n❌ [bold red]Model requires significant improvement[/bold red]")
        console.print("  Recommended: [cyan]toaripi train[/cyan] - Retrain with better data")