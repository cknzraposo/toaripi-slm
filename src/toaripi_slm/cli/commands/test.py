#!/usr/bin/env python3
"""
Test/Evaluation command for Toaripi SLM CLI.

Provides comprehensive model testing and evaluation capabilities.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import pandas as pd
from loguru import logger


def find_available_models() -> List[Tuple[str, Path]]:
    """Find available trained models"""
    models = []
    
    # Check for HuggingFace format models in checkpoints
    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        for model_dir in checkpoints_dir.glob("*/"):
            if (model_dir / "config.json").exists():
                models.append(("checkpoint", model_dir))
    
    # Check models directory
    models_dir = Path("models")
    if models_dir.exists():
        # HuggingFace format
        hf_dir = models_dir / "hf"
        if hf_dir.exists():
            for model_dir in hf_dir.glob("*/"):
                if (model_dir / "config.json").exists():
                    models.append(("hf", model_dir))
        
        # GGUF format
        gguf_dir = models_dir / "gguf"
        if gguf_dir.exists():
            for gguf_file in gguf_dir.glob("*.gguf"):
                models.append(("gguf", gguf_file))
    
    return models


def validate_test_data(data_path: Path) -> Tuple[bool, List[str], Dict[str, int]]:
    """Validate test data format and content"""
    issues = []
    stats = {}
    
    if not data_path.exists():
        issues.append(f"Test data file not found: {data_path}")
        return False, issues, stats
    
    try:
        df = pd.read_csv(data_path)
        
        # Check required columns
        required_cols = ["english", "toaripi"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Calculate stats
        stats = {
            "total_samples": len(df),
            "avg_english_length": df["english"].str.len().mean() if "english" in df.columns else 0,
            "avg_toaripi_length": df["toaripi"].str.len().mean() if "toaripi" in df.columns else 0,
        }
        
        # Check for empty data
        if stats["total_samples"] == 0:
            issues.append("Test data is empty")
        elif stats["total_samples"] < 10:
            issues.append(f"Very small test set ({stats['total_samples']} samples)")
        
        # Check for empty values
        if "english" in df.columns:
            empty_english = df["english"].isna().sum()
            if empty_english > 0:
                issues.append(f"{empty_english} empty English entries")
        
        if "toaripi" in df.columns:
            empty_toaripi = df["toaripi"].isna().sum()
            if empty_toaripi > 0:
                issues.append(f"{empty_toaripi} empty Toaripi entries")
        
    except Exception as e:
        issues.append(f"Error reading test data: {e}")
        return False, issues, stats
    
    return len(issues) == 0, issues, stats


def run_basic_generation_test(model_path: Path, num_samples: int = 5) -> Dict:
    """Run basic generation test to check model functionality"""
    
    test_prompts = [
        "Children playing by the river",
        "A story about fishing",
        "Learning new words",
        "Family dinner time",
        "Walking to school"
    ]
    
    results = {
        "success": False,
        "error": None,
        "generated_samples": [],
        "avg_generation_time": 0,
        "model_loaded": False
    }
    
    try:
        # Try to import and load the generator
        from ....inference import ToaripiGenerator
        
        click.echo("📁 Loading model...")
        start_time = time.time()
        generator = ToaripiGenerator.load(str(model_path))
        load_time = time.time() - start_time
        
        results["model_loaded"] = True
        click.echo(f"✅ Model loaded in {load_time:.2f}s")
        
        # Generate samples
        generation_times = []
        for i, prompt in enumerate(test_prompts[:num_samples]):
            click.echo(f"🔄 Generating sample {i+1}/{num_samples}: {prompt}")
            
            start_time = time.time()
            try:
                generated = generator.generate_story(
                    prompt, 
                    age_group="primary",
                    max_length=100
                )
                generation_time = time.time() - start_time
                generation_times.append(generation_time)
                
                results["generated_samples"].append({
                    "prompt": prompt,
                    "generated": generated,
                    "time": generation_time,
                    "length": len(generated) if generated else 0
                })
                
                click.echo(f"   Generated: {generated[:50]}{'...' if len(generated) > 50 else ''}")
                click.echo(f"   Time: {generation_time:.2f}s")
                
            except Exception as e:
                results["generated_samples"].append({
                    "prompt": prompt,
                    "generated": None,
                    "error": str(e),
                    "time": 0
                })
                click.echo(f"   ❌ Generation failed: {e}")
        
        if generation_times:
            results["avg_generation_time"] = sum(generation_times) / len(generation_times)
            results["success"] = True
        
    except ImportError as e:
        results["error"] = f"Could not import generator: {e}"
    except Exception as e:
        results["error"] = str(e)
    
    return results


def run_content_quality_assessment(generated_samples: List[Dict]) -> Dict:
    """Assess the quality of generated content"""
    
    assessment = {
        "total_samples": len(generated_samples),
        "successful_generations": 0,
        "avg_length": 0,
        "unique_outputs": 0,
        "quality_issues": []
    }
    
    successful_samples = [s for s in generated_samples if s.get("generated")]
    assessment["successful_generations"] = len(successful_samples)
    
    if successful_samples:
        lengths = [len(s["generated"]) for s in successful_samples]
        assessment["avg_length"] = sum(lengths) / len(lengths)
        
        # Check for unique outputs
        outputs = [s["generated"] for s in successful_samples]
        assessment["unique_outputs"] = len(set(outputs))
        
        # Quality checks
        if assessment["unique_outputs"] < len(successful_samples):
            assessment["quality_issues"].append("Some generated outputs are identical")
        
        very_short = [s for s in successful_samples if len(s["generated"]) < 20]
        if very_short:
            assessment["quality_issues"].append(f"{len(very_short)} very short outputs (< 20 chars)")
        
        very_long = [s for s in successful_samples if len(s["generated"]) > 500]
        if very_long:
            assessment["quality_issues"].append(f"{len(very_long)} very long outputs (> 500 chars)")
    
    return assessment


@click.command()
@click.option(
    '--model-path',
    type=click.Path(path_type=Path),
    help='Path to trained model to test'
)
@click.option(
    '--test-data',
    type=click.Path(path_type=Path),
    help='Path to test dataset (CSV format)'
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    default=Path("evaluation"),
    help='Directory to save evaluation results'
)
@click.option(
    '--test-type',
    type=click.Choice(['basic', 'generation', 'quality', 'all']),
    default='all',
    help='Type of test to run'
)
@click.option(
    '--num-samples',
    type=int,
    default=10,
    help='Number of samples to test for generation'
)
@click.option(
    '--guided',
    is_flag=True,
    help='Run in guided mode with interactive prompts'
)
@click.option(
    '--save-results',
    is_flag=True,
    default=True,
    help='Save detailed results to files'
)
def test(
    model_path: Optional[Path],
    test_data: Optional[Path],
    output_dir: Path,
    test_type: str,
    num_samples: int,
    guided: bool,
    save_results: bool
):
    """
    Test and evaluate a trained Toaripi language model.
    
    This command provides comprehensive testing capabilities including
    basic functionality checks, generation quality assessment, and
    performance benchmarking.
    
    \b
    Examples:
        toaripi test --guided                     # Interactive testing
        toaripi test --model-path models/my-model # Quick test
        toaripi test --test-type generation       # Generation test only
        toaripi test --num-samples 20             # Extended testing
    """
    
    if guided or not model_path:
        click.echo("🧪 Welcome to Toaripi SLM Model Testing!\n")
        
        # Model selection
        if not model_path:
            click.echo("🔍 Searching for available models...")
            available_models = find_available_models()
            
            if not available_models:
                click.echo("❌ No trained models found!")
                click.echo("💡 Tips:")
                click.echo("   • Train a model first: toaripi train")
                click.echo("   • Check that models are in checkpoints/ or models/ directories")
                return
            
            click.echo("📚 Available models:")
            for i, (model_type, path) in enumerate(available_models, 1):
                click.echo(f"  {i}. {path} ({model_type})")
            
            choice = click.prompt(
                f"\nSelect model to test (1-{len(available_models)})",
                type=click.IntRange(1, len(available_models))
            )
            
            model_path = available_models[choice - 1][1]
        
        # Test data selection
        if test_type in ['quality', 'all'] and not test_data:
            click.echo(f"\n📊 For quality assessment, we need test data.")
            
            # Check common locations
            common_test_locations = [
                Path("data/processed/test.csv"),
                Path("data/processed/validation.csv"),
                Path("data/samples/sample_parallel.csv"),
            ]
            
            for loc in common_test_locations:
                if loc.exists():
                    if click.confirm(f"Use test data from {loc}?"):
                        test_data = loc
                        break
            
            if not test_data:
                test_data_str = click.prompt(
                    "Enter path to test data (or press enter to skip)",
                    default="",
                    show_default=False
                )
                if test_data_str:
                    test_data = Path(test_data_str)
        
        # Test type selection
        if guided:
            click.echo(f"\n🎯 What type of testing would you like to perform?")
            test_options = [
                ("basic", "Basic model loading and functionality"),
                ("generation", "Text generation quality and speed"),
                ("quality", "Content quality assessment (requires test data)"),
                ("all", "Complete evaluation suite")
            ]
            
            for i, (name, desc) in enumerate(test_options, 1):
                click.echo(f"  {i}. {name.title()}: {desc}")
            
            choice = click.prompt(
                f"\nSelect test type (1-{len(test_options)})",
                type=click.IntRange(1, len(test_options)),
                default=4
            )
            
            test_type = test_options[choice - 1][0]
    
    # Validate model path
    if not model_path or not model_path.exists():
        click.echo(f"❌ Model not found: {model_path}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start testing
    click.echo(f"\n🚀 Starting {test_type} testing of: {model_path}")
    
    results = {
        "model_path": str(model_path),
        "test_type": test_type,
        "timestamp": time.time(),
        "results": {}
    }
    
    # Basic testing
    if test_type in ['basic', 'all']:
        click.echo("\n" + "="*50)
        click.echo("🔧 BASIC MODEL TESTING")
        click.echo("="*50)
        
        basic_results = run_basic_generation_test(model_path, num_samples=min(3, num_samples))
        results["results"]["basic"] = basic_results
        
        if basic_results["model_loaded"]:
            click.echo("✅ Model loading: SUCCESS")
        else:
            click.echo("❌ Model loading: FAILED")
            if basic_results["error"]:
                click.echo(f"   Error: {basic_results['error']}")
        
        if basic_results["success"]:
            click.echo(f"✅ Generation test: SUCCESS")
            click.echo(f"   Average generation time: {basic_results['avg_generation_time']:.2f}s")
        else:
            click.echo("❌ Generation test: FAILED")
    
    # Generation testing
    if test_type in ['generation', 'all']:
        click.echo("\n" + "="*50)
        click.echo("📝 GENERATION TESTING")
        click.echo("="*50)
        
        generation_results = run_basic_generation_test(model_path, num_samples)
        results["results"]["generation"] = generation_results
        
        if generation_results["success"]:
            assessment = run_content_quality_assessment(generation_results["generated_samples"])
            results["results"]["quality_assessment"] = assessment
            
            click.echo(f"✅ Generated {assessment['successful_generations']}/{assessment['total_samples']} samples")
            click.echo(f"📏 Average length: {assessment['avg_length']:.1f} characters")
            click.echo(f"🎯 Unique outputs: {assessment['unique_outputs']}/{assessment['successful_generations']}")
            
            if assessment["quality_issues"]:
                click.echo("⚠️  Quality issues:")
                for issue in assessment["quality_issues"]:
                    click.echo(f"   • {issue}")
        else:
            click.echo("❌ Generation testing failed")
    
    # Quality testing with test data
    if test_type in ['quality', 'all'] and test_data:
        click.echo("\n" + "="*50)
        click.echo("📊 QUALITY ASSESSMENT")
        click.echo("="*50)
        
        # Validate test data
        valid, issues, stats = validate_test_data(test_data)
        results["results"]["test_data_validation"] = {
            "valid": valid,
            "issues": issues,
            "stats": stats
        }
        
        if valid:
            click.echo(f"✅ Test data validated: {stats['total_samples']} samples")
            
            # TODO: Implement more sophisticated quality metrics
            # This would include BLEU scores, semantic similarity, etc.
            click.echo("🔄 Advanced quality metrics coming soon...")
            
        else:
            click.echo("❌ Test data validation failed:")
            for issue in issues:
                click.echo(f"   • {issue}")
    
    # Save results
    if save_results:
        results_file = output_dir / f"test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        click.echo(f"\n💾 Results saved to: {results_file}")
    
    # Summary
    click.echo("\n" + "="*50)
    click.echo("📋 TEST SUMMARY")
    click.echo("="*50)
    
    if results["results"].get("basic", {}).get("model_loaded"):
        click.echo("✅ Model loads successfully")
    else:
        click.echo("❌ Model loading issues")
    
    if results["results"].get("generation", {}).get("success"):
        click.echo("✅ Can generate content")
    else:
        click.echo("❌ Content generation issues")
    
    # Overall assessment
    success_count = sum([
        results["results"].get("basic", {}).get("model_loaded", False),
        results["results"].get("generation", {}).get("success", False),
    ])
    
    if success_count >= 2:
        click.echo("\n🎉 Overall: Model is working well!")
        click.echo("💡 Ready for interactive use: toaripi interact")
    elif success_count >= 1:
        click.echo("\n⚠️  Overall: Model has some issues but may be usable")
    else:
        click.echo("\n❌ Overall: Model has significant issues")
        click.echo("💡 Consider retraining or checking model files")


if __name__ == '__main__':
    test()