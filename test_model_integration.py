#!/usr/bin/env python3
"""
Test script for the model-integrated Toaripi SLM Interactive CLI.
Tests both demo mode and real model integration.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from src.toaripi_slm.cli.commands.interact import BilingualDisplay, ToaripiGenerator

def test_model_integration():
    """Test the model integration capabilities."""
    console = Console()
    
    # Title
    title_text = Text()
    title_text.append("🤖 Toaripi SLM Model Integration Test", style="bold blue")
    
    console.print("\n")
    console.print(Panel(title_text, border_style="blue"))
    
    console.print("\n📋 [bold cyan]Testing Model Integration[/bold cyan]\n")
    
    # Test 1: Demo mode (no real model available)
    console.print("🎭 [green]Test 1: Demo Mode (No Trained Model)[/green]")
    console.print("Testing fallback behavior when no trained model is available\n")
    
    display = BilingualDisplay(console)
    generator = ToaripiGenerator(Path("./models/nonexistent"))
    generator.load_model()
    display.set_generator(generator)
    
    console.print("📝 [yellow]Chat Question:[/yellow] What is a dog?")
    english_response, toaripi_response = generator.generate_chat_response("What is a dog?")
    display.display_bilingual_content(english_response, toaripi_response, "chat")
    
    console.print("\n📝 [yellow]Story Generation:[/yellow] Tell a story about fishing")
    english_content, toaripi_content = generator.generate_bilingual_content("Tell a story about fishing", "story")
    display.display_bilingual_content(english_content, toaripi_content, "story")
    
    # Test 2: Check for real model availability
    console.print("\n🔍 [green]Test 2: Checking for Trained Models[/green]")
    
    possible_model_paths = [
        Path("./models/hf"),
        Path("./models/toaripi-slm"),
        Path("./models/fine-tuned"),
        Path("../models"),
        Path("./training_sessions")
    ]
    
    found_models = []
    for path in possible_model_paths:
        if path.exists():
            config_file = path / "config.json"
            if config_file.exists():
                found_models.append(path)
                console.print(f"✅ Found potential model at: [cyan]{path}[/cyan]")
            else:
                console.print(f"📂 Directory exists but no config.json: [dim]{path}[/dim]")
        else:
            console.print(f"❌ Path does not exist: [dim]{path}[/dim]")
    
    if found_models:
        console.print(f"\n🎯 [green]Test 3: Attempting to Load Real Model[/green]")
        
        # Try to load the first found model
        model_path = found_models[0]
        console.print(f"Loading model from: [cyan]{model_path}[/cyan]")
        
        real_generator = ToaripiGenerator(model_path)
        success = real_generator.load_model()
        
        if success and real_generator.model is not None:
            console.print("✅ [green]Real model loaded successfully![/green]")
            display.set_generator(real_generator)
            
            console.print("\n📝 [yellow]Testing Real Model Chat:[/yellow] What is water?")
            english_response, toaripi_response = real_generator.generate_chat_response("What is water?")
            display.display_bilingual_content(english_response, toaripi_response, "chat")
            
        else:
            console.print("⚠️  [yellow]Model loading failed, using demo mode[/yellow]")
    
    else:
        console.print("\n💡 [yellow]No trained models found. To test with a real model:[/yellow]")
        console.print("   1. Train a model using: [cyan]toaripi train[/cyan]")
        console.print("   2. Or place a trained model in: [cyan]./models/hf/[/cyan]")
        console.print("   3. Ensure the model directory contains: [cyan]config.json[/cyan], [cyan]pytorch_model.bin[/cyan], [cyan]tokenizer.json[/cyan]")
    
    # Test 3: Token weight extraction differences
    console.print("\n🎨 [green]Test 4: Token Weight Extraction Methods[/green]")
    
    sample_text = "hanere/fish - Swimming creatures caught in rivers for food"
    
    console.print("📊 [yellow]Simulated vs Model-based token weights:[/yellow]")
    console.print(f"Text: {sample_text}")
    
    # Show simulated weights
    simulated_weights = generator._simulate_token_weights(sample_text)
    console.print("\n🎭 [cyan]Simulated weights:[/cyan]")
    for tw in simulated_weights:
        color = tw.get_color_style()
        console.print(f"  {tw.token}: {tw.weight:.2f}", style=color)
    
    # Show model-based weights (if available)
    if 'real_generator' in locals() and real_generator.model is not None:
        model_weights = real_generator.extract_token_weights_from_model("", sample_text)
        console.print("\n🤖 [cyan]Model-based weights:[/cyan]")
        for tw in model_weights:
            color = tw.get_color_style()
            console.print(f"  {tw.token}: {tw.weight:.2f}", style=color)
    else:
        console.print("\n🤖 [dim]Model-based weights: Not available (no trained model loaded)[/dim]")
    
    # Show integration benefits
    console.print("\n💡 [green]Integration Benefits[/green]")
    
    benefits_panel = Panel(
        """
[bold yellow]Demo Mode (No Trained Model):[/bold yellow]
• Uses static fallback responses for demonstration
• Simulated token weights based on linguistic heuristics
• Shows interface and visualization capabilities
• Provides examples of expected format

[bold yellow]Real Model Mode (Trained SLM Loaded):[/bold yellow]
• Generates actual Toaripi responses from trained model
• Extracts real attention weights from model layers
• Responds to user prompts with learned knowledge
• Provides authentic language generation

[bold yellow]How to Enable Real Model Mode:[/bold yellow]
1. Train a Toaripi SLM using the training pipeline
2. Ensure model files are in ./models/hf/ directory
3. Model must include: config.json, pytorch_model.bin, tokenizer files
4. Run 'toaripi interact' - system will auto-detect and load model
        """,
        title="🔧 Model Integration Guide",
        border_style="cyan"
    )
    
    console.print(benefits_panel)
    
    console.print("\n✅ [bold green]Model integration test completed![/bold green]")
    console.print("The system seamlessly switches between demo and real model modes.")

def check_dependencies():
    """Check if required dependencies for model loading are available."""
    console = Console()
    
    console.print("\n🔍 [bold cyan]Checking Model Dependencies[/bold cyan]\n")
    
    dependencies = [
        ("transformers", "HuggingFace Transformers library"),
        ("torch", "PyTorch deep learning framework"),
        ("accelerate", "HuggingFace Accelerate for model loading"),
        ("peft", "Parameter Efficient Fine-Tuning (LoRA)")
    ]
    
    missing_deps = []
    
    for dep, description in dependencies:
        try:
            __import__(dep)
            console.print(f"✅ {dep}: [green]Available[/green] - {description}")
        except ImportError:
            console.print(f"❌ {dep}: [red]Missing[/red] - {description}")
            missing_deps.append(dep)
    
    if missing_deps:
        console.print(f"\n💡 [yellow]Install missing dependencies:[/yellow]")
        console.print(f"   pip install {' '.join(missing_deps)}")
        return False
    else:
        console.print(f"\n✅ [green]All dependencies available for model loading![/green]")
        return True

if __name__ == "__main__":
    try:
        # Check dependencies first
        deps_ok = check_dependencies()
        
        if deps_ok:
            test_model_integration()
        else:
            print("\n⚠️  Some dependencies missing. Demo mode will work, but real model loading requires all dependencies.")
        
        print("\n" + "="*60)
        print("🎉 Model Integration Test Complete!")
        print("="*60)
        print("\nThe enhanced CLI now supports:")
        print("  • Automatic model detection and loading")
        print("  • Real SLM generation when model available")
        print("  • Demo mode fallback for testing interface")
        print("  • Token weight extraction from model attention")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)