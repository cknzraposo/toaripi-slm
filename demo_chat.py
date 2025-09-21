#!/usr/bin/env python3
"""
Simple demo of the Toaripi SLM Chat Functionality.
Shows English questions → Toaripi answers with token weights.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from src.toaripi_slm.cli.commands.interact import BilingualDisplay, ToaripiGenerator

def run_chat_demo():
    """Run a simple demonstration of the chat functionality."""
    console = Console()
    
    # Title
    console.print("\n")
    console.print("🌟 [bold blue]Toaripi SLM Chat Demo[/bold blue] 🌟\n")
    
    # Initialize components
    display = BilingualDisplay(console)
    generator = ToaripiGenerator(Path("./models/demo"))
    generator.load_model()
    
    # Demo questions
    demo_questions = [
        "What is a dog?",
        "What is water?", 
        "What is a mother?",
        "What is fishing?",
        "What is good?"
    ]
    
    console.print("💬 [cyan]Interactive Chat Examples[/cyan]\n")
    console.print("Ask questions in English, get Toaripi answers with token weights!\n")
    
    for i, question in enumerate(demo_questions, 1):
        console.print(f"[bold yellow]Question {i}:[/bold yellow] {question}")
        
        # Generate response
        english_response, toaripi_response = generator.generate_chat_response(question)
        
        # Display with token weights
        display.display_bilingual_content(english_response, toaripi_response, "chat")
        
        console.print()  # Space between questions
    
    # Show the format explanation
    console.print("📝 [green]Response Format:[/green]")
    console.print("   [cyan]toaripi_word/english_word[/cyan] - Cultural description")
    console.print("   Token weights show word importance with colors!")
    
    console.print("\n🚀 [bold green]Try it yourself:[/bold green]")
    console.print("   [cyan]toaripi interact[/cyan]")
    console.print("   [cyan]/type chat[/cyan]")
    console.print("   [cyan]What is a fish?[/cyan]")

if __name__ == "__main__":
    try:
        run_chat_demo()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)