#!/usr/bin/env python3
"""
Demo of the Enhanced Toaripi SLM Interactive CLI
Shows the complete integration working with model-based generation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from src.toaripi_slm.cli.commands.interact import interactive_mode

def demo_enhanced_cli():
    """Demonstrate the enhanced CLI features."""
    console = Console()
    
    # Title
    title_text = Text()
    title_text.append("🌺 Enhanced Toaripi SLM Interactive CLI Demo", style="bold blue")
    
    console.print("\n")
    console.print(Panel(title_text, border_style="blue"))
    
    console.print("\n🎯 [bold cyan]Features Demonstration[/bold cyan]\n")
    
    features_panel = Panel(
        """
[bold yellow]Enhanced Features:[/bold yellow]

🔄 [green]Automatic Model Detection[/green]
• Detects trained models in ./models/hf/ directory
• Gracefully falls back to demo mode if no model found
• Shows loading progress and status

🎨 [green]Side-by-Side Display[/green]  
• English text and Toaripi translation displayed side-by-side
• Clear visual separation with Rich panels and borders
• Responsive layout adapts to terminal width

🌈 [green]Token Weight Visualization[/green]
• Color-coded token weights show model attention
• Gradient from red (high) to blue (low attention)
• Real attention weights from model when available
• Simulated weights based on linguistic features in demo mode

💬 [green]Chat Functionality[/green]
• Ask questions in English, get responses in Toaripi
• Format: "toaripi_word/english_word - description"  
• Shows both semantic and morphological information
• Powered by trained SLM when model is loaded

⚙️ [green]Interactive Controls[/green]
• Toggle token weight display on/off
• Switch between different content types
• Real-time visualization updates
• Help system with command reference

[bold yellow]Commands in Interactive Mode:[/bold yellow]
• [cyan]chat <question>[/cyan] - Ask questions in English, get Toaripi responses
• [cyan]story <prompt>[/cyan] - Generate educational stories  
• [cyan]vocab <topic>[/cyan] - Generate vocabulary lists
• [cyan]toggle-weights[/cyan] - Toggle token weight visualization
• [cyan]help[/cyan] - Show available commands
• [cyan]exit[/cyan] - Exit interactive mode
        """,
        title="🔧 CLI Features Overview",
        border_style="cyan"
    )
    
    console.print(features_panel)
    
    console.print("\n💡 [bold yellow]Ready to Start Interactive Mode![/bold yellow]")
    console.print("The CLI will automatically detect and load any trained models.")
    console.print("If no trained model is found, it will run in demo mode with sample responses.")
    console.print("\n🚀 [green]Starting Interactive Mode...[/green]")

if __name__ == "__main__":
    try:
        demo_enhanced_cli()
        
        # Start the actual interactive mode
        interactive_mode()
        
    except KeyboardInterrupt:
        print("\n\n👋 Interactive mode ended by user.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()