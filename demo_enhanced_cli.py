#!/usr/bin/env python3
"""
Demo script to showcase the enhanced Toaripi SLM Interactive CLI
with side-by-side display and token weight visualization.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from src.toaripi_slm.cli.commands.interact import BilingualDisplay, ToaripiGenerator

def demo_enhanced_features():
    """Demonstrate the enhanced interactive CLI features."""
    console = Console()
    
    # Title
    title_text = Text()
    title_text.append("🌟 Toaripi SLM Enhanced Interactive CLI Demo", style="bold blue")
    
    console.print("\n")
    console.print(Panel(title_text, border_style="blue"))
    
    # Initialize components
    display = BilingualDisplay(console)
    
    console.print("\n📋 [bold cyan]New Features Demonstration[/bold cyan]\n")
    
    # Demo 1: Side-by-side display with token weights
    console.print("🎯 [green]Feature 1: Side-by-Side Display with Token Weights[/green]")
    console.print("English text and Toaripi translation are displayed in parallel columns")
    console.print("with color-coded token weights showing model attention.\n")
    
    english_story = "The children learned traditional fishing methods from their elders"
    toaripi_story = "Bada-bada na'a nene-ida hanere taumate mina gabua-harigi"
    
    display.display_bilingual_content(english_story, toaripi_story, "story")
    
    # Demo 2: Different content types
    console.print("\n🎯 [green]Feature 2: Multiple Content Types[/green]")
    console.print("Different educational content types with appropriate formatting:\n")
    
    # Vocabulary
    console.print("📚 [yellow]Vocabulary Training:[/yellow]")
    vocab_en = "Fish children water river traditional knowledge"
    vocab_to = "Hanere bada-bada peni malolo taumate nene-ida"
    display.display_bilingual_content(vocab_en, vocab_to, "vocabulary")
    
    # Dialogue
    console.print("\n💬 [yellow]Dialogue Practice:[/yellow]")
    dialogue_en = "Elder: Children, watch how we cast the net. Child: Yes, we are learning carefully."
    dialogue_to = "Gabua-harigi: Bada-bada, ra'u ami na'a gola hanere. Bada: Aba, mina na'a nene-ida mane-mane."
    display.display_bilingual_content(dialogue_en, dialogue_to, "dialogue")
    
    # Demo 3: Interactive controls
    console.print("\n🎯 [green]Feature 3: Interactive Visualization Controls[/green]")
    console.print("Use commands to control the display:\n")
    
    console.print("🔧 [cyan]Toggle token weights OFF:[/cyan]")
    display.toggle_weights()
    display.display_bilingual_content("Simple display without weights", "Mina gabua kura hareva", "translation")
    
    console.print("\n🔧 [cyan]Toggle token weights ON and alignment OFF:[/cyan]")
    display.toggle_weights()
    display.toggle_alignment()
    display.display_bilingual_content("Different alignment strategy", "Hareva kura gola-ida mane", "translation")
    
    # Demo 4: Available commands
    console.print("\n🎯 [green]Feature 4: Enhanced Command Set[/green]")
    
    commands_panel = Panel(
        """
[bold yellow]New Interactive Commands:[/bold yellow]

• [cyan]/weights[/cyan] - Toggle token weight visualization
• [cyan]/align[/cyan] - Toggle token alignment between languages  
• [cyan]/legend[/cyan] - Show color legend for token weights
• [cyan]/type <type>[/cyan] - Switch between content types
• [cyan]/settings[/cyan] - Adjust generation parameters

[bold yellow]Content Types:[/bold yellow]

• [green]story[/green] - Educational stories with cultural context
• [green]vocabulary[/green] - Vocabulary exercises with translations
• [green]dialogue[/green] - Conversational practice materials
• [green]questions[/green] - Comprehension questions
• [green]translation[/green] - Translation practice
        """,
        title="📖 Interactive Commands Guide",
        border_style="magenta"
    )
    
    console.print(commands_panel)
    
    # Demo 5: Educational benefits
    console.print("\n🎯 [green]Feature 5: Educational Benefits[/green]")
    
    benefits_panel = Panel(
        """
[bold cyan]For Teachers:[/bold cyan]
• Visual feedback on model attention helps understand generation quality
• Side-by-side display aids in teaching translation techniques
• Token weights show which words the model considers important
• Different content types support varied lesson plans

[bold cyan]For Language Learners:[/bold cyan]
• Clear visual connection between English and Toaripi
• Token weights highlight key vocabulary and concepts
• Interactive controls allow exploration of language patterns
• Multiple content formats support different learning styles

[bold cyan]For Linguists & Researchers:[/bold cyan]
• Token attention visualization reveals model behavior
• Alignment display shows cross-linguistic correspondences
• Session logging captures interaction patterns for analysis
• Customizable display settings for research needs
        """,
        title="🎓 Educational Impact",
        border_style="green"
    )
    
    console.print(benefits_panel)
    
    # How to run
    console.print("\n🚀 [bold blue]How to Run the Enhanced Interactive Mode[/bold blue]\n")
    
    run_panel = Panel(
        """
[bold yellow]Start Interactive Mode:[/bold yellow]

1. [cyan]toaripi interact[/cyan] - Launch with default settings
2. [cyan]toaripi interact --model path/to/model[/cyan] - Use specific model
3. [cyan]toaripi interact --content-type vocabulary[/cyan] - Start with vocabulary mode

[bold yellow]Example Session:[/bold yellow]

```
$ toaripi interact
💬 Toaripi SLM Interactive Mode

🌟 Enhanced Interactive Session Started
Current model: ./models/hf
Content type: story

You: Create a story about children learning to fish
💬 Generated Content (story):

[Side-by-side display with colored token weights]

You: /weights
💡 Token weights display: OFF

You: /type vocabulary  
✅ Content type changed to: vocabulary

You: fishing vocabulary for children
[Vocabulary display with translations]
```
        """,
        title="🔧 Usage Instructions",
        border_style="cyan"
    )
    
    console.print(run_panel)
    
    console.print("\n✨ [bold green]Demo Complete![/bold green]")
    console.print("The enhanced Toaripi SLM Interactive CLI is ready for educational content generation")
    console.print("with advanced visualization features!\n")

if __name__ == "__main__":
    try:
        demo_enhanced_features()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)