#!/usr/bin/env python3
"""
Test script for the enhanced interactive CLI with side-by-side display
and token weight visualization.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from src.toaripi_slm.cli.commands.interact import BilingualDisplay, TokenWeight

def test_token_weight_visualization():
    """Test the token weight visualization system."""
    console = Console()
    display = BilingualDisplay(console)
    
    console.print("🧪 [bold blue]Testing Token Weight Visualization[/bold blue]\n")
    
    # Test sample texts
    english_text = "The children went fishing by the river"
    toaripi_text = "Bada-bada na'a gola hanere malolo peni"
    
    console.print("📝 [yellow]Sample Content:[/yellow]")
    console.print(f"English: {english_text}")
    console.print(f"Toaripi: {toaripi_text}\n")
    
    # Test 1: Basic bilingual display with weights
    console.print("🎯 [green]Test 1: Basic Bilingual Display with Token Weights[/green]")
    display.display_bilingual_content(english_text, toaripi_text, "story")
    
    # Test 2: Display without weights
    console.print("\n🎯 [green]Test 2: Display without Token Weights[/green]")
    display.toggle_weights()
    display.display_bilingual_content(english_text, toaripi_text, "story")
    
    # Test 3: Re-enable weights and test alignment toggle
    console.print("\n🎯 [green]Test 3: Toggle Alignment (weights re-enabled)[/green]")
    display.toggle_weights()  # Re-enable weights
    display.toggle_alignment()  # Disable alignment
    display.display_bilingual_content(english_text, toaripi_text, "story")
    
    # Test 4: Test different content types
    console.print("\n🎯 [green]Test 4: Different Content Types[/green]")
    display.toggle_alignment()  # Re-enable alignment
    
    vocab_english = "Fish swimming water children learning"
    vocab_toaripi = "Hanere potopoto peni bada-bada nene-ida"
    
    console.print("\n📚 [cyan]Vocabulary Example:[/cyan]")
    display.display_bilingual_content(vocab_english, vocab_toaripi, "vocabulary")
    
    dialogue_english = "Teacher: What did you learn? Student: I learned about fishing."
    dialogue_toaripi = "Amo-harigi: Ami na'a nene? Amo-nene: Mina na'a nene hanere."
    
    console.print("\n💬 [cyan]Dialogue Example:[/cyan]")
    display.display_bilingual_content(dialogue_english, dialogue_toaripi, "dialogue")
    
    # Test 5: Individual token weight creation
    console.print("\n🎯 [green]Test 5: Individual Token Weight Examples[/green]")
    
    tokens = [
        TokenWeight("High", 0.9),
        TokenWeight("Medium-High", 0.7),
        TokenWeight("Medium", 0.5),
        TokenWeight("Low", 0.3),
        TokenWeight("Very-Low", 0.1)
    ]
    
    from rich.text import Text
    test_text = Text()
    for i, tw in enumerate(tokens):
        if i > 0:
            test_text.append(" ")
        test_text.append(tw.token, style=tw.get_color_style())
        test_text.append(f"({tw.weight:.1f})", style="dim white")
    
    console.print("Color Scale Example:")
    console.print(test_text)
    
    # Show legend
    console.print("\n🎨 [green]Weight Legend:[/green]")
    display.display_weight_legend()
    
    console.print("\n✅ [bold green]All tests completed![/bold green]")

def test_interactive_commands():
    """Test the available interactive commands."""
    console = Console()
    
    console.print("\n📋 [bold blue]Available Interactive Commands[/bold blue]\n")
    
    commands = [
        ("/help", "Show help message"),
        ("/type story", "Change to story generation"),
        ("/weights", "Toggle token weight display"),
        ("/align", "Toggle token alignment"),
        ("/legend", "Show color legend"),
        ("/settings", "Adjust generation settings"),
        ("/history", "Show conversation history"),
        ("/save", "Save session to file"),
        ("/clear", "Clear conversation history"),
        ("/quit", "Exit interactive mode")
    ]
    
    from rich.table import Table
    
    cmd_table = Table(title="Interactive Commands", show_header=True, header_style="bold magenta")
    cmd_table.add_column("Command", style="cyan")
    cmd_table.add_column("Description", style="green")
    
    for cmd, desc in commands:
        cmd_table.add_row(cmd, desc)
    
    console.print(cmd_table)

if __name__ == "__main__":
    try:
        test_token_weight_visualization()
        test_interactive_commands()
        
        print("\n" + "="*50)
        print("🎉 Enhanced Interactive CLI Test Complete!")
        print("="*50)
        print("\nTo run the interactive mode:")
        print("  toaripi interact")
        print("\nNew features available:")
        print("  • Side-by-side English ↔ Toaripi display")
        print("  • Colored token weight visualization")
        print("  • Token alignment between languages")
        print("  • Interactive visualization controls")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)