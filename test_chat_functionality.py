#!/usr/bin/env python3
"""
Test script for the chat functionality in the enhanced Toaripi SLM Interactive CLI.
Demonstrates Q&A format with English questions and Toaripi responses with token weights.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from src.toaripi_slm.cli.core import BilingualDisplay, ToaripiGenerator

def test_chat_functionality():
    """Test the chat Q&A functionality with token weight visualization."""
    console = Console()
    
    # Title
    title_text = Text()
    title_text.append("💬 Toaripi SLM Chat Functionality Test", style="bold blue")
    
    console.print("\n")
    console.print(Panel(title_text, border_style="blue"))
    
    # Initialize components
    display = BilingualDisplay(console)
    generator = ToaripiGenerator(Path("./models/test"))
    generator.load()
    
    console.print("\n📋 [bold cyan]Chat Q&A Testing[/bold cyan]\n")
    
    # Test different types of questions
    test_questions = [
        "What is a dog?",
        "What is water?",
        "What is a mother?",
        "What is fishing?",
        "What is a house?",
        "What is food?",
        "What is good?",
        "What is a banana?",
        "What is teaching?",
        "What is a crocodile?",
        "What is something I don't know?"  # Test unknown question
    ]
    
    console.print("🎯 [green]Testing Chat Responses with Token Weights[/green]\n")
    
    for i, question in enumerate(test_questions, 1):
        console.print(f"📝 [yellow]Question {i}:[/yellow] {question}")
        eng, tqo = generator.bilingual(question, content_type="chat", max_length=50, temperature=0.7)
        display.display(eng, tqo, "chat")
        console.print()  # Add spacing between questions
    
    # Test keyword extraction
    console.print("🔍 [green]Testing Complex Questions[/green]\n")
    
    complex_questions = [
        "Can you tell me what a fish is?",
        "I want to know about dogs",
        "Please explain what water means",
        "Do you know anything about children?",
        "Tell me about the village"
    ]
    
    for question in complex_questions:
        console.print(f"📝 [yellow]Complex Question:[/yellow] {question}")
        eng, tqo = generator.bilingual(question, content_type="chat", max_length=50, temperature=0.7)
        display.display(eng, tqo, "chat")
        console.print()
    
    # Test without token weights
    console.print("🎯 [green]Testing Chat Display Without Token Weights[/green]\n")
    display.toggle_weights()
    
    question = "What is a teacher?"
    console.print(f"📝 [yellow]Question:[/yellow] {question}")
    eng, tqo = generator.bilingual(question, content_type="chat", max_length=50, temperature=0.7)
    display.display(eng, tqo, "chat")
    
    # Re-enable weights
    display.toggle_weights()
    
    # Show available knowledge
    console.print("\n📚 [green]Available Knowledge Categories[/green]\n")
    
    categories = {
        "Animals": ["dog", "fish", "bird", "pig", "chicken", "crocodile"],
        "Family": ["mother", "father", "child", "children", "elder", "teacher", "student"],
        "Nature": ["water", "river", "tree", "village", "house", "garden", "forest", "sea"],
        "Activities": ["fishing", "hunting", "cooking", "learning", "teaching", "swimming", "walking"],
        "Food": ["food", "rice", "banana", "coconut", "sweet potato"],
        "Objects": ["net", "boat", "fire", "knife"],
        "Concepts": ["good", "bad", "big", "small", "happy", "sad"]
    }
    
    from rich.table import Table
    
    knowledge_table = Table(title="Chat Knowledge Base", show_header=True, header_style="bold magenta")
    knowledge_table.add_column("Category", style="cyan")
    knowledge_table.add_column("Available Terms", style="green")
    
    for category, terms in categories.items():
        terms_text = ", ".join(terms[:8])  # Show first 8 terms
        if len(terms) > 8:
            terms_text += f" ... (+{len(terms)-8} more)"
        knowledge_table.add_row(category, terms_text)
    
    console.print(knowledge_table)
    
    # Usage example
    console.print("\n🚀 [bold blue]How to Use Chat Mode[/bold blue]\n")
    
    usage_panel = Panel(
        """
[bold yellow]Interactive Chat Mode:[/bold yellow]

1. Start interactive mode: [cyan]toaripi interact[/cyan]
2. Switch to chat mode: [cyan]/type chat[/cyan]
3. Ask questions in English: [cyan]What is a dog?[/cyan]
4. Get Toaripi responses with token weights: [green]ruru/dog - Description[/green]

[bold yellow]Example Session:[/bold yellow]

```
You: /type chat
✅ Content type changed to: chat
💬 Chat mode enabled! Ask questions like 'What is a dog?' or 'What is water?'

You: What is a fish?
💬 Generated Content (chat):

[English panel]          [Toaripi panel with token weights]
What is fish?           hanere/fish - Swimming creatures...

You: /weights
💡 Token weights display: OFF

You: What is water?
[Simple display without weights]
```

[bold yellow]Question Types:[/bold yellow]

• Direct: "What is X?" 
• Descriptive: "Tell me about X"
• Casual: "Do you know about X?"
• Complex: "Can you explain what X means?"
        """,
        title="💬 Chat Mode Usage Guide",
        border_style="cyan"
    )
    
    console.print(usage_panel)
    
    console.print("\n✅ [bold green]Chat functionality test completed![/bold green]")
    console.print("The chat mode enables natural Q&A with visual token weight feedback.")

if __name__ == "__main__":
    try:
        test_chat_functionality()
        
        print("\n" + "="*60)
        print("🎉 Chat Functionality Test Complete!")
        print("="*60)
        print("\nThe enhanced Toaripi SLM now supports:")
        print("  • English questions → Toaripi answers")
        print("  • Token weight visualization for responses")
        print("  • Comprehensive knowledge base")
        print("  • Natural language question processing")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)