"""Interactive command for the Toaripi SLM CLI (refactored modular version)."""

from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import json

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm

from ..core import (
    ToaripiGenerator,
    BilingualDisplay,
    SimulatedTokenWeightProvider,
    InteractiveSession,
)
from ..core.versioning import resolve_version_dir

console = Console()

def _new_session() -> InteractiveSession:
    display = BilingualDisplay(console=console, provider=SimulatedTokenWeightProvider())
    return InteractiveSession(display=display)

def show_help_panel():
    """Display help information for interactive mode."""
    
    help_text = """
    [bold cyan]Interactive Commands:[/bold cyan]
    
    [yellow]/help[/yellow] - Show this help message
    [yellow]/type <content_type>[/yellow] - Change content type (story, vocabulary, dialogue, questions, translation, chat)
    [yellow]/settings[/yellow] - Adjust generation settings
    [yellow]/history[/yellow] - Show conversation history
    [yellow]/save[/yellow] - Save conversation to file
    [yellow]/clear[/yellow] - Clear conversation history
    [yellow]/weights[/yellow] - Toggle token weight display
    [yellow]/align[/yellow] - Toggle token alignment between languages
    [yellow]/legend[/yellow] - Show token weight color legend
    [yellow]/quit or /exit[/yellow] - Exit interactive mode
    
    [bold cyan]Content Types:[/bold cyan]
    
    [green]story[/green] - Generate educational stories
    [green]vocabulary[/green] - Create vocabulary lists with translations
    [green]dialogue[/green] - Create conversations between characters
    [green]questions[/green] - Generate comprehension questions
    [green]translation[/green] - Translate English text to Toaripi
    [green]chat[/green] - Ask questions in English, get answers in Toaripi with token weights
    
    [bold cyan]Visualization Features:[/bold cyan]
    
    • [magenta]Side-by-side display[/magenta] - English and Toaripi shown in parallel
    • [magenta]Token weights[/magenta] - Color-coded attention weights for each word
    • [magenta]Weight legend[/magenta] - Color guide for understanding token importance
    • [magenta]Token alignment[/magenta] - Visual alignment between corresponding words
    
    [bold cyan]Chat Functionality:[/bold cyan]
    
    In [green]chat[/green] mode, ask questions in English and get Toaripi responses:
    • "What is a dog?" → "ruru/dog - A four-legged animal that helps with hunting"
    • "What is water?" → "peni/water - Clear liquid essential for life"
    • Supports animals, family, nature, activities, food, and objects
    • Token weights show which parts of the response are most important
    
    [bold cyan]Tips:[/bold cyan]
    
    • Be specific in your prompts for better results
    • Mention age group (e.g., "for primary school children")
    • Include cultural context when relevant
    • Use simple, clear language in your requests
    • Toggle token weights to see model attention patterns
    • Try chat mode for vocabulary learning with visual feedback
    """
    
    console.print(Panel(help_text, title="Interactive Mode Help", border_style="blue"))

def show_settings_menu() -> Dict[str, Any]:
    """Show and update generation settings."""
    
    current_settings = {
        "max_length": 200,
        "temperature": 0.7,
        "content_type": "story"
    }
    
    console.print("\n⚙️  [bold blue]Generation Settings[/bold blue]")
    
    settings_table = Table(show_header=True, header_style="bold magenta")
    settings_table.add_column("Setting", style="cyan")
    settings_table.add_column("Current Value", style="green")
    settings_table.add_column("Description", style="dim")
    
    settings_table.add_row("Max Length", str(current_settings["max_length"]), "Maximum tokens to generate")
    settings_table.add_row("Temperature", str(current_settings["temperature"]), "Creativity level (0.1-1.0)")
    settings_table.add_row("Content Type", current_settings["content_type"], "Type of content to generate")
    
    console.print(settings_table)
    
    if Confirm.ask("\nModify settings?"):
        new_max_length = Prompt.ask("Max length", default=str(current_settings["max_length"]))
        new_temperature = Prompt.ask("Temperature", default=str(current_settings["temperature"]))
        new_content_type = Prompt.ask("Content type", default=current_settings["content_type"])
        
        try:
            current_settings["max_length"] = int(new_max_length)
            current_settings["temperature"] = float(new_temperature)
            current_settings["content_type"] = new_content_type
            console.print("✅ Settings updated!")
        except ValueError:
            console.print("❌ Invalid settings, keeping current values")
    
    return current_settings

def display_conversation_history(session: InteractiveSession):
    """Display the conversation history."""
    
    if not session.conversation:
        console.print("📝 No conversation history yet.")
        return
    console.print(f"\n📚 [bold blue]Conversation History[/bold blue] ({len(session.conversation)} exchanges)\n")
    for i, exchange in enumerate(session.conversation, 1):
        console.print(f"[bold cyan]Exchange {i}:[/bold cyan] [dim]({exchange['content_type']})[/dim]")
        console.print(f"[yellow]You:[/yellow] {exchange['user_input']}")
        session.display.display(
            exchange.get('english_content', ''),
            exchange.get('toaripi_content', ''),
            exchange.get('content_type', 'translation')
        )
        console.print()

 # Version resolution now handled by core.versioning

@click.command()
@click.option("--version", "-v", help="Model version to load (e.g., v0.0.3). Defaults to latest if omitted.")
@click.option("--content-type", "-t", default="story", help="Default content type")
@click.option("--temperature", default=0.7, help="Generation temperature")
@click.option("--max-length", default=200, help="Maximum generation length")
@click.option("--save-session", is_flag=True, help="Automatically save session")
def interact(version, content_type, temperature, max_length, save_session):
    """
    Interactive chat interface with the Toaripi SLM model.
    
    This command provides a conversational interface for:
    - Generating educational stories in Toaripi
    - Creating vocabulary exercises
    - Developing dialogues and conversations
    - Translating content
    - Generating comprehension questions
    """
    
    console.print("💬 [bold blue]Toaripi SLM Interactive Mode[/bold blue]\n")
    
    # Resolve model version path
    model_path = resolve_version_dir(version)
    if model_path is None:
        console.print("❌ No versioned models found. Train one first: toaripi train")
        return
    console.print(f"� Using model version directory: [cyan]{model_path}[/cyan]")
    
    # Initialize generator and session
    generator = ToaripiGenerator(model_path, console=console)
    if not generator.load():
        console.print("❌ Failed to load model. Exiting.")
        return
    session = _new_session()
    
    # Generation settings
    settings = {
        "max_length": max_length,
        "temperature": temperature,
        "content_type": content_type
    }
    
    # Welcome message
    model_status = "🤖 Trained SLM"
    welcome_panel = Panel(
        f"""
        Welcome to the Toaripi SLM Interactive Mode! 🎉
        
    Current model: [cyan]{model_path}[/cyan] ({model_status})
    Version: [cyan]{version or 'latest'}[/cyan]
        Content type: [green]{settings['content_type']}[/green]
        
        [bold cyan]Features:[/bold cyan]
        • Side-by-side English ↔ Toaripi display
        • Token weight visualization from SLM attention
        • Word alignment between languages
        • Real-time model generation (when trained model is loaded)
        
        Type your educational content requests in English, and the model will generate 
        appropriate content in Toaripi language with visual token analysis.
        
        Type [yellow]/help[/yellow] for commands or [yellow]/quit[/yellow] to exit.
        """,
        title="🌟 Enhanced Interactive Session Started",
        border_style="green"
    )
    
    console.print(welcome_panel)
    
    # Main interaction loop
    try:
        while True:
            console.print()
            user_input = Prompt.ask("[bold blue]You[/bold blue]", default="").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                command = user_input[1:].lower().split()
                
                if command[0] in ["quit", "exit"]:
                    break
                    
                elif command[0] == "help":
                    show_help_panel()
                    continue
                    
                elif command[0] == "type" and len(command) > 1:
                    new_type = command[1]
                    if new_type in ["story", "vocabulary", "dialogue", "questions", "translation", "chat"]:
                        settings["content_type"] = new_type
                        console.print(f"✅ Content type changed to: [green]{new_type}[/green]")
                        if new_type == "chat":
                            console.print("💬 [cyan]Chat mode enabled![/cyan] Ask questions like 'What is a dog?' or 'What is water?'")
                    else:
                        console.print("❌ Invalid content type. Use: story, vocabulary, dialogue, questions, translation, chat")
                    continue
                    
                elif command[0] == "settings":
                    settings = show_settings_menu()
                    continue
                    
                elif command[0] == "history":
                    display_conversation_history(session)
                    continue
                    
                elif command[0] == "save":
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = Path("./chat_sessions") / f"session_{timestamp}.json"
                    session.save(save_path)
                    console.print(f"💾 Session saved to: {save_path}")
                    continue
                    
                elif command[0] == "clear":
                    if Confirm.ask("Clear conversation history?"):
                        session.conversation = []
                        console.print("🗑️  Conversation history cleared")
                    continue
                    
                elif command[0] == "weights":
                    session.display.toggle_weights()
                    continue
                    
                elif command[0] == "align":
                    session.display.toggle_alignment()
                    continue
                    
                elif command[0] == "legend":
                    # legend auto-shown after display; re-display by toggling weights twice
                    session.display.toggle_weights(); session.display.toggle_weights()
                    continue
                    
                else:
                    console.print(f"❌ Unknown command: {command[0]}. Type /help for available commands.")
                    continue
            
            # Generate content
            console.print(f"[dim]Generating {settings['content_type']} content...[/dim]")
            
            english_content, toaripi_content = generator.bilingual(
                prompt=user_input,
                content_type=settings["content_type"],
                max_length=settings["max_length"],
                temperature=settings["temperature"],
            )
            
            # Display result using bilingual display
            console.print(f"\n💬 [bold blue]Generated Content[/bold blue] [dim]({settings['content_type']})[/dim]:\n")
            
            session.display.display(english_content, toaripi_content, settings["content_type"])
            
            # Add to conversation history
            session.add(
                user=user_input,
                english=english_content,
                toaripi=toaripi_content,
                content_type=settings["content_type"],
            )
            
            # Auto-save if requested
            if save_session and len(session.conversation) % 5 == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = Path("./chat_sessions") / f"auto_session_{timestamp}.json"
                session.save(save_path)
    
    except KeyboardInterrupt:
        console.print("\n\n⏸️  Session interrupted.")
    
    # Session summary
    console.print(f"\n📊 [bold blue]Session Summary[/bold blue]")
    console.print(f"  • Exchanges: {len(session.conversation)}")
    console.print(f"  • Duration: {datetime.now() - session.start}")
    
    # Offer to save session
    if session.conversation and Confirm.ask("Save this session?", default=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path("./chat_sessions") / f"session_{timestamp}.json"
        session.save(save_path)
        console.print(f"💾 Session saved to: {save_path}")
    
    console.print("\n👋 Thanks for using Toaripi SLM Interactive Mode!")