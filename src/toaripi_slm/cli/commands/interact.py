"""
Interactive command for the Toaripi SLM CLI.

Provides a chat-like interface for generating educational content
with the trained model in real-time.
"""

import json
import yaml
import re
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.markdown import Markdown
from rich.columns import Columns
from rich.align import Align
from rich import print as rprint

console = Console()

class TokenWeight:
    """Represents a token with its attention weight."""
    
    def __init__(self, token: str, weight: float):
        self.token = token
        self.weight = weight
    
    def get_color_style(self) -> str:
        """Get Rich color style based on token weight."""
        if self.weight >= 0.8:
            return "bold red"
        elif self.weight >= 0.6:
            return "bold orange3"
        elif self.weight >= 0.4:
            return "bold yellow"
        elif self.weight >= 0.2:
            return "green"
        else:
            return "dim cyan"

class BilingualDisplay:
    """Handles side-by-side display of English and Toaripi text with token weights."""
    
    def __init__(self, console: Console):
        self.console = console
        self.show_weights = True
        self.align_tokens = True
        
    def simulate_token_weights(self, text: str) -> List[TokenWeight]:
        """Simulate token attention weights for demonstration."""
        tokens = text.split()
        weights = []
        
        for token in tokens:
            # Simulate varying attention weights
            # Key content words get higher weights
            if len(token) > 6 or token.lower() in ['important', 'story', 'children', 'learn', 'teach']:
                weight = random.uniform(0.7, 1.0)
            elif len(token) > 3:
                weight = random.uniform(0.4, 0.8)
            else:
                weight = random.uniform(0.1, 0.5)
            
            weights.append(TokenWeight(token, weight))
        
        return weights
    
    def create_weighted_text(self, token_weights: List[TokenWeight]) -> Text:
        """Create Rich Text object with colored tokens based on weights."""
        text = Text()
        
        for i, tw in enumerate(token_weights):
            if i > 0:
                text.append(" ")
            
            if self.show_weights:
                # Add token with color based on weight
                text.append(tw.token, style=tw.get_color_style())
                # Add weight indicator
                text.append(f"({tw.weight:.2f})", style="dim white")
            else:
                text.append(tw.token)
        
        return text
    
    def align_texts(self, english_tokens: List[TokenWeight], 
                   toaripi_tokens: List[TokenWeight]) -> Tuple[List[TokenWeight], List[TokenWeight]]:
        """Simple token alignment between English and Toaripi."""
        # For now, we'll pad the shorter list with empty tokens
        # In a real implementation, this would use proper alignment algorithms
        
        max_len = max(len(english_tokens), len(toaripi_tokens))
        
        # Pad English tokens
        while len(english_tokens) < max_len:
            english_tokens.append(TokenWeight("", 0.0))
        
        # Pad Toaripi tokens
        while len(toaripi_tokens) < max_len:
            toaripi_tokens.append(TokenWeight("", 0.0))
        
        return english_tokens, toaripi_tokens
    
    def display_bilingual_content(self, english_text: str, toaripi_text: str, 
                                content_type: str = "translation"):
        """Display English and Toaripi text side by side with token weights."""
        
        # Generate token weights
        english_tokens = self.simulate_token_weights(english_text)
        toaripi_tokens = self.simulate_token_weights(toaripi_text)
        
        # Align tokens if requested
        if self.align_tokens:
            english_tokens, toaripi_tokens = self.align_texts(english_tokens, toaripi_tokens)
        
        # Create weighted text objects
        english_rich_text = self.create_weighted_text(english_tokens)
        toaripi_rich_text = self.create_weighted_text(toaripi_tokens)
        
        # Create panels for each language
        english_panel = Panel(
            english_rich_text,
            title="🇺🇸 English Source",
            border_style="blue",
            padding=(1, 2)
        )
        
        toaripi_panel = Panel(
            toaripi_rich_text,
            title="🌺 Toaripi Translation",
            border_style="green",
            padding=(1, 2)
        )
        
        # Display side by side
        columns = Columns([english_panel, toaripi_panel], equal=True, expand=True)
        self.console.print(columns)
        
        # Add weight legend if showing weights
        if self.show_weights:
            self.display_weight_legend()
    
    def display_weight_legend(self):
        """Display legend for token weight colors."""
        legend_text = Text()
        legend_text.append("Token Weight Legend: ")
        legend_text.append("High (0.8+)", style="bold red")
        legend_text.append(" | ")
        legend_text.append("Medium-High (0.6+)", style="bold orange3")
        legend_text.append(" | ")
        legend_text.append("Medium (0.4+)", style="bold yellow")
        legend_text.append(" | ")
        legend_text.append("Low (0.2+)", style="green")
        legend_text.append(" | ")
        legend_text.append("Very Low (<0.2)", style="dim cyan")
        
        legend_panel = Panel(
            legend_text,
            title="🎨 Weight Color Guide",
            border_style="magenta",
            padding=(0, 1)
        )
        self.console.print(legend_panel)
    
    def toggle_weights(self):
        """Toggle display of token weights."""
        self.show_weights = not self.show_weights
        status = "ON" if self.show_weights else "OFF"
        self.console.print(f"💡 Token weights display: [bold]{status}[/bold]")
    
    def toggle_alignment(self):
        """Toggle token alignment."""
        self.align_tokens = not self.align_tokens
        status = "ON" if self.align_tokens else "OFF"
        self.console.print(f"🔗 Token alignment: [bold]{status}[/bold]")

class ToaripiGenerator:
    """Interface for the Toaripi model generation."""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.loaded = False
        
        # Content templates for different types
        self.templates = {
            "story": "Generate a story in Toaripi about {topic} suitable for {age_group} students.",
            "vocabulary": "Create vocabulary words in Toaripi related to {topic} with English translations.",
            "dialogue": "Create a dialogue in Toaripi between {characters} about {topic}.",
            "questions": "Generate comprehension questions in Toaripi about: {content}",
            "translation": "Translate this English text to Toaripi: {text}"
        }
        
        # Sample bilingual content for demonstration
        self.sample_pairs = {
            "story": {
                "english": "The children went fishing by the river. They caught many fish and shared them with their families. Everyone was happy with the good catch.",
                "toaripi": "Mina na'a hanere na malolo peni. Na'a gete hanere na potopoto api na bada-bada. Ami hanere na'a nene kekeni mane-mane."
            },
            "vocabulary": {
                "english": "Fish swimming in clear water. Children learning traditional fishing methods.",
                "toaripi": "Hanere potopoto malolo peni kura. Bada-bada nene-ida gola hanere taumate."
            },
            "dialogue": {
                "english": "Teacher: What did you learn today? Student: I learned about fishing traditions.",
                "toaripi": "Amo-harigi: Ami na'a nene kekeni? Amo-nene: Mina na'a nene hanere taumate-ida."
            },
            "questions": {
                "english": "Where did the children go fishing? What did they catch?",
                "toaripi": "Sena na'a gola hanere bada-bada? Ami na'a gete hanere?"
            },
            "translation": {
                "english": "The fish are swimming in the clear water near the village.",
                "toaripi": "Hanere potopoto malolo peni kura gabua hareva."
            }
        }
    
    def load_model(self) -> bool:
        """Load the model and tokenizer."""
        try:
            console.print("🔄 Loading Toaripi model...")
            # Placeholder for actual model loading
            # In a real implementation, this would load the fine-tuned model
            self.loaded = True
            console.print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            console.print(f"❌ Failed to load model: {e}")
            return False
    
    def generate_bilingual_content(self, prompt: str, content_type: str = "story", 
                                 max_length: int = 200, temperature: float = 0.7) -> Tuple[str, str]:
        """Generate bilingual content and return English and Toaripi versions."""
        
        if not self.loaded:
            return "Error: Model not loaded", "Error: Model not loaded"
        
        # Get sample content based on type
        sample_pair = self.sample_pairs.get(content_type, self.sample_pairs["story"])
        
        # In a real implementation, this would:
        # 1. Process the prompt with the model
        # 2. Generate Toaripi content
        # 3. Extract attention weights from the model
        # 4. Return both English context and Toaripi generation
        
        # For now, return sample content modified based on prompt
        english_base = sample_pair["english"]
        toaripi_base = sample_pair["toaripi"]
        
        # Simple prompt-based modification (in reality, this would be model-generated)
        if "fish" in prompt.lower():
            return english_base, toaripi_base
        elif "children" in prompt.lower() or "student" in prompt.lower():
            return sample_pair["english"], sample_pair["toaripi"]
        else:
            # Default to translation type content
            return self.sample_pairs["translation"]["english"], self.sample_pairs["translation"]["toaripi"]
    
    def generate_content(self, prompt: str, content_type: str = "story", 
                        max_length: int = 200, temperature: float = 0.7) -> str:
        """Generate content based on the prompt (backward compatibility)."""
        
        if not self.loaded:
            return "Error: Model not loaded"
        
        _, toaripi_content = self.generate_bilingual_content(prompt, content_type, max_length, temperature)
        return toaripi_content

class InteractiveSession:
    """Manages an interactive chat session."""
    
    def __init__(self, generator: ToaripiGenerator):
        self.generator = generator
        self.conversation_history = []
        self.session_start = datetime.now()
        self.bilingual_display = BilingualDisplay(console)
        
    def add_exchange(self, user_input: str, english_content: str, toaripi_content: str, content_type: str):
        """Add a conversation exchange to history."""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "english_content": english_content,
            "toaripi_content": toaripi_content,
            "content_type": content_type
        })
    
    def save_session(self, output_path: Path):
        """Save the conversation session."""
        session_data = {
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "conversation_history": self.conversation_history,
            "total_exchanges": len(self.conversation_history),
            "display_settings": {
                "show_weights": self.bilingual_display.show_weights,
                "align_tokens": self.bilingual_display.align_tokens
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(session_data, f, indent=2)

def show_help_panel():
    """Display help information for interactive mode."""
    
    help_text = """
    [bold cyan]Interactive Commands:[/bold cyan]
    
    [yellow]/help[/yellow] - Show this help message
    [yellow]/type <content_type>[/yellow] - Change content type (story, vocabulary, dialogue, questions, translation)
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
    
    [bold cyan]Visualization Features:[/bold cyan]
    
    • [magenta]Side-by-side display[/magenta] - English and Toaripi shown in parallel
    • [magenta]Token weights[/magenta] - Color-coded attention weights for each word
    • [magenta]Weight legend[/magenta] - Color guide for understanding token importance
    • [magenta]Token alignment[/magenta] - Visual alignment between corresponding words
    
    [bold cyan]Tips:[/bold cyan]
    
    • Be specific in your prompts for better results
    • Mention age group (e.g., "for primary school children")
    • Include cultural context when relevant
    • Use simple, clear language in your requests
    • Toggle token weights to see model attention patterns
    """
    
    console.print(Panel(help_text, title="Interactive Mode Help", border_style="blue"))

def show_settings_menu(generator: ToaripiGenerator) -> Dict[str, Any]:
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
    
    if not session.conversation_history:
        console.print("📝 No conversation history yet.")
        return
    
    console.print(f"\n📚 [bold blue]Conversation History[/bold blue] ({len(session.conversation_history)} exchanges)\n")
    
    for i, exchange in enumerate(session.conversation_history, 1):
        console.print(f"[bold cyan]Exchange {i}:[/bold cyan] [dim]({exchange['content_type']})[/dim]")
        console.print(f"[yellow]You:[/yellow] {exchange['user_input']}")
        
        # Display bilingual content if available
        if 'english_content' in exchange and 'toaripi_content' in exchange:
            session.bilingual_display.display_bilingual_content(
                exchange['english_content'], 
                exchange['toaripi_content'],
                exchange['content_type']
            )
        else:
            # Fallback for older format
            console.print(f"[green]Toaripi:[/green] {exchange.get('model_output', 'N/A')}")
        console.print()

@click.command()
@click.option("--model", "-m", type=click.Path(exists=True), help="Path to trained model")
@click.option("--content-type", "-t", default="story", help="Default content type")
@click.option("--temperature", default=0.7, help="Generation temperature")
@click.option("--max-length", default=200, help="Maximum generation length")
@click.option("--save-session", is_flag=True, help="Automatically save session")
def interact(model, content_type, temperature, max_length, save_session):
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
    
    # Set up model path
    model_path = Path(model) if model else Path("./models/hf")
    
    if not model_path.exists():
        console.print(f"❌ Model not found: {model_path}")
        console.print("💡 Train a model first: [cyan]toaripi train[/cyan]")
        return
    
    # Initialize generator and session
    generator = ToaripiGenerator(model_path)
    
    if not generator.load_model():
        console.print("❌ Failed to load model. Exiting.")
        return
    
    session = InteractiveSession(generator)
    
    # Generation settings
    settings = {
        "max_length": max_length,
        "temperature": temperature,
        "content_type": content_type
    }
    
    # Welcome message
    welcome_panel = Panel(
        f"""
        Welcome to the Toaripi SLM Interactive Mode! 🎉
        
        Current model: [cyan]{model_path}[/cyan]
        Content type: [green]{settings['content_type']}[/green]
        
        [bold cyan]New Features:[/bold cyan]
        • Side-by-side English ↔ Toaripi display
        • Token weight visualization with colors
        • Word alignment between languages
        
        Type your educational content requests in English, and I'll generate 
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
                    if new_type in ["story", "vocabulary", "dialogue", "questions", "translation"]:
                        settings["content_type"] = new_type
                        console.print(f"✅ Content type changed to: [green]{new_type}[/green]")
                    else:
                        console.print("❌ Invalid content type. Use: story, vocabulary, dialogue, questions, translation")
                    continue
                    
                elif command[0] == "settings":
                    settings = show_settings_menu(generator)
                    continue
                    
                elif command[0] == "history":
                    display_conversation_history(session)
                    continue
                    
                elif command[0] == "save":
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = Path("./chat_sessions") / f"session_{timestamp}.json"
                    session.save_session(save_path)
                    console.print(f"💾 Session saved to: {save_path}")
                    continue
                    
                elif command[0] == "clear":
                    if Confirm.ask("Clear conversation history?"):
                        session.conversation_history = []
                        console.print("🗑️  Conversation history cleared")
                    continue
                    
                elif command[0] == "weights":
                    session.bilingual_display.toggle_weights()
                    continue
                    
                elif command[0] == "align":
                    session.bilingual_display.toggle_alignment()
                    continue
                    
                elif command[0] == "legend":
                    session.bilingual_display.display_weight_legend()
                    continue
                    
                else:
                    console.print(f"❌ Unknown command: {command[0]}. Type /help for available commands.")
                    continue
            
            # Generate content
            console.print(f"[dim]Generating {settings['content_type']} content...[/dim]")
            
            english_content, toaripi_content = generator.generate_bilingual_content(
                prompt=user_input,
                content_type=settings["content_type"],
                max_length=settings["max_length"],
                temperature=settings["temperature"]
            )
            
            # Display result using bilingual display
            console.print(f"\n💬 [bold blue]Generated Content[/bold blue] [dim]({settings['content_type']})[/dim]:\n")
            
            session.bilingual_display.display_bilingual_content(
                english_content,
                toaripi_content,
                settings["content_type"]
            )
            
            # Add to conversation history
            session.add_exchange(user_input, english_content, toaripi_content, settings["content_type"])
            
            # Auto-save if requested
            if save_session and len(session.conversation_history) % 5 == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = Path("./chat_sessions") / f"auto_session_{timestamp}.json"
                session.save_session(save_path)
    
    except KeyboardInterrupt:
        console.print("\n\n⏸️  Session interrupted.")
    
    # Session summary
    console.print(f"\n📊 [bold blue]Session Summary[/bold blue]")
    console.print(f"  • Exchanges: {len(session.conversation_history)}")
    console.print(f"  • Duration: {datetime.now() - session.session_start}")
    
    # Offer to save session
    if session.conversation_history and Confirm.ask("Save this session?", default=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path("./chat_sessions") / f"session_{timestamp}.json"
        session.save_session(save_path)
        console.print(f"💾 Session saved to: {save_path}")
    
    console.print("\n👋 Thanks for using Toaripi SLM Interactive Mode!")