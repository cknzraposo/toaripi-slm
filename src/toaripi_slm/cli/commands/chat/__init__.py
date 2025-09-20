"""
Enhanced chat and interaction commands.
"""

import click
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.layout import Layout
from rich.live import Live

from ...context import get_context

console = Console()

@click.group()
def chat():
    """Interactive chat and session management commands."""
    pass

@chat.command()
@click.option("--version", "-v", help="Model version to use")
@click.option("--mode", type=click.Choice(["compact", "detailed", "teaching"]), 
              default="detailed", help="Display mode")
@click.option("--content-type", "-t", default="story", help="Default content type")
@click.option("--temperature", type=float, default=0.7, help="Generation temperature")
@click.option("--max-length", type=int, default=200, help="Maximum generation length")
@click.option("--save-session", is_flag=True, help="Auto-save session")
def interactive(version, mode, content_type, temperature, max_length, save_session):
    """Enhanced interactive chat with the Toaripi SLM model."""
    ctx = get_context()
    
    console.print("💬 [bold blue]Toaripi SLM Interactive Chat[/bold blue]\n")
    
    # Find model
    model_path = find_model_version(version)
    if not model_path:
        console.print("❌ No model found. Train a model first with [cyan]toaripi model train[/cyan]")
        return
    
    console.print(f"🤖 Using model: [cyan]{model_path}[/cyan]")
    
    # Initialize session
    session = InteractiveSession(
        model_path=model_path,
        mode=mode,
        content_type=content_type,
        temperature=temperature,
        max_length=max_length
    )
    
    # Welcome panel
    welcome_panel = Panel(
        f"""
        [bold cyan]Enhanced Interactive Session Started! 🌟[/bold cyan]
        
        Model: [green]{model_path}[/green]
        Mode: [yellow]{mode}[/yellow]
        Content Type: [magenta]{content_type}[/magenta]
        
        [bold cyan]Available Commands:[/bold cyan]
        • [yellow]/help[/yellow] - Show all commands
        • [yellow]/mode [compact|detailed|teaching][/yellow] - Change display mode
        • [yellow]/type [story|vocabulary|dialogue|qa|chat][/yellow] - Change content type
        • [yellow]/settings[/yellow] - Adjust generation parameters
        • [yellow]/compare [version][/yellow] - Compare with another model
        • [yellow]/export[/yellow] - Export current session
        • [yellow]/quit[/yellow] - Exit chat
        
        [dim]Type your message to generate Toaripi educational content![/dim]
        """,
        title="🎉 Welcome",
        border_style="green"
    )
    console.print(welcome_panel)
    
    # Main chat loop
    start_chat_session(session, save_session)

def start_chat_session(session: 'InteractiveSession', auto_save: bool):
    """Main chat interaction loop."""
    
    try:
        while True:
            console.print()
            user_input = Prompt.ask("[bold blue]You[/bold blue]", default="").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                command_result = handle_chat_command(user_input, session)
                if command_result == "quit":
                    break
                continue
            
            # Generate response
            console.print("[dim]Generating response...[/dim]")
            response = session.generate_response(user_input)
            
            # Display response based on mode
            display_response(response, session.mode)
            
            # Auto-save if enabled
            if auto_save and len(session.conversation) % 5 == 0:
                session.save_auto()
    
    except KeyboardInterrupt:
        console.print("\n\n⏸️  Session interrupted.")
    
    # Session summary
    display_session_summary(session)

def handle_chat_command(command: str, session: 'InteractiveSession') -> Optional[str]:
    """Handle chat commands and return action result."""
    
    parts = command[1:].lower().split()
    cmd = parts[0]
    
    if cmd in ["quit", "exit"]:
        if Confirm.ask("Save session before exiting?", default=True):
            session.save()
        return "quit"
    
    elif cmd == "help":
        show_enhanced_help()
    
    elif cmd == "mode" and len(parts) > 1:
        new_mode = parts[1]
        if new_mode in ["compact", "detailed", "teaching"]:
            session.mode = new_mode
            console.print(f"✅ Display mode changed to: [green]{new_mode}[/green]")
        else:
            console.print("❌ Invalid mode. Use: compact, detailed, or teaching")
    
    elif cmd == "type" and len(parts) > 1:
        new_type = parts[1]
        if new_type in ["story", "vocabulary", "dialogue", "qa", "chat", "translation"]:
            session.content_type = new_type
            console.print(f"✅ Content type changed to: [green]{new_type}[/green]")
            
            # Show type-specific tips
            show_content_type_tips(new_type)
        else:
            console.print("❌ Invalid content type. Use: story, vocabulary, dialogue, qa, chat, translation")
    
    elif cmd == "settings":
        adjust_settings(session)
    
    elif cmd == "compare" and len(parts) > 1:
        compare_with_version(session, parts[1])
    
    elif cmd == "export":
        export_session(session)
    
    elif cmd == "history":
        display_conversation_history(session)
    
    elif cmd == "clear":
        if Confirm.ask("Clear conversation history?"):
            session.clear_history()
            console.print("🗑️  Conversation history cleared")
    
    elif cmd == "save":
        session.save()
        console.print("💾 Session saved")
    
    elif cmd == "stats":
        display_session_stats(session)
    
    else:
        console.print(f"❌ Unknown command: {cmd}. Type [yellow]/help[/yellow] for available commands.")
    
    return None

def show_enhanced_help():
    """Show comprehensive help for interactive mode."""
    
    help_panel = Panel(
        """
        [bold cyan]Interactive Chat Commands[/bold cyan]
        
        [bold yellow]Basic Commands:[/bold yellow]
        • [yellow]/help[/yellow] - Show this help
        • [yellow]/quit[/yellow] or [yellow]/exit[/yellow] - Exit chat
        • [yellow]/save[/yellow] - Save current session
        • [yellow]/clear[/yellow] - Clear conversation history
        
        [bold yellow]Content & Display:[/bold yellow]
        • [yellow]/type <type>[/yellow] - Change content type
          [dim]Available: story, vocabulary, dialogue, qa, chat, translation[/dim]
        • [yellow]/mode <mode>[/yellow] - Change display mode
          [dim]Available: compact, detailed, teaching[/dim]
        
        [bold yellow]Configuration:[/bold yellow]
        • [yellow]/settings[/yellow] - Adjust generation parameters
        • [yellow]/compare <version>[/yellow] - Compare with another model
        
        [bold yellow]Session Management:[/bold yellow]
        • [yellow]/history[/yellow] - View conversation history
        • [yellow]/export[/yellow] - Export session to file
        • [yellow]/stats[/yellow] - Show session statistics
        
        [bold yellow]Content Types:[/bold yellow]
        • [green]story[/green] - Educational stories for students
        • [green]vocabulary[/green] - Word lists with translations
        • [green]dialogue[/green] - Conversations between characters
        • [green]qa[/green] - Question and answer pairs
        • [green]chat[/green] - Simple vocabulary lookup
        • [green]translation[/green] - English to Toaripi translation
        
        [bold yellow]Display Modes:[/bold yellow]
        • [blue]compact[/blue] - Minimal output
        • [blue]detailed[/blue] - Full linguistic analysis (default)
        • [blue]teaching[/blue] - Educational explanations included
        """,
        title="💡 Help",
        border_style="blue"
    )
    console.print(help_panel)

def show_content_type_tips(content_type: str):
    """Show tips for specific content types."""
    
    tips = {
        "story": "📖 Try: 'Write a story about children helping with fishing'",
        "vocabulary": "📝 Try: 'Generate vocabulary for animals' or 'Words about family'",
        "dialogue": "💬 Try: 'Create a dialogue between a teacher and student'",
        "qa": "❓ Try: 'Create questions about water safety'",
        "chat": "🗣️ Try: 'What is a dog?' or 'How do you say water?'",
        "translation": "🔄 Try: 'Translate: The children are playing by the river'"
    }
    
    if content_type in tips:
        console.print(f"💡 [dim]{tips[content_type]}[/dim]")

def adjust_settings(session: 'InteractiveSession'):
    """Interactive settings adjustment."""
    
    settings_table = Table(show_header=True, header_style="bold magenta")
    settings_table.add_column("Setting", style="cyan")
    settings_table.add_column("Current Value", style="green")
    settings_table.add_column("Description", style="dim")
    
    settings_table.add_row("Temperature", f"{session.temperature}", "Creativity (0.1-1.0)")
    settings_table.add_row("Max Length", f"{session.max_length}", "Maximum response length")
    settings_table.add_row("Content Type", f"{session.content_type}", "Type of content to generate")
    settings_table.add_row("Display Mode", f"{session.mode}", "How responses are displayed")
    
    console.print("⚙️  Current Settings:")
    console.print(settings_table)
    
    if Confirm.ask("\nAdjust settings?", default=False):
        # Temperature
        new_temp = Prompt.ask(
            f"Temperature (current: {session.temperature})", 
            default=str(session.temperature)
        )
        try:
            session.temperature = float(new_temp)
        except ValueError:
            console.print("⚠️  Invalid temperature value")
        
        # Max length
        new_length = Prompt.ask(
            f"Max length (current: {session.max_length})",
            default=str(session.max_length)
        )
        try:
            session.max_length = int(new_length)
        except ValueError:
            console.print("⚠️  Invalid length value")
        
        console.print("✅ Settings updated!")

def display_response(response: Dict[str, Any], mode: str):
    """Display generated response based on mode."""
    
    if mode == "compact":
        # Minimal display
        console.print(f"[bold green]Toaripi:[/bold green] {response.get('toaripi', '')}")
        console.print(f"[bold blue]English:[/bold blue] {response.get('english', '')}")
    
    elif mode == "detailed":
        # Full display with analysis
        response_panel = Panel(
            f"""
            [bold green]Toaripi Text:[/bold green]
            {response.get('toaripi', '')}
            
            [bold blue]English Text:[/bold blue] 
            {response.get('english', '')}
            
            [bold yellow]Analysis:[/bold yellow]
            • Content Type: {response.get('content_type', 'unknown')}
            • Confidence: {response.get('confidence', 0):.1%}
            • Word Count: {len(response.get('toaripi', '').split())} words
            """,
            title="🤖 Generated Response",
            border_style="green"
        )
        console.print(response_panel)
    
    elif mode == "teaching":
        # Educational display with explanations
        console.print("📚 [bold green]Educational Response[/bold green]")
        console.print(f"\n[bold]Toaripi:[/bold] {response.get('toaripi', '')}")
        console.print(f"[bold]English:[/bold] {response.get('english', '')}")
        
        # Add educational notes
        if response.get('content_type') == 'vocabulary':
            console.print("\n[bold cyan]Learning Notes:[/bold cyan]")
            console.print("• Practice pronunciation by reading aloud")
            console.print("• Try using these words in sentences")
        elif response.get('content_type') == 'story':
            console.print("\n[bold cyan]Learning Notes:[/bold cyan]")
            console.print("• Notice the sentence structure")
            console.print("• Identify new vocabulary words")
        
        # Show key vocabulary if available
        if response.get('key_words'):
            vocab_table = Table(show_header=True, header_style="bold magenta")
            vocab_table.add_column("Toaripi", style="green")
            vocab_table.add_column("English", style="blue")
            
            for word_pair in response.get('key_words', []):
                vocab_table.add_row(word_pair.get('toaripi', ''), word_pair.get('english', ''))
            
            console.print("\n📝 Key Vocabulary:")
            console.print(vocab_table)

@chat.command()
@click.option("--session-dir", type=Path, default="chat_sessions", help="Sessions directory")
def sessions(session_dir):
    """Manage chat sessions."""
    console.print("📁 [bold blue]Chat Sessions[/bold blue]\n")
    
    if not session_dir.exists():
        console.print(f"No sessions directory found at {session_dir}")
        return
    
    # List all session files
    session_files = list(session_dir.glob("*.json"))
    
    if not session_files:
        console.print("No saved sessions found.")
        return
    
    # Display sessions table
    sessions_table = Table(show_header=True, header_style="bold magenta")
    sessions_table.add_column("ID", style="cyan")
    sessions_table.add_column("Date", style="green")
    sessions_table.add_column("Messages", style="yellow")
    sessions_table.add_column("Model", style="blue")
    sessions_table.add_column("File", style="dim")
    
    for i, session_file in enumerate(sorted(session_files, key=lambda x: x.stat().st_mtime, reverse=True)):
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            date_str = session_data.get('timestamp', 'Unknown')[:10]
            msg_count = len(session_data.get('conversation', []))
            model = session_data.get('model_path', 'Unknown').split('/')[-1]
            
            sessions_table.add_row(
                str(i + 1),
                date_str,
                str(msg_count),
                model,
                session_file.name
            )
        except Exception:
            sessions_table.add_row(str(i + 1), "Error", "0", "Unknown", session_file.name)
    
    console.print(sessions_table)

@chat.command()
@click.argument("session_id", type=int)
@click.option("--session-dir", type=Path, default="chat_sessions", help="Sessions directory")
def resume(session_id, session_dir):
    """Resume a previous chat session."""
    console.print(f"🔄 [bold blue]Resuming Session {session_id}[/bold blue]\n")
    
    session_files = sorted(session_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if session_id < 1 or session_id > len(session_files):
        console.print(f"❌ Invalid session ID. Use [cyan]toaripi chat sessions[/cyan] to list available sessions.")
        return
    
    session_file = session_files[session_id - 1]
    
    try:
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        console.print(f"📄 Loading session from: [cyan]{session_file.name}[/cyan]")
        
        # Display session info
        info_panel = Panel(
            f"""
            [bold cyan]Session Information[/bold cyan]
            
            Date: {session_data.get('timestamp', 'Unknown')}
            Model: {session_data.get('model_path', 'Unknown')}
            Messages: {len(session_data.get('conversation', []))}
            Content Type: {session_data.get('content_type', 'Unknown')}
            """,
            title="📋 Session Info",
            border_style="blue"
        )
        console.print(info_panel)
        
        # Show conversation history
        conversation = session_data.get('conversation', [])
        if conversation:
            console.print("\n💬 [bold blue]Previous Conversation:[/bold blue]")
            for msg in conversation[-5:]:  # Show last 5 messages
                if msg.get('role') == 'user':
                    console.print(f"[bold blue]You:[/bold blue] {msg.get('content', '')}")
                else:
                    console.print(f"[bold green]Toaripi:[/bold green] {msg.get('content', '')}")
        
        if Confirm.ask("\nContinue this session?", default=True):
            # Resume session with loaded data
            session = InteractiveSession.from_data(session_data)
            start_chat_session(session, True)
        
    except Exception as e:
        console.print(f"❌ Error loading session: {e}")

# Helper classes and functions

class InteractiveSession:
    """Enhanced interactive session manager."""
    
    def __init__(self, model_path: str, mode: str = "detailed", 
                 content_type: str = "story", temperature: float = 0.7, 
                 max_length: int = 200):
        self.model_path = model_path
        self.mode = mode
        self.content_type = content_type
        self.temperature = temperature
        self.max_length = max_length
        self.conversation = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_response(self, user_input: str) -> Dict[str, Any]:
        """Generate response to user input."""
        # Mock response - replace with actual model inference
        response = {
            "english": f"English response to: {user_input}",
            "toaripi": f"Toaripi response to: {user_input}",
            "content_type": self.content_type,
            "confidence": 0.85,
            "key_words": [
                {"toaripi": "ruru", "english": "dog"},
                {"toaripi": "peni", "english": "water"}
            ]
        }
        
        # Add to conversation
        self.conversation.append({"role": "user", "content": user_input})
        self.conversation.append({"role": "assistant", "content": response})
        
        return response
    
    def save(self, path: Optional[Path] = None):
        """Save session to file."""
        if not path:
            path = Path("chat_sessions") / f"session_{self.session_id}.json"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        session_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "model_path": self.model_path,
            "mode": self.mode,
            "content_type": self.content_type,
            "temperature": self.temperature,
            "max_length": self.max_length,
            "conversation": self.conversation
        }
        
        with open(path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        console.print(f"💾 Session saved to: [cyan]{path}[/cyan]")
    
    def save_auto(self):
        """Auto-save with timestamp."""
        auto_path = Path("chat_sessions") / f"auto_session_{self.session_id}.json"
        self.save(auto_path)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation = []
    
    @classmethod
    def from_data(cls, session_data: Dict[str, Any]) -> 'InteractiveSession':
        """Create session from saved data."""
        session = cls(
            model_path=session_data.get('model_path', ''),
            mode=session_data.get('mode', 'detailed'),
            content_type=session_data.get('content_type', 'story'),
            temperature=session_data.get('temperature', 0.7),
            max_length=session_data.get('max_length', 200)
        )
        session.conversation = session_data.get('conversation', [])
        session.session_id = session_data.get('session_id', session.session_id)
        return session

def find_model_version(version: Optional[str]) -> Optional[str]:
    """Find model by version or return latest."""
    # Mock implementation - replace with actual model finding logic
    if version:
        return f"./models/hf/toaripi-slm-{version}"
    else:
        return "./models/hf/toaripi-slm-latest"

def compare_with_version(session: InteractiveSession, version: str):
    """Compare current model with another version."""
    console.print(f"⚖️  Comparing with version: [cyan]{version}[/cyan]")
    console.print("🚧 Feature coming soon!")

def export_session(session: InteractiveSession):
    """Export session to various formats."""
    console.print("📤 Exporting session...")
    
    formats = ["JSON", "Text", "PDF", "HTML"]
    format_choice = Prompt.ask("Export format", choices=[f.lower() for f in formats], default="json")
    
    if format_choice == "json":
        session.save()
    else:
        console.print(f"🚧 {format_choice.upper()} export coming soon!")

def display_conversation_history(session: InteractiveSession):
    """Display conversation history."""
    if not session.conversation:
        console.print("No conversation history.")
        return
    
    console.print("💬 [bold blue]Conversation History[/bold blue]\n")
    
    for i, msg in enumerate(session.conversation[-10:], 1):  # Show last 10 messages
        if msg.get('role') == 'user':
            console.print(f"{i}. [bold blue]You:[/bold blue] {msg.get('content', '')}")
        else:
            response = msg.get('content', {})
            if isinstance(response, dict):
                console.print(f"{i}. [bold green]Toaripi:[/bold green] {response.get('toaripi', '')}")
            else:
                console.print(f"{i}. [bold green]Toaripi:[/bold green] {response}")

def display_session_stats(session: InteractiveSession):
    """Display session statistics."""
    stats_table = Table(show_header=True, header_style="bold magenta")
    stats_table.add_column("Statistic", style="cyan")
    stats_table.add_column("Value", style="green")
    
    user_messages = len([msg for msg in session.conversation if msg.get('role') == 'user'])
    assistant_messages = len([msg for msg in session.conversation if msg.get('role') == 'assistant'])
    
    stats_table.add_row("Session ID", session.session_id)
    stats_table.add_row("Model", session.model_path.split('/')[-1])
    stats_table.add_row("Content Type", session.content_type)
    stats_table.add_row("Display Mode", session.mode)
    stats_table.add_row("User Messages", str(user_messages))
    stats_table.add_row("Generated Responses", str(assistant_messages))
    stats_table.add_row("Temperature", f"{session.temperature}")
    stats_table.add_row("Max Length", f"{session.max_length}")
    
    console.print("📊 Session Statistics:")
    console.print(stats_table)

def display_session_summary(session: InteractiveSession):
    """Display session summary at end."""
    summary_panel = Panel(
        f"""
        [bold blue]Session Summary[/bold blue]
        
        Messages exchanged: {len(session.conversation)}
        Content type: {session.content_type}
        Model used: {session.model_path.split('/')[-1]}
        
        [dim]Session ID: {session.session_id}[/dim]
        """,
        title="📊 Summary",
        border_style="blue"
    )
    console.print(summary_panel)
    console.print("\n👋 Thanks for using Toaripi SLM Interactive Chat!")