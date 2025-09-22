"""
Smart Welcome System

Provides intelligent welcome experiences that adapt to user context,
project state, and experience level.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..framework import CLIContext
from ..user_profiles import UserProfile, create_default_profile


class SmartWelcome:
    """Intelligent welcome system that adapts to user and project context."""
    
    def __init__(self, context: CLIContext):
        self.context = context
        self.console = Console() if RICH_AVAILABLE else None
    
    def show_welcome(self, force_onboarding: bool = False) -> None:
        """Show appropriate welcome based on user state."""
        
        # Create or use existing session profile
        if not self.context.user_profile:
            self.context.user_profile = create_default_profile()
        
        # Show welcome with session profile
        self._show_session_welcome()
        
        # Show project analysis and next steps
        self._show_project_status()
        self._show_contextual_guidance()
    
    def _show_session_welcome(self) -> None:
        """Show welcome for current session."""
        
        if not RICH_AVAILABLE:
            print("🌟 Welcome to Toaripi SLM!")
            print("Educational AI for creating learning materials in Toaripi.")
            return
        
        welcome_text = Text()
        welcome_text.append("🌟 ", style="bold yellow")
        welcome_text.append("Toaripi SLM", style="bold blue")
        welcome_text.append(" - Educational AI �\n\n", style="bold yellow")
        
        welcome_text.append("📚 ", style="bold")
        welcome_text.append("Create stories, vocabulary, and learning materials\n", style="bold blue")
        welcome_text.append("Culturally appropriate content for Toaripi learners\n\n", style="cyan")
        
        welcome_text.append("✨ Features:\n", style="bold green")
        welcome_text.append("  � Stories for primary school students\n", style="green")
        welcome_text.append("  📝 Vocabulary exercises and activities\n", style="green")
        welcome_text.append("  🏫 Works offline for classroom use\n", style="green")
        welcome_text.append("  🤖 AI that understands educational needs\n\n", style="green")
        
        welcome_text.append("Ready to create educational content? Let's get started! �", style="cyan")
        
        panel = Panel(
            Align.center(welcome_text),
            title="🎉 Educational Content Generator",
            border_style="blue",
            padding=(1, 2)
        )
        
        self.console.print()
        self.console.print(panel)
        self.console.print()
    
    def _show_project_status(self) -> None:
        """Show current project status and health."""
        
        status = self._analyze_project_state()
        
        if not RICH_AVAILABLE:
            print("\nProject Status:")
            for component, info in status.items():
                status_icon = "✅" if info["ok"] else "⚠️"
                print(f"  {status_icon} {component}: {info['message']}")
            return
        
        # Create status table
        status_table = Table(show_header=False, box=None, padding=(0, 1))
        status_table.add_column("Component", style="bold")
        status_table.add_column("Status", justify="left")
        
        for component, info in status.items():
            if info["ok"]:
                status_table.add_row(
                    f"✅ {component}",
                    f"[green]{info['message']}[/green]"
                )
            else:
                status_table.add_row(
                    f"⚠️  {component}",
                    f"[yellow]{info['message']}[/yellow]"
                )
        
        status_panel = Panel(
            status_table,
            title="📊 Project Status",
            border_style="cyan"
        )
        
        self.console.print(status_panel)
    
    def _analyze_project_state(self) -> Dict[str, Dict[str, Any]]:
        """Analyze current project state."""
        
        status = {}
        
        # Check Python environment
        try:
            import transformers, torch
            status["Python Environment"] = {
                "ok": True,
                "message": "All required packages available"
            }
        except ImportError as e:
            status["Python Environment"] = {
                "ok": False,
                "message": f"Missing packages: {e.name}"
            }
        
        # Check data directory
        data_dir = self.context.working_directory / "data"
        if data_dir.exists():
            data_files = list(data_dir.rglob("*.csv"))
            if data_files:
                status["Training Data"] = {
                    "ok": True,
                    "message": f"Found {len(data_files)} data files"
                }
            else:
                status["Training Data"] = {
                    "ok": False,
                    "message": "No training data found"
                }
        else:
            status["Training Data"] = {
                "ok": False,
                "message": "Data directory missing"
            }
        
        # Check models directory
        models_dir = self.context.working_directory / "models"
        if models_dir.exists():
            model_files = list(models_dir.rglob("*.bin")) + list(models_dir.rglob("*.gguf"))
            if model_files:
                status["Trained Models"] = {
                    "ok": True,
                    "message": f"Found {len(model_files)} trained models"
                }
            else:
                status["Trained Models"] = {
                    "ok": False,
                    "message": "No trained models yet"
                }
        else:
            status["Trained Models"] = {
                "ok": False,
                "message": "Models directory missing"
            }
        
        return status
    
    def _show_contextual_guidance(self) -> None:
        """Show contextual guidance based on project state and user profile."""
        
        # Get suggestions from guidance system
        from ..guidance_system import GuidanceEngine
        guidance = GuidanceEngine(self.context)
        suggestions = guidance.suggest_next_actions(max_suggestions=3)
        
        if not suggestions:
            return
        
        if not RICH_AVAILABLE:
            print("\nSuggested next steps:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion.title}: toaripi {suggestion.command}")
            return
        
        # Create suggestions display
        suggestions_text = Text()
        suggestions_text.append("🎯 What would you like to do?\n\n", style="bold cyan")
        
        for i, suggestion in enumerate(suggestions, 1):
            suggestions_text.append(f"{i}. ", style="bold white")
            suggestions_text.append(f"{suggestion.emoji} ", style="yellow")
            suggestions_text.append(f"{suggestion.title}", style="cyan")
            suggestions_text.append(f" ({suggestion.estimated_time})\n", style="dim")
            suggestions_text.append(f"   {suggestion.description}\n", style="white")
            suggestions_text.append(f"   💻 ", style="dim")
            suggestions_text.append(f"toaripi {suggestion.command}", style="bold green")
            suggestions_text.append("\n\n")
        
        # Add natural language option
        suggestions_text.append("💬 ", style="yellow")
        suggestions_text.append("Or just tell me what you want to do!", style="cyan")
        suggestions_text.append("\n   Example: \"train a model\" or \"check my data\"", style="dim")
        
        suggestions_panel = Panel(
            suggestions_text,
            title="🧭 Getting Started",
            border_style="cyan"
        )
        
        self.console.print()
        self.console.print(suggestions_panel)
    
    def handle_natural_language_request(self, user_input: str) -> Optional[str]:
        """Handle natural language request from welcome screen."""
        
        from ..guidance_system import GuidanceEngine
        guidance = GuidanceEngine(self.context)
        suggested_command = guidance.parse_natural_language(user_input)
        
        if suggested_command:
            if RICH_AVAILABLE:
                should_run = Confirm.ask(
                    f"💡 It sounds like you want to run: [bold green]toaripi {suggested_command}[/bold green]\n   Is this correct?",
                    default=True
                )
            else:
                response = input(f"💡 Run 'toaripi {suggested_command}'? (Y/n): ").strip().lower()
                should_run = response in ["", "y", "yes"]
            
            if should_run:
                return suggested_command
            else:
                self._show_command_help()
        else:
            if RICH_AVAILABLE:
                self.console.print("🤔 I'm not sure what you'd like to do. Let me show you the available options.")
            else:
                print("🤔 I'm not sure what you'd like to do. Here are the available options:")
            self._show_command_help()
        
        return None
    
    def _show_command_help(self) -> None:
        """Show basic command help."""
        
        if not RICH_AVAILABLE:
            print("\nMain commands:")
            print("  toaripi train start    - Train a new model")
            print("  toaripi data validate  - Check your training data")
            print("  toaripi interact       - Generate educational content")
            print("  toaripi status         - Check system status")
            print("  toaripi help           - Get detailed help")
            return
        
        help_table = Table(title="📚 Main Commands", show_header=True, header_style="bold magenta")
        help_table.add_column("Command", style="bold cyan")
        help_table.add_column("Description", style="white")
        help_table.add_column("Good for", style="dim")
        
        commands = [
            ("train start", "Train a new model", "First-time setup"),
            ("data validate", "Check training data", "Before training"),
            ("interact", "Generate content", "After training"),
            ("status", "System health check", "Troubleshooting"),
            ("help", "Detailed help", "Learning the system")
        ]
        
        for cmd, desc, good_for in commands:
            help_table.add_row(f"toaripi {cmd}", desc, good_for)
        
        self.console.print()
        self.console.print(help_table)
        self.console.print()
        self.console.print("💡 [cyan]Tip:[/cyan] You can also describe what you want to do in plain English!")
    
    def interactive_mode(self) -> None:
        """Run interactive welcome mode where user can ask questions."""
        
        self.show_welcome()
        
        if not RICH_AVAILABLE:
            print("\nWhat would you like to do? (type 'help' for options, 'quit' to exit)")
        else:
            self.console.print("\n💬 [bold cyan]What would you like to do?[/bold cyan]")
            self.console.print("   Type your request in plain English, or 'help' for options")
            self.console.print("   Examples: 'train a model', 'check my data', 'generate a story'")
            self.console.print("   Type 'quit' or press Ctrl+C to exit\n")
        
        while True:
            try:
                if RICH_AVAILABLE:
                    user_input = Prompt.ask("🎯 [bold blue]You[/bold blue]", default="").strip()
                else:
                    user_input = input("🎯 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    if RICH_AVAILABLE:
                        self.console.print("👋 [green]Thanks for using Toaripi SLM! Happy teaching![/green]")
                    else:
                        print("👋 Thanks for using Toaripi SLM! Happy teaching!")
                    break
                
                if user_input.lower() in ['help', 'h']:
                    self._show_command_help()
                    continue
                
                # Try to parse natural language
                suggested_command = self.handle_natural_language_request(user_input)
                if suggested_command:
                    # This would execute the command
                    if RICH_AVAILABLE:
                        self.console.print(f"🚀 [green]Running:[/green] toaripi {suggested_command}")
                    else:
                        print(f"🚀 Running: toaripi {suggested_command}")
                    break
                
            except KeyboardInterrupt:
                if RICH_AVAILABLE:
                    self.console.print("\n👋 [yellow]Goodbye![/yellow]")
                else:
                    print("\n👋 Goodbye!")
                break
            except Exception as e:
                if RICH_AVAILABLE:
                    self.console.print(f"❌ [red]Error:[/red] {e}")
                else:
                    print(f"❌ Error: {e}")