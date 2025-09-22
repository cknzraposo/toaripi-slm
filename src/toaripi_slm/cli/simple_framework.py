"""
Modern CLI Framework - Fallback Version

A version of the modern CLI framework that works without external dependencies
like Rich, while still providing an enhanced user experience.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class CLIContext:
    """Enhanced CLI context for managing global state with user experience focus."""
    
    # Core configuration
    config_file: Optional[Path] = None
    verbose: bool = False
    quiet: bool = False
    working_directory: Path = field(default_factory=Path.cwd)
    
    # User experience
    user_profile: Optional['UserProfile'] = None
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    interface_style: str = "modern"  # modern, classic, minimal
    
    # Educational focus
    educational_mode: bool = True
    cultural_validation: bool = True
    age_group_focus: str = "primary"
    
    # Runtime state
    current_workflow: Optional[str] = None
    last_command: Optional[str] = None
    
    # Modern CLI components (lazy loaded)
    modern_cli: Optional['ModernCLI'] = None
    guidance_engine: Optional['GuidanceEngine'] = None
    progress_manager: Optional['ProgressManager'] = None


class FallbackConsole:
    """Simple console class that works without Rich."""
    
    def __init__(self):
        self.use_colors = os.name != 'nt' or 'TERM' in os.environ
    
    def print(self, text: str, style: str = None):
        """Print text with optional style (fallback to plain text)."""
        if style and self.use_colors:
            # Simple color mapping for terminals that support it
            color_map = {
                'bold red': '\033[1;31m',
                'bold yellow': '\033[1;33m', 
                'bold green': '\033[1;32m',
                'bold blue': '\033[1;34m',
                'bold cyan': '\033[1;36m',
                'bold magenta': '\033[1;35m',
                'dim': '\033[2m',
                'reset': '\033[0m'
            }
            
            # Extract color from style
            for style_name, color_code in color_map.items():
                if style_name in style.lower():
                    print(f"{color_code}{text}{color_map['reset']}")
                    return
        
        print(text)
    
    def confirm(self, question: str, default: bool = True) -> bool:
        """Simple confirmation prompt."""
        default_str = "Y/n" if default else "y/N"
        response = input(f"{question} ({default_str}): ").strip().lower()
        
        if not response:
            return default
        
        return response.startswith('y')


class SimpleCLI:
    """Simple CLI interface that works without external dependencies."""
    
    def __init__(self, context: CLIContext):
        self.context = context
        self.console = FallbackConsole()
    
    def show_welcome(self):
        """Show welcome message with simple formatting."""
        print("\n" + "=" * 60)
        print("🌟 TOARIPI SLM - Educational Content Generation System")
        print("=" * 60)
        print("\n📚 Purpose:")
        print("   Generate educational content for Toaripi language learners")
        print("\n🎯 Target:")
        print("   Teachers, students, and community members")
        print("\n🎓 Focus:")
        print("   Primary school age-appropriate content with cultural sensitivity")
        print("\n✨ Key Features:")
        print("   • Offline-compatible for classroom use")
        print("   • Age-appropriate content validation")
        print("   • Cultural sensitivity built-in")
        print("   • Beginner-friendly guided workflows")
        print("=" * 60)
    
    def show_status(self, components: Dict[str, Any]):
        """Show system status in simple format."""
        print("\n📊 System Status:")
        print("-" * 20)
        
        for name, status in components.items():
            icon = "✅" if status else "❌"
            print(f"  {icon} {name}")
    
    def show_next_steps(self, steps: List[Dict[str, str]]):
        """Show next steps recommendations."""
        print("\n💡 Recommended Next Steps:")
        print("-" * 28)
        
        for i, step in enumerate(steps, 1):
            print(f"\n  {i}. {step.get('title', 'Unknown step')}")
            if 'description' in step:
                print(f"     {step['description']}")
            if 'command' in step:
                print(f"     Command: toaripi {step['command']}")
    
    def handle_natural_language_input(self, query: str) -> Optional[str]:
        """Simple natural language processing."""
        query_lower = query.lower()
        
        # Simple keyword matching
        if any(word in query_lower for word in ['train', 'training', 'model']):
            return "train start"
        elif any(word in query_lower for word in ['data', 'prepare', 'validate']):
            return "data validate"
        elif any(word in query_lower for word in ['status', 'check', 'health']):
            return "status"
        elif any(word in query_lower for word in ['help', 'guide', 'how']):
            return "help"
        elif any(word in query_lower for word in ['serve', 'deploy', 'run']):
            return "serve start"
        
        return None
    
    def confirm_action(self, message: str, default: bool = True) -> bool:
        """Ask for user confirmation."""
        return self.console.confirm(message, default)


def create_simple_cli_context(
    config_file: Optional[Path] = None,
    verbose: bool = False,
    quiet: bool = False,
    working_directory: Optional[Path] = None
) -> CLIContext:
    """Create a simple CLI context without external dependencies."""
    
    if working_directory is None:
        working_directory = Path.cwd()
    
    context = CLIContext(
        config_file=config_file,
        verbose=verbose,
        quiet=quiet,
        working_directory=working_directory
    )
    
    return context


# Simple user profile system
@dataclass
class SimpleUserProfile:
    """Simple user profile without external dependencies."""
    
    display_name: str = "User"
    user_type: str = "teacher"  # teacher, student, developer, contributor
    experience_level: str = "beginner"  # beginner, intermediate, advanced, expert
    target_age_group: str = "primary"  # primary, secondary, adult
    preferred_workflow: str = "guided"  # guided, advanced, custom
    show_tips: bool = True


class SimpleUserProfileManager:
    """Simple profile manager that works without external dependencies."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".toaripi"
        self.profile_file = self.config_dir / "profile.json"
    
    def get_current_profile(self) -> SimpleUserProfile:
        """Get current user profile or create default."""
        if self.profile_file.exists():
            try:
                with open(self.profile_file, 'r') as f:
                    data = json.load(f)
                return SimpleUserProfile(**data)
            except Exception:
                pass
        
        return SimpleUserProfile()
    
    def save_profile(self, profile: SimpleUserProfile):
        """Save user profile to disk."""
        self.config_dir.mkdir(exist_ok=True)
        
        with open(self.profile_file, 'w') as f:
            json.dump(profile.__dict__, f, indent=2)


# Simple welcome system
class SimpleWelcome:
    """Simple welcome system without external dependencies."""
    
    def __init__(self, context: CLIContext):
        self.context = context
        self.cli = SimpleCLI(context)
    
    def show_welcome(self):
        """Show intelligent welcome based on context."""
        
        # Check if first time user
        profile_manager = SimpleUserProfileManager()
        profile = profile_manager.get_current_profile()
        
        is_first_time = not profile_manager.profile_file.exists()
        
        if is_first_time:
            self._show_first_time_welcome()
        else:
            self._show_returning_user_welcome(profile)
    
    def _show_first_time_welcome(self):
        """Welcome for first-time users."""
        print("\n🎉 Welcome to Toaripi SLM!")
        print("This is your first time using the system.")
        print("\nToaripi SLM helps create educational content in Toaripi language")
        print("for primary school students and teachers.")
        
        print("\n🚀 Quick Setup:")
        print("1. Check your training data: toaripi data validate")
        print("2. Configure training: toaripi config")
        print("3. Start training: toaripi train start")
        
        print("\n💡 Need help? Try: toaripi help")
    
    def _show_returning_user_welcome(self, profile: SimpleUserProfile):
        """Welcome for returning users."""
        print(f"\n👋 Welcome back, {profile.display_name}!")
        
        # Show project status
        data_dir = self.context.working_directory / "data" / "processed"
        models_dir = self.context.working_directory / "models"
        
        if data_dir.exists() and (data_dir / "train.csv").exists():
            print("✅ Training data is ready")
        else:
            print("⚠️  Training data needs preparation")
        
        if models_dir.exists() and any(models_dir.iterdir()):
            print("✅ Models are available")
        else:
            print("🎓 Ready to train your first model")
        
        print(f"\n🎯 Profile: {profile.experience_level.title()} {profile.user_type}")
        print(f"📚 Focus: {profile.target_age_group.title()} education")


# Demo function that works without dependencies
def demo_simple_cli():
    """Demonstrate the simple CLI framework."""
    
    print("🌟 Toaripi SLM - Simple CLI Demo")
    print("(Fallback version without external dependencies)")
    print("=" * 60)
    
    # Create context
    context = create_simple_cli_context(verbose=True)
    
    # Show welcome
    welcome = SimpleWelcome(context)
    welcome.show_welcome()
    
    # Show user profile
    print("\n👤 User Profile System:")
    profile_manager = SimpleUserProfileManager()
    profile = profile_manager.get_current_profile()
    
    print(f"Display name: {profile.display_name}")
    print(f"User type: {profile.user_type}")
    print(f"Experience level: {profile.experience_level}")
    print(f"Target age group: {profile.target_age_group}")
    
    # Show project analysis
    print("\n🔍 Project Analysis:")
    checks = [
        ("Training data", context.working_directory / "data" / "processed"),
        ("Configuration", context.working_directory / "configs"),
        ("Models", context.working_directory / "models"),
        ("Scripts", context.working_directory / "scripts"),
    ]
    
    for name, path in checks:
        status = "✅ Found" if path.exists() else "❌ Missing"
        print(f"  {status}: {name}")
    
    # Show guidance
    print("\n💡 Next Steps:")
    cli = SimpleCLI(context)
    
    steps = [
        {
            "title": "Validate Training Data",
            "description": "Check that your parallel text data is properly formatted",
            "command": "data validate"
        },
        {
            "title": "Configure Training",
            "description": "Set up training parameters for your educational model",
            "command": "config training"
        },
        {
            "title": "Start Training",
            "description": "Begin training your Toaripi educational content model",
            "command": "train start"
        }
    ]
    
    cli.show_next_steps(steps)
    
    print(f"\n✨ Simple CLI framework ready!")
    print(f"📁 Working in: {context.working_directory}")


if __name__ == "__main__":
    demo_simple_cli()