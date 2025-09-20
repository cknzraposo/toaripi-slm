"""
Smart help system with command suggestions and context-aware assistance.
"""

import difflib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.tree import Tree

from .context import get_context

console = Console()

class SmartHelp:
    """Smart help system with command suggestions and context awareness."""
    
    def __init__(self):
        self.command_registry = self._build_command_registry()
        self.context_keywords = self._build_context_keywords()
    
    def _build_command_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build a registry of all available commands with metadata."""
        return {
            # Workflow commands
            "workflow.quickstart": {
                "category": "workflow",
                "description": "Get started quickly with guided setup",
                "keywords": ["start", "begin", "setup", "quick", "first", "new"],
                "user_levels": ["beginner", "intermediate"],
                "prerequisites": [],
                "related": ["system.status", "data.validate"]
            },
            "workflow.full": {
                "category": "workflow", 
                "description": "Complete production pipeline from data to model",
                "keywords": ["complete", "full", "production", "pipeline", "end-to-end"],
                "user_levels": ["intermediate", "advanced", "expert"],
                "prerequisites": ["data"],
                "related": ["model.train", "model.test", "model.export"]
            },
            "workflow.train-and-test": {
                "category": "workflow",
                "description": "Quick train and test cycle",
                "keywords": ["train", "test", "cycle", "quick", "evaluate"],
                "user_levels": ["beginner", "intermediate", "advanced"],
                "prerequisites": ["data"],
                "related": ["model.train", "model.test"]
            },
            
            # Model commands
            "model.train": {
                "category": "model",
                "description": "Train a new Toaripi language model",
                "keywords": ["train", "learning", "create", "build", "fine-tune"],
                "user_levels": ["beginner", "intermediate", "advanced", "expert"],
                "prerequisites": ["data"],
                "related": ["model.test", "data.validate", "model.export"]
            },
            "model.test": {
                "category": "model",
                "description": "Test and evaluate model performance",
                "keywords": ["test", "evaluate", "performance", "accuracy", "metrics"],
                "user_levels": ["beginner", "intermediate", "advanced", "expert"],
                "prerequisites": ["model"],
                "related": ["model.train", "model.compare"]
            },
            "model.list": {
                "category": "model",
                "description": "List all trained models",
                "keywords": ["list", "show", "models", "available", "trained"],
                "user_levels": ["beginner", "intermediate", "advanced", "expert"],
                "prerequisites": [],
                "related": ["model.compare", "model.export"]
            },
            "model.compare": {
                "category": "model",
                "description": "Compare multiple model performances",
                "keywords": ["compare", "benchmark", "performance", "metrics"],
                "user_levels": ["intermediate", "advanced", "expert"],
                "prerequisites": ["multiple_models"],
                "related": ["model.test", "model.list"]
            },
            "model.export": {
                "category": "model",
                "description": "Export models for deployment",
                "keywords": ["export", "deploy", "convert", "gguf", "production"],
                "user_levels": ["intermediate", "advanced", "expert"],
                "prerequisites": ["model"],
                "related": ["model.train", "system.status"]
            },
            
            # Data commands
            "data.validate": {
                "category": "data",
                "description": "Validate training data quality",
                "keywords": ["validate", "check", "data", "quality", "verify"],
                "user_levels": ["beginner", "intermediate", "advanced", "expert"],
                "prerequisites": [],
                "related": ["data.split", "data.preview", "workflow.quickstart"]
            },
            "data.split": {
                "category": "data",
                "description": "Split data into train/validation/test sets",
                "keywords": ["split", "divide", "train", "validation", "test"],
                "user_levels": ["intermediate", "advanced", "expert"],
                "prerequisites": ["data"],
                "related": ["data.validate", "model.train"]
            },
            "data.merge": {
                "category": "data",
                "description": "Merge multiple data files",
                "keywords": ["merge", "combine", "join", "files", "datasets"],
                "user_levels": ["intermediate", "advanced", "expert"],
                "prerequisites": ["multiple_files"],
                "related": ["data.validate", "data.split"]
            },
            "data.preview": {
                "category": "data",
                "description": "Preview data samples",
                "keywords": ["preview", "show", "sample", "examine", "view"],
                "user_levels": ["beginner", "intermediate", "advanced", "expert"],
                "prerequisites": ["data"],
                "related": ["data.validate"]
            },
            
            # Chat commands
            "chat": {
                "category": "chat",
                "description": "Start interactive chat session",
                "keywords": ["chat", "talk", "interact", "conversation", "test"],
                "user_levels": ["beginner", "intermediate", "advanced", "expert"],
                "prerequisites": ["model"],
                "related": ["model.test", "chat.sessions"]
            },
            "chat.sessions": {
                "category": "chat",
                "description": "Manage chat session history",
                "keywords": ["sessions", "history", "conversations", "saved"],
                "user_levels": ["intermediate", "advanced", "expert"],
                "prerequisites": ["chat_history"],
                "related": ["chat"]
            },
            
            # System commands
            "system.status": {
                "category": "system",
                "description": "Check system health and setup",
                "keywords": ["status", "health", "check", "system", "setup"],
                "user_levels": ["beginner", "intermediate", "advanced", "expert"],
                "prerequisites": [],
                "related": ["system.doctor", "system.config"]
            },
            "system.doctor": {
                "category": "system",
                "description": "Comprehensive system diagnostics",
                "keywords": ["doctor", "diagnose", "fix", "troubleshoot", "repair"],
                "user_levels": ["beginner", "intermediate", "advanced", "expert"],
                "prerequisites": [],
                "related": ["system.status", "workflow.quickstart"]
            },
            "system.config": {
                "category": "system",
                "description": "Manage system configuration",
                "keywords": ["config", "configure", "settings", "preferences", "setup"],
                "user_levels": ["intermediate", "advanced", "expert"],
                "prerequisites": [],
                "related": ["system.status"]
            }
        }
    
    def _build_context_keywords(self) -> Dict[str, List[str]]:
        """Build context-specific keyword mappings."""
        return {
            "first_time": ["start", "begin", "new", "first", "setup", "install"],
            "training": ["train", "learning", "model", "fine-tune", "create"],
            "testing": ["test", "evaluate", "performance", "accuracy", "metrics"],
            "data_issues": ["data", "validate", "check", "error", "problem"],
            "deployment": ["export", "deploy", "production", "convert", "gguf"],
            "troubleshooting": ["error", "problem", "fix", "help", "issue", "broken"],
            "advanced": ["compare", "benchmark", "optimization", "professional"],
            "interaction": ["chat", "talk", "conversation", "interactive"]
        }
    
    def suggest_commands(self, query: str, user_level: str = "beginner", 
                        limit: int = 5) -> List[Tuple[str, float, str]]:
        """Suggest commands based on query and user context."""
        query_lower = query.lower()
        suggestions = []
        
        for cmd_name, cmd_info in self.command_registry.items():
            score = 0.0
            
            # Direct keyword match
            for keyword in cmd_info["keywords"]:
                if keyword in query_lower:
                    score += 2.0
            
            # Description match
            desc_words = cmd_info["description"].lower().split()
            query_words = query_lower.split()
            for q_word in query_words:
                for d_word in desc_words:
                    if q_word in d_word or d_word in q_word:
                        score += 1.0
            
            # Fuzzy matching for typos
            for keyword in cmd_info["keywords"]:
                ratio = difflib.SequenceMatcher(None, query_lower, keyword).ratio()
                if ratio > 0.7:
                    score += ratio
            
            # User level appropriateness
            if user_level in cmd_info["user_levels"]:
                score += 0.5
            
            # Context bonus
            context_bonus = self._get_context_bonus(query_lower, cmd_name)
            score += context_bonus
            
            if score > 0:
                suggestions.append((cmd_name, score, cmd_info["description"]))
        
        # Sort by score and return top results
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:limit]
    
    def _get_context_bonus(self, query: str, command: str) -> float:
        """Get context bonus based on system state and query."""
        bonus = 0.0
        ctx = get_context()
        
        # Check if user is a first-time user
        if not self._has_trained_models() and any(kw in query for kw in self.context_keywords["first_time"]):
            if command == "workflow.quickstart":
                bonus += 1.0
            elif command == "system.status":
                bonus += 0.5
        
        # Check if user has data issues
        if any(kw in query for kw in self.context_keywords["data_issues"]):
            if command.startswith("data."):
                bonus += 1.0
            elif command == "system.doctor":
                bonus += 0.8
        
        # Check if user wants to deploy
        if any(kw in query for kw in self.context_keywords["deployment"]):
            if command == "model.export":
                bonus += 1.0
        
        return bonus
    
    def _has_trained_models(self) -> bool:
        """Check if user has any trained models."""
        ctx = get_context()
        models_dir = ctx.models_dir / "hf"
        return models_dir.exists() and any(models_dir.iterdir())
    
    def get_contextual_help(self, command: Optional[str] = None) -> Dict[str, Any]:
        """Get contextual help based on current system state."""
        ctx = get_context()
        help_data = {
            "system_status": self._get_system_status(),
            "recommendations": [],
            "next_steps": [],
            "related_commands": []
        }
        
        # System-based recommendations
        if not self._has_trained_models():
            help_data["recommendations"].append({
                "priority": "high",
                "message": "No trained models found. Start with quickstart workflow.",
                "commands": ["workflow.quickstart", "system.status"]
            })
        
        if not self._has_sample_data():
            help_data["recommendations"].append({
                "priority": "high", 
                "message": "No training data found. Validate or create sample data.",
                "commands": ["data.validate", "workflow.quickstart"]
            })
        
        # Profile-based recommendations
        if ctx.profile == "beginner":
            help_data["next_steps"] = [
                "Run 'toaripi workflow quickstart' for guided setup",
                "Check system health with 'toaripi system status'",
                "Validate your data with 'toaripi data validate'"
            ]
        elif ctx.profile == "intermediate":
            help_data["next_steps"] = [
                "Use 'toaripi model train --interactive' for guided training",
                "Try 'toaripi workflow full' for complete pipeline",
                "Compare models with 'toaripi model compare'"
            ]
        elif ctx.profile in ["advanced", "expert"]:
            help_data["next_steps"] = [
                "Use 'toaripi model train --config' for custom training",
                "Optimize with 'toaripi model export --quantization'",
                "Monitor with 'toaripi system doctor --detailed'"
            ]
        
        # Command-specific help
        if command and command in self.command_registry:
            cmd_info = self.command_registry[command]
            help_data["related_commands"] = cmd_info["related"]
            help_data["prerequisites"] = cmd_info["prerequisites"]
        
        return help_data
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "has_models": self._has_trained_models(),
            "has_data": self._has_sample_data(),
            "config_exists": self._has_config_files()
        }
    
    def _has_sample_data(self) -> bool:
        """Check if sample data exists."""
        ctx = get_context()
        sample_files = [
            ctx.data_dir / "samples" / "sample_parallel.csv",
            ctx.data_dir / "raw",
            ctx.data_dir / "processed"
        ]
        return any(f.exists() for f in sample_files)
    
    def _has_config_files(self) -> bool:
        """Check if configuration files exist."""
        ctx = get_context()
        config_files = [
            ctx.config_dir / "training" / "base_config.yaml",
            ctx.config_dir / "data" / "preprocessing_config.yaml"
        ]
        return any(f.exists() for f in config_files)

# Global smart help instance
smart_help = SmartHelp()

def show_command_suggestions(query: str, user_level: str = "beginner"):
    """Show command suggestions for a query."""
    suggestions = smart_help.suggest_commands(query, user_level)
    
    if not suggestions:
        console.print(f"❌ No commands found for: [cyan]{query}[/cyan]")
        console.print("\n💡 Try these general commands:")
        console.print("  • [cyan]toaripi workflow quickstart[/cyan] - Get started")
        console.print("  • [cyan]toaripi system status[/cyan] - Check system")
        console.print("  • [cyan]toaripi --help[/cyan] - Full command list")
        return
    
    console.print(f"🔍 [bold blue]Command suggestions for:[/bold blue] [cyan]{query}[/cyan]\n")
    
    # Create suggestions table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")
    table.add_column("Relevance", style="yellow")
    
    for cmd_name, score, description in suggestions:
        # Convert command name to CLI format
        cli_cmd = cmd_name.replace(".", " ")
        relevance = "🔥" if score > 2.0 else "⭐" if score > 1.0 else "💡"
        table.add_row(f"toaripi {cli_cmd}", description, relevance)
    
    console.print(table)
    
    # Show contextual help
    help_data = smart_help.get_contextual_help()
    
    if help_data["recommendations"]:
        console.print("\n💡 [bold blue]Recommendations:[/bold blue]")
        for rec in help_data["recommendations"]:
            priority_icon = "🚨" if rec["priority"] == "high" else "ℹ️"
            console.print(f"   {priority_icon} {rec['message']}")
    
    if help_data["next_steps"]:
        console.print("\n🚀 [bold blue]Next Steps:[/bold blue]")
        for step in help_data["next_steps"][:3]:
            console.print(f"   • {step}")

def show_progressive_help(user_level: str = "beginner"):
    """Show progressive help based on user level."""
    console.print(f"📚 [bold blue]Help for {user_level.title()} Users[/bold blue]\n")
    
    help_sections = {
        "beginner": {
            "Essential Commands": [
                ("toaripi workflow quickstart", "Get started with guided setup"),
                ("toaripi system status", "Check if everything is working"),
                ("toaripi data validate", "Check your training data"),
                ("toaripi model train --interactive", "Train your first model")
            ],
            "Learning Path": [
                "1. Run system status to check setup",
                "2. Use quickstart workflow for first-time setup", 
                "3. Validate your training data",
                "4. Train your first model interactively",
                "5. Test the model with chat"
            ]
        },
        "intermediate": {
            "Core Commands": [
                ("toaripi workflow full", "Complete training pipeline"),
                ("toaripi model compare", "Compare model performance"),
                ("toaripi data split", "Prepare training datasets"),
                ("toaripi model export", "Export models for deployment")
            ],
            "Advanced Workflows": [
                "1. Use full workflow for production training",
                "2. Compare multiple model configurations",
                "3. Optimize data preprocessing",
                "4. Export models for deployment"
            ]
        },
        "advanced": {
            "Professional Tools": [
                ("toaripi model train --config", "Custom training configurations"),
                ("toaripi system doctor --detailed", "Comprehensive diagnostics"),
                ("toaripi model export --quantization", "Optimized model exports"),
                ("toaripi data merge --strategy", "Advanced data management")
            ],
            "Optimization Tips": [
                "1. Use custom configurations for training",
                "2. Monitor performance with detailed diagnostics",
                "3. Optimize models with quantization",
                "4. Manage large datasets efficiently"
            ]
        },
        "expert": {
            "Expert Features": [
                ("toaripi model train --profile expert", "Full manual control"),
                ("toaripi system config --advanced", "System-level configuration"),
                ("toaripi model compare --metrics custom", "Custom evaluation metrics"),
                ("toaripi data validate --deep", "Comprehensive data analysis")
            ],
            "Professional Workflows": [
                "1. Design custom training pipelines",
                "2. Implement advanced evaluation metrics",
                "3. Optimize for production deployment",
                "4. Contribute improvements back to project"
            ]
        }
    }
    
    sections = help_sections.get(user_level, help_sections["beginner"])
    
    for section_title, content in sections.items():
        console.print(f"\n📋 [bold green]{section_title}:[/bold green]")
        
        if section_title.endswith("Commands"):
            # Show as table
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Command", style="cyan")
            table.add_column("Description", style="green")
            
            for cmd, desc in content:
                table.add_row(cmd, desc)
            
            console.print(table)
        else:
            # Show as list
            for item in content:
                console.print(f"   {item}")

def show_contextual_examples():
    """Show contextual examples based on system state."""
    ctx = get_context()
    console.print("💡 [bold blue]Contextual Examples[/bold blue]\n")
    
    help_data = smart_help.get_contextual_help()
    status = help_data["system_status"]
    
    # First-time user examples
    if not status["has_models"] and not status["has_data"]:
        console.print("🆕 [bold green]First-time Setup:[/bold green]")
        examples = [
            "toaripi workflow quickstart              # Complete guided setup",
            "toaripi system status                   # Check your environment",
            "toaripi system doctor                   # Fix any issues"
        ]
    
    # Has data but no models
    elif status["has_data"] and not status["has_models"]:
        console.print("📊 [bold green]Ready to Train:[/bold green]")
        examples = [
            "toaripi data validate                   # Check data quality",
            "toaripi model train --interactive       # Train with guidance",
            "toaripi workflow train-and-test         # Quick training cycle"
        ]
    
    # Has models
    elif status["has_models"]:
        console.print("🤖 [bold green]Model Management:[/bold green]")
        examples = [
            "toaripi model list                      # See all your models",
            "toaripi model test --model latest       # Test model performance",
            "toaripi chat                           # Try interactive chat",
            "toaripi model compare                   # Compare model versions",
            "toaripi model export --format gguf      # Export for deployment"
        ]
    
    # General examples
    else:
        console.print("🔧 [bold green]General Usage:[/bold green]")
        examples = [
            "toaripi system status                   # Check system health",
            "toaripi workflow quickstart             # Get started quickly",
            "toaripi --help                         # See all commands"
        ]
    
    for example in examples:
        console.print(f"   [cyan]{example}[/cyan]")
    
    # Show related commands
    console.print(f"\n🔗 [bold blue]Related Commands:[/bold blue]")
    console.print("   [cyan]toaripi COMMAND --help[/cyan]           # Get help for any command")
    console.print("   [cyan]toaripi system doctor --detailed[/cyan] # Comprehensive diagnostics")
    console.print("   [cyan]toaripi workflow --help[/cyan]          # See all workflows")

class DidYouMeanMixin:
    """Mixin to add 'did you mean' functionality to CLI commands."""
    
    def format_usage(self, ctx, formatter):
        """Override to add command suggestions for unknown commands."""
        # Get the original usage
        super().format_usage(ctx, formatter)
        
        # If this is an unknown command error, suggest alternatives
        if hasattr(ctx, 'info_name') and ctx.info_name:
            command_name = ctx.info_name
            
            # Get user context
            try:
                context = get_context()
                user_level = context.profile
            except:
                user_level = "beginner"
            
            # Get suggestions
            suggestions = smart_help.suggest_commands(command_name, user_level, limit=3)
            
            if suggestions:
                formatter.write("\n")
                formatter.write("💡 Did you mean:\n")
                for cmd_name, score, description in suggestions:
                    cli_cmd = cmd_name.replace(".", " ")
                    formatter.write(f"   toaripi {cli_cmd:<25} # {description}\n")
                
                formatter.write("\n")
                formatter.write("💬 Or try: toaripi --help\n")

def enhanced_error_handler(ctx, param, value):
    """Enhanced error handler with suggestions."""
    console.print(f"❌ [red]Unknown command: {value}[/red]\n")
    
    # Get user context  
    try:
        context = get_context()
        user_level = context.profile
    except:
        user_level = "beginner"
    
    # Show suggestions
    show_command_suggestions(value, user_level)
    ctx.exit(1)