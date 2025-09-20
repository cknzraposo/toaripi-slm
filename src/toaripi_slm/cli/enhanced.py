"""
Enhanced CLI with smart help and command discovery features.
"""

import click
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from .smart_help import (
    show_command_suggestions, 
    show_progressive_help, 
    show_contextual_examples,
    smart_help
)
from .context import get_context

console = Console()

class EnhancedGroup(click.Group):
    """Enhanced click Group with smart help features."""
    
    def resolve_command(self, ctx, args):
        """Resolve command with smart suggestions for unknown commands."""
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as e:
            if args:
                command_name = args[0]
                
                # Get user context
                try:
                    context = get_context()
                    user_level = context.profile
                except:
                    user_level = "beginner"
                
                # Show smart suggestions
                console.print(f"❌ [red]Unknown command: {command_name}[/red]\n")
                show_command_suggestions(command_name, user_level)
                
                ctx.exit(1)
            else:
                raise e
    
    def format_usage(self, ctx, formatter):
        """Enhanced usage formatting with smart help."""
        super().format_usage(ctx, formatter)
        
        # Add contextual help for main command
        if ctx.info_name == "cli":
            try:
                context = get_context()
                user_level = context.profile
                
                formatter.write(f"\n💡 Smart Help (Profile: {user_level.title()}):\n")
                formatter.write("   toaripi help                    # Get contextual help\n")
                formatter.write("   toaripi suggest <query>         # Find commands\n")
                formatter.write("   toaripi examples                # See examples\n")
                
                # Quick status check
                help_data = smart_help.get_contextual_help()
                status = help_data["system_status"]
                
                if not status["has_models"] and not status["has_data"]:
                    formatter.write("\n🆕 Quick Start:\n")
                    formatter.write("   toaripi workflow quickstart    # First-time setup\n")
                    formatter.write("   toaripi system status          # Check environment\n")
                elif status["has_data"] and not status["has_models"]:
                    formatter.write("\n📊 Ready to Train:\n")
                    formatter.write("   toaripi model train             # Train a model\n")
                    formatter.write("   toaripi data validate           # Check data\n")
                elif status["has_models"]:
                    formatter.write("\n🤖 Model Ready:\n")
                    formatter.write("   toaripi chat                    # Try your model\n")
                    formatter.write("   toaripi model test              # Evaluate model\n")
                
            except Exception:
                pass

class EnhancedCommand(click.Command):
    """Enhanced click Command with smart help features."""
    
    def get_help(self, ctx):
        """Enhanced help with contextual information."""
        help_text = super().get_help(ctx)
        
        # Add contextual help for this specific command
        command_name = f"{ctx.parent.info_name}.{ctx.info_name}" if ctx.parent else ctx.info_name
        
        try:
            help_data = smart_help.get_contextual_help(command_name)
            
            if help_data.get("related_commands"):
                help_text += "\n\n🔗 Related Commands:\n"
                for related in help_data["related_commands"]:
                    cli_cmd = related.replace(".", " ")
                    help_text += f"   toaripi {cli_cmd}\n"
            
            if help_data.get("prerequisites"):
                help_text += "\n📋 Prerequisites:\n"
                for prereq in help_data["prerequisites"]:
                    help_text += f"   • {prereq.replace('_', ' ').title()}\n"
            
        except Exception:
            pass
        
        return help_text

def smart_help_command():
    """Create smart help commands."""
    
    @click.group(name="help")
    def help_group():
        """Smart help and command discovery."""
        pass
    
    @help_group.command()
    @click.argument("query", required=False)
    @click.option("--limit", "-l", default=5, help="Number of suggestions to show")
    def suggest(query, limit):
        """Suggest commands based on query."""
        if not query:
            console.print("❓ [yellow]Please provide a query to search for commands.[/yellow]")
            console.print("\nExample: [cyan]toaripi help suggest train[/cyan]")
            return
        
        try:
            context = get_context()
            user_level = context.profile
        except:
            user_level = "beginner"
        
        show_command_suggestions(query, user_level)
    
    @help_group.command()
    def guide():
        """Show progressive help based on your experience level."""
        try:
            context = get_context()
            user_level = context.profile
        except:
            user_level = "beginner"
        
        show_progressive_help(user_level)
    
    @help_group.command()
    def examples():
        """Show contextual examples based on your current setup."""
        show_contextual_examples()
    
    @help_group.command() 
    @click.option("--category", type=click.Choice(["workflow", "model", "data", "chat", "system"]),
                  help="Show commands for specific category")
    def commands(category):
        """List all available commands with descriptions."""
        console.print("📚 [bold blue]Available Commands[/bold blue]\n")
        
        commands_by_category = {}
        
        for cmd_name, cmd_info in smart_help.command_registry.items():
            cat = cmd_info["category"]
            if category and cat != category:
                continue
                
            if cat not in commands_by_category:
                commands_by_category[cat] = []
            
            cli_cmd = cmd_name.replace(".", " ")
            commands_by_category[cat].append((f"toaripi {cli_cmd}", cmd_info["description"]))
        
        for cat, cmds in commands_by_category.items():
            console.print(f"\n🔧 [bold green]{cat.title()} Commands:[/bold green]")
            
            from rich.table import Table
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Command", style="cyan", no_wrap=True)
            table.add_column("Description", style="green")
            
            for cmd, desc in sorted(cmds):
                table.add_row(cmd, desc)
            
            console.print(table)
    
    @help_group.command()
    def status():
        """Show smart help status and recommendations."""
        console.print("🧠 [bold blue]Smart Help Status[/bold blue]\n")
        
        try:
            context = get_context()
            user_level = context.profile
        except:
            user_level = "beginner"
        
        # Show user profile
        profile_panel = Panel(
            f"""
            Current Profile: [green]{user_level.title()}[/green]
            
            Change with: [cyan]toaripi system config --profile LEVEL[/cyan]
            Available levels: beginner, intermediate, advanced, expert
            """,
            title="👤 User Profile",
            border_style="blue"
        )
        console.print(profile_panel)
        
        # Show contextual help
        help_data = smart_help.get_contextual_help()
        
        if help_data["recommendations"]:
            console.print("\n💡 [bold blue]Current Recommendations:[/bold blue]")
            for rec in help_data["recommendations"]:
                priority_icon = "🚨" if rec["priority"] == "high" else "ℹ️"
                console.print(f"   {priority_icon} {rec['message']}")
                if rec.get("commands"):
                    for cmd in rec["commands"]:
                        cli_cmd = cmd.replace(".", " ")
                        console.print(f"      → [cyan]toaripi {cli_cmd}[/cyan]")
        
        if help_data["next_steps"]:
            console.print("\n🚀 [bold blue]Suggested Next Steps:[/bold blue]")
            for i, step in enumerate(help_data["next_steps"][:5], 1):
                console.print(f"   {i}. {step}")
    
    return help_group

def create_smart_cli():
    """Create CLI with smart help integration."""
    
    @click.group(cls=EnhancedGroup)
    @click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
    @click.option("--debug", "-d", is_flag=True, help="Enable debug output")
    @click.option("--config-dir", type=click.Path(exists=True), help="Configuration directory")
    @click.option("--profile", type=click.Choice(["beginner", "intermediate", "advanced", "expert"]),
                  help="Set user profile (overrides saved preference)")
    @click.pass_context
    def cli(ctx, verbose, debug, config_dir, profile):
        """
        Toaripi SLM CLI - Train, test, and interact with language models 
        for Toaripi educational content generation.
        
        This tool guides you through the complete workflow of creating 
        and using a small language model for generating educational 
        content in the Toaripi language.
        
        \b
        Command Groups:
          workflow    Guided workflows for common tasks
          model       Train, test, and manage models  
          data        Validate and process training data
          chat        Interactive chat sessions
          system      System diagnostics and configuration
          help        Smart help and command discovery
        
        Smart Help Features:
          • Command suggestions based on typos or partial matches
          • Contextual help based on your current setup
          • Progressive guidance based on experience level
          • Related command recommendations
        
        Examples:
          toaripi workflow quickstart     # Get started quickly
          toaripi help suggest train      # Find training commands
          toaripi help examples           # See contextual examples
          toaripi help guide              # Get level-appropriate help
        """
        # Standard CLI setup
        ctx.ensure_object(dict)
        
        # Get unified context
        from .context import get_context
        context = get_context()
        
        # Update context with CLI options
        if verbose:
            context.verbose = verbose
        if debug:
            context.debug = debug
        if config_dir:
            context.config_dir = Path(config_dir)
        if profile:
            context.profile = profile
            context.save_preferences()
        
        # Store context for subcommands
        ctx.obj = context
        
        # Show banner and smart help for main command only
        if ctx.invoked_subcommand is None:
            from . import show_banner
            show_banner()
            
            # Show smart contextual help
            console.print("\n🧠 [bold blue]Smart Help:[/bold blue]")
            help_data = smart_help.get_contextual_help()
            
            if help_data["recommendations"]:
                console.print("💡 Recommendations:")
                for rec in help_data["recommendations"][:2]:
                    priority_icon = "🚨" if rec["priority"] == "high" else "ℹ️"
                    console.print(f"   {priority_icon} {rec['message']}")
            
            if help_data["next_steps"]:
                console.print("🚀 Quick Start:")
                for step in help_data["next_steps"][:2]:
                    console.print(f"   • {step}")
            
            console.print("\n💬 Try: [cyan]toaripi help suggest <what you want to do>[/cyan]")
            console.print("📖 Or: [cyan]toaripi help guide[/cyan] for your experience level\n")
    
    # Add smart help commands
    cli.add_command(smart_help_command())
    
    return cli

# Command completion for shells
def get_command_completions():
    """Get command completions for shell integration."""
    completions = []
    
    for cmd_name, cmd_info in smart_help.command_registry.items():
        cli_cmd = cmd_name.replace(".", " ")
        completions.append(f"toaripi {cli_cmd}")
        
        # Add keywords as potential completions
        for keyword in cmd_info["keywords"]:
            completions.append(keyword)
    
    return sorted(set(completions))

def generate_shell_completion():
    """Generate shell completion script."""
    completions = get_command_completions()
    
    bash_script = '''
# Toaripi SLM CLI completion
_toaripi_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    opts="''' + ' '.join(completions) + '''"
    
    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    return 0
}

complete -F _toaripi_completion toaripi
'''
    
    return bash_script