"""User guidance system for Toaripi SLM launcher."""

from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.align import Align

try:
    # Try relative imports first (when used as module)
    from .validator import ValidationResult, ValidationIssue
except ImportError:
    # Fall back to absolute imports (when used as script)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from validator import ValidationResult, ValidationIssue

class UserGuidance:
    """Provides user-friendly guidance for system issues and setup."""
    
    def __init__(self, console: Console):
        self.console = console
        
    def show_system_fixes(self, validation_result: ValidationResult):
        """Show guidance for fixing system issues."""
        if not validation_result.issues:
            self._show_success_message()
            return
            
        self.console.print("\n[bold red]🔧 System Setup Required[/bold red]")
        self.console.print("The following issues need to be resolved for educational content training:\n")
        
        # Group issues by severity
        errors = [i for i in validation_result.issues if i.severity == 'error']
        warnings = [i for i in validation_result.issues if i.severity == 'warning']
        
        # Show critical errors first
        if errors:
            self._show_error_guidance(errors)
            
        # Show warnings
        if warnings:
            self._show_warning_guidance(warnings)
            
        # Show next steps
        self._show_next_steps(validation_result)
        
    def _show_error_guidance(self, errors: List[ValidationIssue]):
        """Show guidance for critical errors."""
        error_table = Table(title="❌ Critical Issues (Must Fix)", show_header=True, border_style="red")
        error_table.add_column("Component", style="red bold", min_width=15)
        error_table.add_column("Issue", style="red", min_width=30)
        error_table.add_column("How to Fix", style="yellow", min_width=40)
        
        for error in errors:
            # Add auto-fix indicator
            fix_text = error.fix_suggestion
            if error.auto_fixable:
                fix_text = f"🤖 {fix_text}\n[dim](Can be auto-fixed)[/dim]"
                
            error_table.add_row(
                error.component,
                error.issue,
                fix_text
            )
            
        self.console.print(error_table)
        self.console.print()
        
    def _show_warning_guidance(self, warnings: List[ValidationIssue]):
        """Show guidance for warnings."""
        warning_table = Table(title="⚠️  Recommendations (Optional)", show_header=True, border_style="yellow")
        warning_table.add_column("Component", style="yellow bold", min_width=15)
        warning_table.add_column("Issue", style="yellow", min_width=30)
        warning_table.add_column("Recommendation", style="cyan", min_width=40)
        
        for warning in warnings:
            warning_table.add_row(
                warning.component,
                warning.issue,
                warning.fix_suggestion
            )
            
        self.console.print(warning_table)
        self.console.print()
        
    def _show_next_steps(self, validation_result: ValidationResult):
        """Show next steps based on validation results."""
        errors = [i for i in validation_result.issues if i.severity == 'error']
        auto_fixable = [i for i in errors if i.auto_fixable]
        
        if errors:
            next_steps_text = "[bold cyan]🚀 Next Steps:[/bold cyan]\n\n"
            
            if auto_fixable:
                next_steps_text += "[green]1. Quick Fix Available:[/green]\n"
                next_steps_text += "   Some issues can be automatically resolved.\n"
                next_steps_text += "   Run the launcher again with --auto-fix flag.\n\n"
                
            next_steps_text += "[yellow]2. Manual Steps:[/yellow]\n"
            next_steps_text += "   Follow the 'How to Fix' instructions above.\n"
            next_steps_text += "   After fixing issues, run the launcher again.\n\n"
            
            next_steps_text += "[blue]3. Need Help?[/blue]\n"
            next_steps_text += "   See docs/setup/DEVELOPER_SETUP.md for detailed instructions.\n"
            next_steps_text += "   Or run: toaripi-slm --help\n"
            
            self.console.print(Panel(next_steps_text, title="What to do next", border_style="cyan"))
        else:
            self._show_success_message()
            
    def _show_success_message(self):
        """Show success message when validation passes."""
        success_text = """
[bold green]✅ All Systems Ready![/bold green]

Your system is properly configured for Toaripi educational content training.

[dim]• Python environment: Ready
• Dependencies: Installed  
• Educational data: Available
• Configuration: Valid[/dim]

The trainer will now start in beginner-friendly mode.
        """
        self.console.print(Panel(success_text, title="System Validation Passed", border_style="green"))
        
    def show_educational_setup_guide(self):
        """Show guidance for educational setup."""
        guide_text = """
[bold blue]🎓 Educational Content Training[/bold blue]

The Toaripi SLM generates educational content for primary school students
while preserving cultural authenticity and language traditions.

[bold green]What this trainer does:[/bold green]
• Creates age-appropriate stories in Toaripi language
• Generates vocabulary exercises with cultural context
• Produces dialogues for classroom conversation practice
• Ensures all content respects Toaripi cultural values

[bold yellow]Target Age Groups:[/bold yellow]
• Early Childhood (3-5 years): Simple words and concepts
• Primary Lower (6-8 years): Basic stories and vocabulary  
• Primary Upper (9-11 years): Complex narratives and exercises

[bold cyan]Cultural Guidelines:[/bold cyan]
• Preserves traditional Toaripi knowledge and practices
• Respects cultural sensitivity in all generated content
• Maintains authentic language patterns and expressions
• Supports intergenerational knowledge transfer

[dim]No technical expertise required - the system guides you through each step![/dim]
        """
        self.console.print(Panel(guide_text, title="Educational Focus", border_style="blue"))
        
    def show_beginner_welcome(self):
        """Show welcome message for beginner users."""
        welcome_text = """
[bold blue]Welcome to Toaripi Language Preservation![/bold blue]

You're about to train an AI model that will help create educational
content in the Toaripi language for primary school students.

[bold green]This process will:[/bold green]
• Teach the AI about Toaripi language patterns
• Focus on age-appropriate educational content
• Preserve cultural knowledge and traditions
• Generate stories, vocabulary, and exercises

[bold yellow]Training Steps:[/bold yellow]
1. 📚 Prepare educational data with cultural validation
2. 🧠 Train the model to understand Toaripi patterns  
3. ✅ Validate content for age appropriateness
4. 🎯 Generate sample educational materials
5. 📖 Export model for classroom use

[dim]Estimated time: 30-60 minutes depending on your computer.[/dim]

Ready to preserve and teach the Toaripi language?
        """
        self.console.print(Panel(welcome_text, title="Beginner Training Mode", border_style="cyan"))
        
    def show_troubleshooting_guide(self, issue_type: str):
        """Show specific troubleshooting guidance."""
        guides = {
            "python_not_found": self._get_python_install_guide(),
            "venv_missing": self._get_venv_setup_guide(),
            "dependencies_missing": self._get_dependencies_guide(),
            "data_missing": self._get_data_setup_guide(),
            "config_missing": self._get_config_guide()
        }
        
        if issue_type in guides:
            self.console.print(guides[issue_type])
        else:
            self._show_general_troubleshooting()
            
    def _get_python_install_guide(self) -> Panel:
        """Get Python installation guide."""
        guide_text = """
[bold red]Python Not Found[/bold red]

The Toaripi SLM requires Python 3.10 or newer.

[bold green]Installation Instructions:[/bold green]

[yellow]Windows:[/yellow]
1. Visit https://python.org/downloads/
2. Download Python 3.11 (recommended)
3. Run installer and check "Add Python to PATH"
4. Restart your command prompt/terminal

[yellow]macOS:[/yellow]
brew install python@3.11

[yellow]Ubuntu/Debian:[/yellow]
sudo apt update && sudo apt install python3.11

[yellow]CentOS/RHEL:[/yellow]
sudo yum install python3.11

[dim]After installation, restart this launcher to continue.[/dim]
        """
        return Panel(guide_text, title="Python Installation Guide", border_style="red")
        
    def _get_venv_setup_guide(self) -> Panel:
        """Get virtual environment setup guide."""
        guide_text = """
[bold yellow]Virtual Environment Setup[/bold yellow]

A virtual environment keeps your Toaripi SLM installation separate
from other Python projects.

[bold green]Quick Setup:[/bold green]

1. [cyan]Create virtual environment:[/cyan]
   python -m venv .venv

2. [cyan]Activate it:[/cyan]
   Windows: .venv\\Scripts\\activate
   Unix:    source .venv/bin/activate

3. [cyan]Install requirements:[/cyan]
   pip install -e .

[dim]The launcher can auto-create this for you with --auto-fix flag.[/dim]
        """
        return Panel(guide_text, title="Virtual Environment Guide", border_style="yellow")
        
    def _get_dependencies_guide(self) -> Panel:
        """Get dependencies installation guide."""
        guide_text = """
[bold cyan]Installing Dependencies[/bold cyan]

The Toaripi SLM needs several Python packages for educational
content generation and cultural validation.

[bold green]Installation Commands:[/bold green]

1. [yellow]Basic requirements (required):[/yellow]
   pip install -e .

2. [yellow]Development tools (optional):[/yellow]
   pip install -r requirements-dev.txt

3. [yellow]Full educational features:[/yellow]
   pip install -r requirements.txt

[dim]These packages include language models, educational validation,
and cultural sensitivity checking tools.[/dim]
        """
        return Panel(guide_text, title="Dependencies Guide", border_style="cyan")
        
    def _get_data_setup_guide(self) -> Panel:
        """Get data setup guide."""
        guide_text = """
[bold magenta]Educational Data Setup[/bold magenta]

The trainer needs Toaripi-English parallel text to learn
language patterns for educational content generation.

[bold green]Required Data Files:[/bold green]

• [yellow]data/raw/Full_bible_english_toaripi.csv[/yellow]
  Main training data with parallel English-Toaripi text

• [yellow]data/samples/sample_parallel.csv[/yellow]  
  Sample data for testing and validation

[bold blue]Data Format:[/bold blue]
CSV files with columns: english, toaripi, reference

[dim]The sample data is included. For full training, add your
parallel text files to the data/raw/ directory.[/dim]
        """
        return Panel(guide_text, title="Educational Data Guide", border_style="magenta")
        
    def _get_config_guide(self) -> Panel:
        """Get configuration guide."""
        guide_text = """
[bold blue]Configuration Files[/bold blue]

Configuration files control educational content generation,
age-appropriateness, and cultural sensitivity.

[bold green]Key Configuration Files:[/bold green]

• [yellow]configs/training/toaripi_educational_config.yaml[/yellow]
  Educational objectives and cultural guidelines

• [yellow]configs/data/preprocessing_config.yaml[/yellow]
  Data processing and validation rules

[bold cyan]Default Behavior:[/cyan]
If configuration files are missing, the system uses built-in
defaults focused on primary school education and cultural preservation.

[dim]See docs/usage/CONFIGURATION_GUIDE.md for full details.[/dim]
        """
        return Panel(guide_text, title="Configuration Guide", border_style="blue")
        
    def _show_general_troubleshooting(self):
        """Show general troubleshooting information."""
        trouble_text = """
[bold red]General Troubleshooting[/bold red]

[bold green]Common Solutions:[/bold green]

1. [yellow]Restart your terminal/command prompt[/yellow]
   After installing Python or changing environment variables

2. [yellow]Check you're in the right directory[/yellow]
   The launcher must be run from the toaripi-slm project root

3. [yellow]Try the auto-fix option[/yellow]
   Run: launch-trainer.bat --auto-fix (Windows)
   Or:   ./launch-trainer.sh --auto-fix (Unix)

4. [yellow]Check documentation[/yellow]
   See docs/setup/DEVELOPER_SETUP.md for detailed instructions

[bold blue]Still need help?[/bold blue]
Run: toaripi-slm --help
Or check the troubleshooting section in the documentation.
        """
        self.console.print(Panel(trouble_text, title="Troubleshooting Help", border_style="red"))