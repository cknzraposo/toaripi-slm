"""
Workflow Engine

Manages guided step-by-step workflows for different user tasks
like training models, preparing data, and deploying applications.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class WorkflowStatus(Enum):
    """Status of workflow steps."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    
    id: str
    title: str
    description: str
    command: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    optional: bool = False
    estimated_time: str = "2-5 minutes"
    tips: List[str] = field(default_factory=list)
    validation_func: Optional[Callable] = None
    status: WorkflowStatus = WorkflowStatus.NOT_STARTED
    error_message: Optional[str] = None


@dataclass
class Workflow:
    """Complete workflow definition."""
    
    id: str
    title: str
    description: str
    category: str  # training, data, deployment, configuration
    difficulty: str  # beginner, intermediate, advanced
    estimated_total_time: str = "15-30 minutes"
    steps: List[WorkflowStep] = field(default_factory=list)
    current_step: int = 0
    status: WorkflowStatus = WorkflowStatus.NOT_STARTED
    educational_focus: bool = True


class WorkflowEngine:
    """Engine for managing and executing guided workflows."""
    
    def __init__(self, context):
        """Initialize workflow engine with CLI context."""
        self.context = context
        self.workflows: Dict[str, Workflow] = {}
        self.current_workflow: Optional[Workflow] = None
        
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
        
        # Load built-in workflows
        self._load_builtin_workflows()
    
    def _load_builtin_workflows(self):
        """Load built-in educational workflows."""
        
        # Training workflow for beginners
        training_workflow = Workflow(
            id="training_beginner",
            title="🎓 Train Your First Educational Model",
            description="Step-by-step guide to train a Toaripi educational content model",
            category="training",
            difficulty="beginner",
            estimated_total_time="30-45 minutes"
        )
        
        training_workflow.steps = [
            WorkflowStep(
                id="check_data",
                title="📊 Check Training Data",
                description="Verify that your parallel English-Toaripi text data is ready",
                command="data validate",
                tips=[
                    "Your data should be in CSV format with 'english' and 'toaripi' columns",
                    "Aim for at least 1000 parallel sentences for good results",
                    "Educational content works best with simple, clear language"
                ]
            ),
            WorkflowStep(
                id="configure_training",
                title="⚙️ Configure Training Parameters",
                description="Set up model parameters optimized for educational content",
                command="config training",
                dependencies=["check_data"],
                tips=[
                    "Start with default settings - they're optimized for education",
                    "Small models (1-7B parameters) work well for classroom use",
                    "Enable cultural validation for appropriate content"
                ]
            ),
            WorkflowStep(
                id="start_training",
                title="🚀 Start Model Training",
                description="Begin training your educational content model",
                command="train start --guided",
                dependencies=["configure_training"],
                estimated_time="15-30 minutes",
                tips=[
                    "Training will take 15-30 minutes depending on your data size",
                    "The model will learn to generate age-appropriate content",
                    "You can continue other work while training runs"
                ]
            ),
            WorkflowStep(
                id="test_model",
                title="🧪 Test Your Model",
                description="Generate sample educational content to verify the model works",
                command="model test --interactive",
                dependencies=["start_training"],
                tips=[
                    "Try generating a simple story or vocabulary list",
                    "Check that content is appropriate for your target age group",
                    "Test with different prompts to see the range of capabilities"
                ]
            ),
            WorkflowStep(
                id="deploy_model",
                title="📱 Deploy for Classroom Use",
                description="Package your model for offline classroom deployment",
                command="model export --format gguf",
                dependencies=["test_model"],
                optional=True,
                tips=[
                    "GGUF format works well on Raspberry Pi and low-end computers",
                    "Quantized models are smaller but maintain good quality",
                    "Test deployment on your target hardware before classroom use"
                ]
            )
        ]
        
        self.workflows["training_beginner"] = training_workflow
        
        # Data preparation workflow
        data_workflow = Workflow(
            id="data_preparation",
            title="📚 Prepare Educational Training Data",
            description="Guide to preparing high-quality training data for educational models",
            category="data",
            difficulty="beginner",
            estimated_total_time="20-40 minutes"
        )
        
        data_workflow.steps = [
            WorkflowStep(
                id="collect_data",
                title="📥 Collect Parallel Text Data",
                description="Gather English-Toaripi parallel texts suitable for education",
                tips=[
                    "Focus on simple, clear language appropriate for students",
                    "Include diverse topics: stories, conversations, descriptions",
                    "Ensure cultural appropriateness and sensitivity"
                ]
            ),
            WorkflowStep(
                id="format_data",
                title="📋 Format Data for Training",
                description="Convert your data to the required CSV format",
                command="data format --input raw/your_data.txt --output processed/",
                dependencies=["collect_data"],
                tips=[
                    "CSV format with 'english' and 'toaripi' columns is required",
                    "Remove any inappropriate or complex content",
                    "Aim for consistent sentence lengths and complexity"
                ]
            ),
            WorkflowStep(
                id="validate_data",
                title="✅ Validate Data Quality",
                description="Check data quality and educational appropriateness",
                command="data validate --educational",
                dependencies=["format_data"],
                tips=[
                    "Automatic checks include age-appropriateness and cultural sensitivity",
                    "Review any flagged content manually",
                    "Ensure balanced representation of different topics"
                ]
            ),
            WorkflowStep(
                id="split_data",
                title="📊 Split Training and Validation Data",
                description="Divide data into training and validation sets",
                command="data split --ratio 0.8",
                dependencies=["validate_data"],
                tips=[
                    "80% training, 20% validation is a good default split",
                    "Ensure validation set represents all content types",
                    "Keep some data aside for final testing"
                ]
            )
        ]
        
        self.workflows["data_preparation"] = data_workflow
    
    def list_workflows(self, category: Optional[str] = None, difficulty: Optional[str] = None) -> List[Workflow]:
        """List available workflows with optional filtering."""
        workflows = list(self.workflows.values())
        
        if category:
            workflows = [w for w in workflows if w.category == category]
        
        if difficulty:
            workflows = [w for w in workflows if w.difficulty == difficulty]
        
        return workflows
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID."""
        return self.workflows.get(workflow_id)
    
    def start_workflow(self, workflow_id: str) -> bool:
        """Start a workflow."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return False
        
        self.current_workflow = workflow
        workflow.status = WorkflowStatus.IN_PROGRESS
        workflow.current_step = 0
        
        self._show_workflow_intro(workflow)
        return True
    
    def _show_workflow_intro(self, workflow: Workflow):
        """Show workflow introduction."""
        if RICH_AVAILABLE and self.console:
            # Rich formatted intro
            panel = Panel(
                f"[bold blue]{workflow.title}[/bold blue]\n\n"
                f"{workflow.description}\n\n"
                f"[dim]Difficulty: {workflow.difficulty.title()}[/dim]\n"
                f"[dim]Estimated time: {workflow.estimated_total_time}[/dim]\n"
                f"[dim]Steps: {len(workflow.steps)}[/dim]",
                title="🎯 Starting Workflow",
                border_style="blue"
            )
            self.console.print(panel)
        else:
            # Plain text intro
            print(f"\n🎯 Starting Workflow: {workflow.title}")
            print("=" * (len(workflow.title) + 20))
            print(f"\n📝 {workflow.description}")
            print(f"\n⏱️  Estimated time: {workflow.estimated_total_time}")
            print(f"📋 Steps: {len(workflow.steps)}")
            print(f"🎓 Difficulty: {workflow.difficulty.title()}")
    
    def show_current_step(self) -> Optional[WorkflowStep]:
        """Show current workflow step."""
        if not self.current_workflow or self.current_workflow.current_step >= len(self.current_workflow.steps):
            return None
        
        step = self.current_workflow.steps[self.current_workflow.current_step]
        
        if RICH_AVAILABLE and self.console:
            # Rich formatted step
            step_panel = Panel(
                f"[bold green]{step.title}[/bold green]\n\n"
                f"{step.description}\n\n"
                f"[dim]Estimated time: {step.estimated_time}[/dim]" +
                (f"\n[dim]Command: toaripi {step.command}[/dim]" if step.command else ""),
                title=f"Step {self.current_workflow.current_step + 1}/{len(self.current_workflow.steps)}",
                border_style="green"
            )
            self.console.print(step_panel)
            
            # Show tips
            if step.tips:
                tips_text = "\n".join([f"• {tip}" for tip in step.tips])
                tips_panel = Panel(
                    tips_text,
                    title="💡 Tips",
                    border_style="yellow"
                )
                self.console.print(tips_panel)
        else:
            # Plain text step
            step_num = self.current_workflow.current_step + 1
            total_steps = len(self.current_workflow.steps)
            
            print(f"\n📋 Step {step_num}/{total_steps}: {step.title}")
            print("-" * (len(step.title) + 15))
            print(f"\n📝 {step.description}")
            print(f"⏱️  Estimated time: {step.estimated_time}")
            
            if step.command:
                print(f"💻 Command: toaripi {step.command}")
            
            if step.tips:
                print("\n💡 Tips:")
                for tip in step.tips:
                    print(f"   • {tip}")
        
        return step
    
    def complete_step(self, step_id: str) -> bool:
        """Mark a step as completed."""
        if not self.current_workflow:
            return False
        
        # Find the step
        for i, step in enumerate(self.current_workflow.steps):
            if step.id == step_id:
                step.status = WorkflowStatus.COMPLETED
                
                # Move to next step if this is the current step
                if i == self.current_workflow.current_step:
                    self.current_workflow.current_step += 1
                
                return True
        
        return False
    
    def get_workflow_progress(self) -> Dict[str, Any]:
        """Get current workflow progress."""
        if not self.current_workflow:
            return {}
        
        completed_steps = sum(1 for step in self.current_workflow.steps 
                             if step.status == WorkflowStatus.COMPLETED)
        total_steps = len(self.current_workflow.steps)
        
        return {
            "workflow_id": self.current_workflow.id,
            "title": self.current_workflow.title,
            "current_step": self.current_workflow.current_step,
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "progress_percent": (completed_steps / total_steps) * 100 if total_steps > 0 else 0,
            "status": self.current_workflow.status.value
        }
    
    def show_workflow_progress(self):
        """Display workflow progress."""
        progress = self.get_workflow_progress()
        if not progress:
            print("No active workflow")
            return
        
        if RICH_AVAILABLE and self.console:
            # Rich progress display
            progress_table = Table(title=f"📊 {progress['title']} - Progress")
            progress_table.add_column("Step", style="bold")
            progress_table.add_column("Status", style="cyan")
            progress_table.add_column("Title")
            
            for i, step in enumerate(self.current_workflow.steps):
                status_icon = {
                    WorkflowStatus.NOT_STARTED: "⏳",
                    WorkflowStatus.IN_PROGRESS: "🔄",
                    WorkflowStatus.COMPLETED: "✅",
                    WorkflowStatus.FAILED: "❌",
                    WorkflowStatus.SKIPPED: "⏭️"
                }.get(step.status, "❓")
                
                step_num = f"{i + 1}/{len(self.current_workflow.steps)}"
                progress_table.add_row(step_num, status_icon, step.title)
            
            self.console.print(progress_table)
        else:
            # Plain text progress
            print(f"\n📊 {progress['title']} - Progress")
            print(f"Progress: {progress['completed_steps']}/{progress['total_steps']} steps completed ({progress['progress_percent']:.1f}%)")
            print("\nSteps:")
            
            for i, step in enumerate(self.current_workflow.steps):
                status_icon = {
                    WorkflowStatus.NOT_STARTED: "⏳",
                    WorkflowStatus.IN_PROGRESS: "🔄", 
                    WorkflowStatus.COMPLETED: "✅",
                    WorkflowStatus.FAILED: "❌",
                    WorkflowStatus.SKIPPED: "⏭️"
                }.get(step.status, "❓")
                
                print(f"  {status_icon} {i + 1}. {step.title}")


# Command suggestion engine
class CommandSuggestionEngine:
    """Suggests commands based on workflow context and user input."""
    
    def __init__(self, workflow_engine: WorkflowEngine):
        self.workflow_engine = workflow_engine
    
    def suggest_commands(self, context: str = "", max_suggestions: int = 3) -> List[Dict[str, str]]:
        """Suggest relevant commands based on context."""
        suggestions = []
        
        # If there's an active workflow, suggest next step
        if self.workflow_engine.current_workflow:
            current_step = self.workflow_engine.show_current_step()
            if current_step and current_step.command:
                suggestions.append({
                    "title": f"Continue {self.workflow_engine.current_workflow.title}",
                    "description": current_step.description,
                    "command": current_step.command,
                    "emoji": "🔄",
                    "priority": "high"
                })
        
        # Suggest starting workflows for beginners
        if not self.workflow_engine.current_workflow:
            suggestions.append({
                "title": "Start Training Workflow",
                "description": "Train your first educational content model with guided steps",
                "command": "workflow start training_beginner",
                "emoji": "🎓",
                "priority": "high"
            })
            
            suggestions.append({
                "title": "Prepare Training Data",
                "description": "Guide to preparing high-quality educational training data",
                "command": "workflow start data_preparation", 
                "emoji": "📚",
                "priority": "medium"
            })
        
        # General helpful commands
        suggestions.extend([
            {
                "title": "Check System Status",
                "description": "View current system health and component status",
                "command": "status",
                "emoji": "📊",
                "priority": "low"
            },
            {
                "title": "Get Help",
                "description": "Access help and documentation",
                "command": "help",
                "emoji": "❓",
                "priority": "low"
            }
        ])
        
        # Return top suggestions
        return suggestions[:max_suggestions]