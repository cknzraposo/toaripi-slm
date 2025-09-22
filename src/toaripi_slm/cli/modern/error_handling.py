"""
Smart Error Handling & Recovery System

Provides intelligent error messages, recovery suggestions, and proactive
problem prevention with user-friendly explanations and actionable solutions.
"""

import sys
import traceback
from typing import List, Dict, Any, Optional, Type, Callable
from dataclasses import dataclass
from pathlib import Path
import re

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.prompt import Confirm, Prompt
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .framework import CLIContext


@dataclass
class ErrorSolution:
    """Represents a suggested solution for an error."""
    
    title: str
    description: str
    command: Optional[str] = None
    automatic: bool = False  # Can be applied automatically
    difficulty: str = "easy"  # easy, medium, hard
    estimated_time: str = "1 minute"
    
    def apply_automatically(self) -> bool:
        """Apply the solution automatically if possible."""
        # This would contain logic to automatically fix simple issues
        return False


@dataclass
class RecoveryPlan:
    """Represents a plan to recover from an error."""
    
    error_type: str
    description: str
    solutions: List[ErrorSolution]
    prevention_tips: List[str]
    
    def get_best_solution(self, user_experience: str = "beginner") -> Optional[ErrorSolution]:
        """Get the best solution based on user experience level."""
        
        if not self.solutions:
            return None
        
        # For beginners, prefer automatic or easy solutions
        if user_experience == "beginner":
            for solution in self.solutions:
                if solution.automatic or solution.difficulty == "easy":
                    return solution
        
        # Return first solution as default
        return self.solutions[0]


class ErrorHandler:
    """Handles error formatting and user-friendly error messages."""
    
    def __init__(self, context: CLIContext):
        self.context = context
        self.console = Console() if RICH_AVAILABLE else None
        self._error_patterns = self._initialize_error_patterns()
    
    def format_error(self, error: Exception, context_info: Dict[str, Any] = None) -> None:
        """Format and display a user-friendly error message."""
        
        if context_info is None:
            context_info = {}
        
        # Analyze error to determine type and cause
        error_analysis = self._analyze_error(error, context_info)
        
        # Create recovery plan
        recovery_plan = self._create_recovery_plan(error_analysis)
        
        # Display error information
        self._display_error_message(error, error_analysis, recovery_plan)
        
        # Offer to apply automatic solutions
        self._offer_automatic_recovery(recovery_plan)
    
    def _analyze_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error to determine cause and classification."""
        
        error_type = type(error).__name__
        error_message = str(error)
        
        analysis = {
            "type": error_type,
            "message": error_message,
            "category": "unknown",
            "severity": "medium",
            "user_fault": False,
            "recoverable": True,
            "context": context
        }
        
        # File not found errors
        if isinstance(error, FileNotFoundError):
            analysis.update({
                "category": "missing_file",
                "severity": "medium",
                "user_fault": True,
                "recoverable": True
            })
        
        # Permission errors
        elif isinstance(error, PermissionError):
            analysis.update({
                "category": "permission",
                "severity": "medium",
                "user_fault": False,
                "recoverable": True
            })
        
        # Import errors (missing dependencies)
        elif isinstance(error, ImportError):
            analysis.update({
                "category": "missing_dependency",
                "severity": "high",
                "user_fault": False,
                "recoverable": True
            })
        
        # Memory errors
        elif isinstance(error, MemoryError):
            analysis.update({
                "category": "insufficient_memory",
                "severity": "high",
                "user_fault": False,
                "recoverable": True
            })
        
        # Keyboard interrupt (user cancellation)
        elif isinstance(error, KeyboardInterrupt):
            analysis.update({
                "category": "user_cancellation",
                "severity": "low",
                "user_fault": True,
                "recoverable": True
            })
        
        # Value errors (bad input)
        elif isinstance(error, ValueError):
            analysis.update({
                "category": "invalid_input",
                "severity": "low",
                "user_fault": True,
                "recoverable": True
            })
        
        return analysis
    
    def _create_recovery_plan(self, error_analysis: Dict[str, Any]) -> RecoveryPlan:
        """Create a recovery plan based on error analysis."""
        
        category = error_analysis["category"]
        
        if category == "missing_file":
            return RecoveryPlan(
                error_type="Missing File",
                description="A required file could not be found.",
                solutions=[
                    ErrorSolution(
                        title="Check file path",
                        description="Verify the file path is correct and the file exists",
                        difficulty="easy",
                        estimated_time="1 minute"
                    ),
                    ErrorSolution(
                        title="Create missing directories",
                        description="Create any missing directories in the path",
                        command="mkdir -p",
                        automatic=True,
                        difficulty="easy"
                    )
                ],
                prevention_tips=[
                    "Use absolute paths when possible",
                    "Check file existence before operations",
                    "Keep project files organized in standard directories"
                ]
            )
        
        elif category == "missing_dependency":
            return RecoveryPlan(
                error_type="Missing Dependency",
                description="A required Python package is not installed.",
                solutions=[
                    ErrorSolution(
                        title="Install missing packages",
                        description="Install the required packages using pip",
                        command="pip install -r requirements.txt",
                        automatic=True,
                        difficulty="easy",
                        estimated_time="2-5 minutes"
                    )
                ],
                prevention_tips=[
                    "Run 'pip install -r requirements.txt' after setup",
                    "Use virtual environments for isolation",
                    "Keep requirements.txt up to date"
                ]
            )
        
        elif category == "permission":
            return RecoveryPlan(
                error_type="Permission Denied",
                description="You don't have permission to access this file or directory.",
                solutions=[
                    ErrorSolution(
                        title="Run with appropriate permissions",
                        description="Try running the command with administrator/sudo privileges",
                        difficulty="medium",
                        estimated_time="1 minute"
                    ),
                    ErrorSolution(
                        title="Change file permissions",
                        description="Modify file permissions to allow access",
                        difficulty="medium"
                    )
                ],
                prevention_tips=[
                    "Ensure you have write access to project directories",
                    "Run commands from your user directory when possible",
                    "Use virtual environments to avoid system conflicts"
                ]
            )
        
        elif category == "insufficient_memory":
            return RecoveryPlan(
                error_type="Insufficient Memory",
                description="Your system doesn't have enough memory for this operation.",
                solutions=[
                    ErrorSolution(
                        title="Reduce batch size",
                        description="Use smaller batch sizes for training",
                        difficulty="easy",
                        estimated_time="1 minute"
                    ),
                    ErrorSolution(
                        title="Close other applications",
                        description="Free up memory by closing unnecessary programs",
                        difficulty="easy"
                    ),
                    ErrorSolution(
                        title="Use model quantization",
                        description="Use 4-bit or 8-bit quantization to reduce memory usage",
                        difficulty="medium"
                    )
                ],
                prevention_tips=[
                    "Monitor memory usage during training",
                    "Start with smaller models and datasets",
                    "Use CPU-only mode if GPU memory is limited"
                ]
            )
        
        elif category == "user_cancellation":
            return RecoveryPlan(
                error_type="Operation Cancelled",
                description="You cancelled the operation.",
                solutions=[
                    ErrorSolution(
                        title="Resume operation",
                        description="Restart the operation if needed",
                        command="Resume last command",
                        difficulty="easy"
                    )
                ],
                prevention_tips=[
                    "Use Ctrl+C to safely cancel operations",
                    "Save progress frequently during long operations"
                ]
            )
        
        else:
            # Generic recovery plan
            return RecoveryPlan(
                error_type="Unknown Error",
                description="An unexpected error occurred.",
                solutions=[
                    ErrorSolution(
                        title="Check system status",
                        description="Run system diagnostics to identify issues",
                        command="toaripi status",
                        difficulty="easy"
                    ),
                    ErrorSolution(
                        title="Get help",
                        description="Consult documentation or get support",
                        command="toaripi help",
                        difficulty="easy"
                    )
                ],
                prevention_tips=[
                    "Keep your system and dependencies up to date",
                    "Run 'toaripi status' regularly to check system health"
                ]
            )
    
    def _display_error_message(
        self, 
        error: Exception, 
        analysis: Dict[str, Any], 
        recovery_plan: RecoveryPlan
    ) -> None:
        """Display a user-friendly error message."""
        
        if not RICH_AVAILABLE:
            self._display_simple_error(error, analysis, recovery_plan)
            return
        
        # Create error content
        error_text = Text()
        
        # Error header
        if analysis["severity"] == "high":
            error_text.append("❌ ", style="bold red")
            header_style = "bold red"
        elif analysis["severity"] == "medium":
            error_text.append("⚠️ ", style="bold yellow")
            header_style = "bold yellow"
        else:
            error_text.append("ℹ️ ", style="bold blue")
            header_style = "bold blue"
        
        error_text.append(f"{recovery_plan.error_type}\n\n", style=header_style)
        
        # What happened
        error_text.append("🔍 What happened:\n", style="bold cyan")
        error_text.append(f"   {recovery_plan.description}\n\n")
        
        # Technical details (for advanced users)
        if self.context.user_profile and self.context.user_profile.experience_level in ["advanced", "expert"]:
            error_text.append("🔧 Technical details:\n", style="bold dim")
            error_text.append(f"   {analysis['type']}: {analysis['message']}\n\n", style="dim")
        
        # Solutions
        error_text.append("✅ What you can do:\n", style="bold green")
        
        best_solution = recovery_plan.get_best_solution(
            self.context.user_profile.experience_level if self.context.user_profile else "beginner"
        )
        
        if best_solution:
            error_text.append(f"   1. {best_solution.title}\n", style="green")
            error_text.append(f"      {best_solution.description}\n", style="dim")
            if best_solution.command:
                error_text.append(f"      💻 Try: ", style="dim")
                error_text.append(f"{best_solution.command}\n", style="bold cyan")
            error_text.append("\n")
        
        # Additional solutions
        for i, solution in enumerate(recovery_plan.solutions[1:3], 2):  # Show up to 3 total
            error_text.append(f"   {i}. {solution.title}\n", style="green")
            error_text.append(f"      {solution.description}\n", style="dim")
        
        # Prevention tips
        if recovery_plan.prevention_tips:
            error_text.append("\n💡 To prevent this in the future:\n", style="bold yellow")
            for tip in recovery_plan.prevention_tips[:2]:  # Show top 2 tips
                error_text.append(f"   • {tip}\n", style="yellow")
        
        # Create panel
        panel_style = "red" if analysis["severity"] == "high" else "yellow"
        panel = Panel(
            error_text,
            title="🚨 Oops! Something went wrong",
            border_style=panel_style,
            padding=(1, 2)
        )
        
        self.console.print()
        self.console.print(panel)
        self.console.print()
    
    def _display_simple_error(
        self, 
        error: Exception, 
        analysis: Dict[str, Any], 
        recovery_plan: RecoveryPlan
    ) -> None:
        """Display error message without rich formatting."""
        
        print(f"\n❌ {recovery_plan.error_type}")
        print(f"🔍 What happened: {recovery_plan.description}")
        
        if recovery_plan.solutions:
            print("✅ What you can do:")
            for i, solution in enumerate(recovery_plan.solutions[:2], 1):
                print(f"   {i}. {solution.title}: {solution.description}")
                if solution.command:
                    print(f"      Try: {solution.command}")
        print()
    
    def _offer_automatic_recovery(self, recovery_plan: RecoveryPlan) -> None:
        """Offer to apply automatic recovery solutions."""
        
        automatic_solutions = [s for s in recovery_plan.solutions if s.automatic]
        
        if not automatic_solutions:
            return
        
        for solution in automatic_solutions:
            if RICH_AVAILABLE:
                should_apply = Confirm.ask(
                    f"🔧 Would you like me to {solution.title.lower()} automatically?",
                    default=True
                )
            else:
                response = input(f"🔧 Would you like me to {solution.title.lower()} automatically? (Y/n): ").strip().lower()
                should_apply = response in ["", "y", "yes"]
            
            if should_apply:
                success = solution.apply_automatically()
                if success:
                    print(f"✅ {solution.title} completed successfully!")
                else:
                    print(f"❌ Could not {solution.title.lower()} automatically. Please try manually.")
    
    def _initialize_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns for common error recognition."""
        
        return {
            "torch_not_found": {
                "pattern": r"No module named 'torch'",
                "category": "missing_dependency",
                "solution": "pip install torch"
            },
            "transformers_not_found": {
                "pattern": r"No module named 'transformers'",
                "category": "missing_dependency", 
                "solution": "pip install transformers"
            },
            "cuda_out_of_memory": {
                "pattern": r"CUDA out of memory",
                "category": "insufficient_memory",
                "solution": "Reduce batch size or use CPU"
            },
            "file_not_found_csv": {
                "pattern": r"No such file.*\.csv",
                "category": "missing_file",
                "solution": "Check data file path"
            }
        }


class SmartErrorRecovery:
    """Main error recovery system that coordinates all error handling."""
    
    def __init__(self, context: CLIContext):
        self.context = context
        self.error_handler = ErrorHandler(context)
        self._original_excepthook = sys.excepthook
    
    def install_global_handler(self) -> None:
        """Install global exception handler for better error messages."""
        
        def enhanced_excepthook(exc_type: Type[BaseException], exc_value: BaseException, exc_traceback):
            # Don't handle KeyboardInterrupt specially in debug mode
            if exc_type == KeyboardInterrupt and not self.context.verbose:
                print("\n🛑 Operation cancelled by user.")
                return
            
            # For expected exceptions, show user-friendly messages
            if exc_type in [FileNotFoundError, PermissionError, ImportError, ValueError]:
                self.error_handler.format_error(exc_value)
                return
            
            # For unexpected exceptions, show technical details in verbose mode
            if self.context.verbose:
                print("\n🐛 Unexpected error occurred:")
                self._original_excepthook(exc_type, exc_value, exc_traceback)
            else:
                print("\n❌ An unexpected error occurred.")
                print("💡 Run with --verbose flag for technical details")
                print(f"   Error: {exc_value}")
        
        sys.excepthook = enhanced_excepthook
    
    def restore_original_handler(self) -> None:
        """Restore the original exception handler."""
        sys.excepthook = self._original_excepthook
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Handle a specific error with context."""
        self.error_handler.format_error(error, context)
    
    @staticmethod
    def with_error_handling(context: CLIContext):
        """Decorator to add error handling to CLI commands."""
        
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                recovery = SmartErrorRecovery(context)
                recovery.install_global_handler()
                
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    recovery.handle_error(e, {"command": func.__name__})
                    sys.exit(1)
                finally:
                    recovery.restore_original_handler()
            
            return wrapper
        return decorator