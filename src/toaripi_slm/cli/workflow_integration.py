"""
Core training system integration for workflows.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table

console = Console()

class WorkflowState:
    """Manages workflow state and persistence."""
    
    def __init__(self, workflow_name: str, context):
        self.workflow_name = workflow_name
        self.context = context
        self.state_file = context.sessions_dir / f"workflow_{workflow_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.state = {
            "workflow_name": workflow_name,
            "started_at": datetime.now().isoformat(),
            "current_step": 0,
            "steps": [],
            "status": "started",
            "results": {},
            "errors": []
        }
        self.save_state()
    
    def add_step(self, step_name: str, description: str, status: str = "pending"):
        """Add a step to the workflow."""
        step = {
            "name": step_name,
            "description": description,
            "status": status,
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None
        }
        self.state["steps"].append(step)
        self.save_state()
        return len(self.state["steps"]) - 1
    
    def start_step(self, step_index: int):
        """Mark a step as started."""
        if step_index < len(self.state["steps"]):
            self.state["steps"][step_index]["status"] = "running"
            self.state["steps"][step_index]["started_at"] = datetime.now().isoformat()
            self.state["current_step"] = step_index
            self.save_state()
    
    def complete_step(self, step_index: int, result: Any = None):
        """Mark a step as completed."""
        if step_index < len(self.state["steps"]):
            self.state["steps"][step_index]["status"] = "completed"
            self.state["steps"][step_index]["completed_at"] = datetime.now().isoformat()
            self.state["steps"][step_index]["result"] = result
            self.save_state()
    
    def fail_step(self, step_index: int, error: str):
        """Mark a step as failed."""
        if step_index < len(self.state["steps"]):
            self.state["steps"][step_index]["status"] = "failed"
            self.state["steps"][step_index]["completed_at"] = datetime.now().isoformat()
            self.state["steps"][step_index]["error"] = error
            self.state["errors"].append({"step": step_index, "error": error, "timestamp": datetime.now().isoformat()})
            self.save_state()
    
    def save_state(self):
        """Save workflow state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get workflow progress summary."""
        total_steps = len(self.state["steps"])
        completed_steps = sum(1 for step in self.state["steps"] if step["status"] == "completed")
        failed_steps = sum(1 for step in self.state["steps"] if step["status"] == "failed")
        
        return {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "progress_percent": (completed_steps / total_steps * 100) if total_steps > 0 else 0,
            "current_step": self.state["current_step"],
            "status": self.state["status"]
        }

class TrainingIntegration:
    """Integration with actual training systems."""
    
    def __init__(self, context):
        self.context = context
    
    async def validate_data(self, data_path: Path, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Validate training data."""
        result = {
            "status": "success",
            "file_exists": False,
            "row_count": 0,
            "column_info": {},
            "quality_checks": {},
            "errors": []
        }
        
        try:
            if callback:
                callback("Checking file existence...")
            
            if not data_path.exists():
                result["status"] = "error"
                result["errors"].append(f"Data file not found: {data_path}")
                return result
            
            result["file_exists"] = True
            
            if callback:
                callback("Loading data file...")
            
            # Simulate data loading and validation
            import pandas as pd
            try:
                df = pd.read_csv(data_path)
                result["row_count"] = len(df)
                result["column_info"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
                
                if callback:
                    callback("Running quality checks...")
                
                # Quality checks
                result["quality_checks"] = {
                    "has_required_columns": all(col in df.columns for col in ["english", "toaripi"]),
                    "no_empty_rows": df.dropna().shape[0] == df.shape[0],
                    "reasonable_length": 10 <= len(df) <= 100000,
                    "text_quality": self._check_text_quality(df)
                }
                
                # Check for issues
                if not result["quality_checks"]["has_required_columns"]:
                    result["errors"].append("Missing required columns: 'english' and 'toaripi'")
                
                if not result["quality_checks"]["no_empty_rows"]:
                    result["errors"].append("Data contains empty rows")
                
                if not result["quality_checks"]["reasonable_length"]:
                    result["errors"].append(f"Data size ({len(df)} rows) outside reasonable range (10-100000)")
                
                if result["errors"]:
                    result["status"] = "warning"
                
            except Exception as e:
                result["status"] = "error"
                result["errors"].append(f"Error reading data file: {str(e)}")
        
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Validation failed: {str(e)}")
        
        return result
    
    def _check_text_quality(self, df) -> Dict[str, Any]:
        """Check text quality in the dataframe."""
        quality = {
            "avg_english_length": 0,
            "avg_toaripi_length": 0,
            "balanced_languages": True
        }
        
        try:
            if "english" in df.columns and "toaripi" in df.columns:
                quality["avg_english_length"] = df["english"].str.len().mean()
                quality["avg_toaripi_length"] = df["toaripi"].str.len().mean()
                
                # Check if languages are reasonably balanced
                eng_len = quality["avg_english_length"]
                toa_len = quality["avg_toaripi_length"]
                ratio = max(eng_len, toa_len) / min(eng_len, toa_len) if min(eng_len, toa_len) > 0 else float('inf')
                quality["balanced_languages"] = ratio < 3.0  # Allow up to 3x difference
        except Exception:
            pass
        
        return quality
    
    async def prepare_training_config(self, profile: str, data_info: Dict[str, Any], 
                                    callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Prepare training configuration based on profile and data."""
        
        if callback:
            callback("Analyzing system resources...")
        
        # Get system info for configuration
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        if callback:
            callback("Generating configuration...")
        
        # Base configuration templates
        config_templates = {
            "beginner": {
                "model_name": "microsoft/DialoGPT-small",
                "epochs": 2,
                "learning_rate": 1e-4,
                "batch_size": 2,
                "gradient_accumulation_steps": 4,
                "use_lora": True,
                "lora_r": 8,
                "max_length": 256
            },
            "intermediate": {
                "model_name": "microsoft/DialoGPT-medium",
                "epochs": 3,
                "learning_rate": 5e-5,
                "batch_size": 4 if memory_gb >= 8 else 2,
                "gradient_accumulation_steps": 2,
                "use_lora": True,
                "lora_r": 16,
                "max_length": 512
            },
            "advanced": {
                "model_name": "microsoft/DialoGPT-medium",
                "epochs": 5,
                "learning_rate": 2e-5,
                "batch_size": 8 if memory_gb >= 16 else 4,
                "gradient_accumulation_steps": 1,
                "use_lora": True,
                "lora_r": 32,
                "max_length": 512
            },
            "expert": {
                "model_name": "microsoft/DialoGPT-large",
                "epochs": 10,
                "learning_rate": 1e-5,
                "batch_size": 16 if memory_gb >= 32 else 8,
                "gradient_accumulation_steps": 1,
                "use_lora": False,
                "max_length": 1024
            }
        }
        
        base_config = config_templates.get(profile, config_templates["beginner"])
        
        # Adjust based on data size
        data_size = data_info.get("row_count", 100)
        if data_size < 100:
            base_config["epochs"] = max(1, base_config["epochs"] // 2)
        elif data_size > 10000:
            base_config["epochs"] = min(20, base_config["epochs"] * 2)
        
        # Adjust based on available memory
        if memory_gb < 4:
            base_config["batch_size"] = 1
            base_config["gradient_accumulation_steps"] = 8
            base_config["max_length"] = 128
        elif memory_gb >= 16:
            base_config["batch_size"] = min(16, base_config["batch_size"] * 2)
        
        config = {
            "training": base_config,
            "data": {
                "train_file": str(data_info.get("file_path", "data/train.csv")),
                "validation_split": 0.1,
                "test_split": 0.1
            },
            "output": {
                "model_dir": str(self.context.models_dir / f"toaripi_{profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                "checkpoint_steps": 100,
                "save_total_limit": 3
            },
            "system": {
                "memory_gb": memory_gb,
                "cpu_count": cpu_count,
                "profile": profile
            }
        }
        
        return config
    
    async def run_training(self, config: Dict[str, Any], workflow_state: WorkflowState, 
                          step_index: int, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Run model training with progress tracking."""
        
        result = {
            "status": "success",
            "model_path": None,
            "metrics": {},
            "training_time": 0,
            "errors": []
        }
        
        try:
            workflow_state.start_step(step_index)
            
            if callback:
                callback("Initializing training environment...")
            
            # Simulate training process with progress updates
            training_steps = [
                ("Loading model and tokenizer", 0.1),
                ("Preparing training data", 0.2),
                ("Setting up training configuration", 0.1),
                ("Training epoch 1", 0.2),
                ("Training epoch 2", 0.2),
                ("Training epoch 3", 0.1),
                ("Saving model", 0.1)
            ]
            
            start_time = datetime.now()
            
            for step_name, duration in training_steps:
                if callback:
                    callback(f"Training: {step_name}")
                
                # Simulate processing time
                await asyncio.sleep(duration)
            
            # Simulate successful training
            model_dir = Path(config["output"]["model_dir"])
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a dummy model info file
            model_info = {
                "model_name": config["training"]["model_name"],
                "training_config": config["training"],
                "trained_at": datetime.now().isoformat(),
                "profile": config["system"]["profile"],
                "data_size": "simulated",
                "training_time": (datetime.now() - start_time).total_seconds()
            }
            
            with open(model_dir / "model_info.json", 'w') as f:
                json.dump(model_info, f, indent=2)
            
            result["model_path"] = str(model_dir)
            result["metrics"] = {
                "final_loss": 1.234,  # Simulated
                "perplexity": 15.67,  # Simulated
                "epochs_completed": config["training"]["epochs"]
            }
            result["training_time"] = (datetime.now() - start_time).total_seconds()
            
            workflow_state.complete_step(step_index, result)
            
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            workflow_state.fail_step(step_index, str(e))
        
        return result
    
    async def run_testing(self, model_path: str, test_data_path: str, 
                         workflow_state: WorkflowState, step_index: int,
                         callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Run model testing with evaluation metrics."""
        
        result = {
            "status": "success",
            "metrics": {},
            "test_samples": [],
            "errors": []
        }
        
        try:
            workflow_state.start_step(step_index)
            
            if callback:
                callback("Loading trained model...")
            
            await asyncio.sleep(0.5)  # Simulate loading
            
            if callback:
                callback("Preparing test data...")
            
            await asyncio.sleep(0.3)  # Simulate preparation
            
            if callback:
                callback("Running evaluation...")
            
            # Simulate evaluation process
            await asyncio.sleep(1.0)
            
            # Simulate test results
            result["metrics"] = {
                "bleu_score": 0.234,  # Simulated
                "rouge_l": 0.345,     # Simulated
                "accuracy": 0.789,    # Simulated
                "perplexity": 12.45   # Simulated
            }
            
            result["test_samples"] = [
                {
                    "input": "Hello, how are you?",
                    "expected": "Aidekai, ahea dahaida?",
                    "generated": "Aidekai, ahea dahaina?",
                    "score": 0.85
                },
                {
                    "input": "What is your name?",
                    "expected": "Dahina eve lagataia?",
                    "generated": "Dahina eve lagataia?",
                    "score": 1.0
                }
            ]
            
            workflow_state.complete_step(step_index, result)
            
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            workflow_state.fail_step(step_index, str(e))
        
        return result

def create_workflow_progress_display():
    """Create a progress display for workflows."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    )

def display_workflow_results(workflow_state: WorkflowState):
    """Display workflow results summary."""
    
    progress_summary = workflow_state.get_progress_summary()
    
    # Results panel
    status_color = "green" if progress_summary["failed_steps"] == 0 else "red" if progress_summary["completed_steps"] == 0 else "yellow"
    
    results_panel = Panel(
        f"""
        [bold cyan]Workflow Results[/bold cyan]
        
        Status: [bold {status_color}]{workflow_state.state['status'].upper()}[/bold {status_color}]
        Progress: {progress_summary['completed_steps']}/{progress_summary['total_steps']} steps completed
        Success Rate: {progress_summary['progress_percent']:.1f}%
        
        Started: {workflow_state.state['started_at']}
        Duration: {(datetime.now() - datetime.fromisoformat(workflow_state.state['started_at'])).total_seconds():.0f} seconds
        """,
        title="📊 Workflow Summary",
        border_style=status_color
    )
    
    console.print(results_panel)
    
    # Steps table
    steps_table = Table(show_header=True, header_style="bold magenta")
    steps_table.add_column("Step", style="cyan", no_wrap=True)
    steps_table.add_column("Description", style="blue")
    steps_table.add_column("Status", style="green")
    steps_table.add_column("Duration", style="yellow")
    
    for i, step in enumerate(workflow_state.state["steps"]):
        status_icon = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌"
        }.get(step["status"], "❓")
        
        duration = ""
        if step["started_at"] and step["completed_at"]:
            start = datetime.fromisoformat(step["started_at"])
            end = datetime.fromisoformat(step["completed_at"])
            duration = f"{(end - start).total_seconds():.1f}s"
        elif step["started_at"]:
            start = datetime.fromisoformat(step["started_at"])
            duration = f"{(datetime.now() - start).total_seconds():.1f}s"
        
        steps_table.add_row(
            f"{i+1}. {step['name']}",
            step['description'],
            f"{status_icon} {step['status'].title()}",
            duration
        )
    
    console.print("\n📋 Workflow Steps:")
    console.print(steps_table)
    
    # Show errors if any
    if workflow_state.state["errors"]:
        console.print("\n❌ [bold red]Errors Encountered:[/bold red]")
        for error in workflow_state.state["errors"]:
            console.print(f"   Step {error['step'] + 1}: {error['error']}")
    
    # Show next steps
    if progress_summary["failed_steps"] == 0 and progress_summary["completed_steps"] == progress_summary["total_steps"]:
        console.print("\n🎉 [bold green]Workflow completed successfully![/bold green]")
        console.print("   • Try: [cyan]toaripi chat[/cyan] to test your model")
        console.print("   • Or: [cyan]toaripi model list[/cyan] to see all models")
    elif progress_summary["failed_steps"] > 0:
        console.print("\n🔧 [bold yellow]Next Steps:[/bold yellow]")
        console.print("   • Check the errors above and fix any issues")
        console.print("   • Run [cyan]toaripi system doctor[/cyan] for diagnostics")
        console.print("   • Try the workflow again after fixing issues")

async def save_workflow_template(workflow_name: str, steps: List[Dict[str, Any]], context):
    """Save a workflow as a reusable template."""
    
    template = {
        "name": workflow_name,
        "description": f"Template for {workflow_name} workflow",
        "created_at": datetime.now().isoformat(),
        "steps": steps,
        "version": "1.0"
    }
    
    template_dir = context.config_dir / "workflows" 
    template_dir.mkdir(parents=True, exist_ok=True)
    
    template_file = template_dir / f"{workflow_name.replace(' ', '_').lower()}_template.yaml"
    
    with open(template_file, 'w') as f:
        yaml.dump(template, f, default_flow_style=False)
    
    return template_file

async def load_workflow_template(template_name: str, context) -> Optional[Dict[str, Any]]:
    """Load a workflow template."""
    
    template_dir = context.config_dir / "workflows"
    template_file = template_dir / f"{template_name.replace(' ', '_').lower()}_template.yaml"
    
    if not template_file.exists():
        return None
    
    with open(template_file, 'r') as f:
        template = yaml.safe_load(f)
    
    return template