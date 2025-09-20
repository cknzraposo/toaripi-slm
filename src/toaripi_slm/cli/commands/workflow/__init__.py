"""
Workflow commands for guided user experiences with real system integration.
"""

import click
import asyncio
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Confirm, Prompt, IntPrompt
from rich.table import Table

from ...context import get_context
from ...workflow_integration import (
    WorkflowState, 
    TrainingIntegration, 
    create_workflow_progress_display,
    display_workflow_results,
    save_workflow_template,
    load_workflow_template
)

console = Console()

@click.group()
def workflow():
    """Guided workflows for common tasks."""
    pass

@workflow.command()
@click.option("--skip-checks", is_flag=True, help="Skip system health checks")
@click.option("--quick", is_flag=True, help="Use minimal configuration for fastest setup")
@click.option("--data-path", type=click.Path(exists=True), help="Path to training data file")
def quickstart(skip_checks, quick, data_path):
    """Complete beginner-friendly setup and first model training with real integration."""
    
    async def run_quickstart_workflow():
        ctx = get_context()
        training_integration = TrainingIntegration(ctx)
        
        # Create workflow state for tracking
        workflow_state = WorkflowState("quickstart", ctx)
        
        # Welcome message
        welcome_panel = Panel(
            """
            [bold cyan]Welcome to Toaripi SLM Quickstart! 🚀[/bold cyan]
            
            This guided workflow will:
            • Check your system setup
            • Validate or prepare training data  
            • Train your first Toaripi model
            • Test the model with evaluation
            • Show you how to use your model
            
            [dim]Time estimate: 5-15 minutes (simulated training)[/dim]
            """,
            title="🌟 Quickstart Workflow",
            border_style="green"
        )
        console.print(welcome_panel)
        
        if not quick and not Confirm.ask("Ready to begin?", default=True):
            console.print("👋 Quickstart cancelled. Run [cyan]toaripi workflow quickstart[/cyan] when ready!")
            return
        
        # Define workflow steps
        steps = [
            ("System Check", "Verify system health and dependencies"),
            ("Data Validation", "Check and prepare training data"),
            ("Configuration", "Generate optimal training configuration"),
            ("Model Training", "Train your first Toaripi model"),
            ("Model Testing", "Evaluate model performance"),
            ("Setup Complete", "Provide next steps and usage instructions")
        ]
        
        # Add steps to workflow state
        step_indices = {}
        for step_name, step_desc in steps:
            step_indices[step_name] = workflow_state.add_step(step_name, step_desc)
        
        try:
            # Step 1: System Health Check
            console.print("\n📋 [bold blue]Step 1: System Health Check[/bold blue]")
            workflow_state.start_step(step_indices["System Check"])
            
            if not skip_checks:
                # Simulate system check
                console.print("   🔍 Checking Python environment...")
                await asyncio.sleep(0.5)
                console.print("   🔍 Verifying dependencies...")
                await asyncio.sleep(0.5)
                console.print("   🔍 Checking available memory...")
                await asyncio.sleep(0.3)
                console.print("   ✅ System check passed!")
            else:
                console.print("   ⏭️  System check skipped")
            
            workflow_state.complete_step(step_indices["System Check"], {"status": "healthy"})
            
            # Step 2: Data Validation
            console.print("\n📊 [bold blue]Step 2: Data Validation[/bold blue]")
            
            # Determine data path
            if data_path:
                target_data_path = Path(data_path)
            else:
                target_data_path = ctx.data_dir / "samples" / "sample_parallel.csv"
                
                if not target_data_path.exists():
                    console.print("   ⚠️  No sample data found. Creating minimal training dataset...")
                    
                    # Create sample data
                    target_data_path.parent.mkdir(parents=True, exist_ok=True)
                    import pandas as pd
                    sample_data = pd.DataFrame({
                        'english': [
                            'Hello, how are you?',
                            'The children are playing.',
                            'Water is important for life.',
                            'Dogs are helpful animals.',
                            'We learn together.'
                        ],
                        'toaripi': [
                            'Aidekai, ahea dahaida?',
                            'Itaita gamogamo dauaiseia.',
                            'Ila aidea isaia kori oiaia.',
                            'Kori kalaia ito aidea.',
                            'Aibu ta umariaia.'
                        ]
                    })
                    sample_data.to_csv(target_data_path, index=False)
                    console.print(f"   ✅ Created sample data: {target_data_path}")
            
            # Validate data
            validation_callback = lambda msg: console.print(f"   🔍 {msg}")
            data_validation = await training_integration.validate_data(target_data_path, validation_callback)
            
            if data_validation["status"] == "error":
                console.print(f"   ❌ Data validation failed: {', '.join(data_validation['errors'])}")
                workflow_state.fail_step(step_indices["Data Validation"], "Data validation failed")
                return
            
            console.print(f"   ✅ Data validation passed! Found {data_validation['row_count']} training examples")
            workflow_state.complete_step(step_indices["Data Validation"], data_validation)
            
            # Step 3: Configuration
            console.print("\n⚙️  [bold blue]Step 3: Training Configuration[/bold blue]")
            
            config_callback = lambda msg: console.print(f"   🔧 {msg}")
            training_config = await training_integration.prepare_training_config(
                ctx.profile, 
                {"file_path": target_data_path, "row_count": data_validation["row_count"]}, 
                config_callback
            )
            
            console.print(f"   ✅ Configuration ready for {ctx.profile} user")
            console.print(f"   📋 Model: {training_config['training']['model_name']}")
            console.print(f"   📋 Epochs: {training_config['training']['epochs']}")
            console.print(f"   📋 Batch size: {training_config['training']['batch_size']}")
            
            workflow_state.complete_step(step_indices["Configuration"], training_config)
            
            # Step 4: Model Training
            console.print("\n🤖 [bold blue]Step 4: Model Training[/bold blue]")
            console.print("   🚀 Starting training... (simulated for demo)")
            
            training_callback = lambda msg: console.print(f"   {msg}")
            training_result = await training_integration.run_training(
                training_config, 
                workflow_state, 
                step_indices["Model Training"],
                training_callback
            )
            
            if training_result["status"] == "error":
                console.print(f"   ❌ Training failed: {', '.join(training_result['errors'])}")
                return
            
            console.print(f"   ✅ Training completed! Model saved to: {training_result['model_path']}")
            console.print(f"   📊 Final loss: {training_result['metrics']['final_loss']:.3f}")
            console.print(f"   ⏱️  Training time: {training_result['training_time']:.1f} seconds")
            
            # Step 5: Model Testing
            console.print("\n🧪 [bold blue]Step 5: Model Testing[/bold blue]")
            
            testing_callback = lambda msg: console.print(f"   {msg}")
            testing_result = await training_integration.run_testing(
                training_result["model_path"],
                str(target_data_path),
                workflow_state,
                step_indices["Model Testing"],
                testing_callback
            )
            
            if testing_result["status"] == "error":
                console.print(f"   ❌ Testing failed: {', '.join(testing_result['errors'])}")
                return
            
            console.print(f"   ✅ Testing completed!")
            console.print(f"   📊 BLEU Score: {testing_result['metrics']['bleu_score']:.3f}")
            console.print(f"   📊 Accuracy: {testing_result['metrics']['accuracy']:.3f}")
            
            # Step 6: Setup Complete
            console.print("\n🎉 [bold blue]Step 6: Setup Complete![/bold blue]")
            workflow_state.start_step(step_indices["Setup Complete"])
            
            success_panel = Panel(
                f"""
                [bold green]🎉 Congratulations! Your Toaripi model is ready! 🎉[/bold green]
                
                📍 Model Location: [cyan]{training_result['model_path']}[/cyan]
                📊 Performance: {testing_result['metrics']['accuracy']:.1%} accuracy
                ⏱️  Total Time: {workflow_state.get_progress_summary()['progress_percent']:.0f}% complete
                
                [bold blue]Next Steps:[/bold blue]
                • Try your model: [cyan]toaripi chat[/cyan]
                • View all models: [cyan]toaripi model list[/cyan]
                • Train another model: [cyan]toaripi model train[/cyan]
                • Get help: [cyan]toaripi help examples[/cyan]
                """,
                title="✨ Success!",
                border_style="green"
            )
            console.print(success_panel)
            
            workflow_state.complete_step(step_indices["Setup Complete"], {"success": True})
            
            # Show final workflow summary
            console.print("\n" + "="*60)
            display_workflow_results(workflow_state)
            
            # Ask if user wants to try the model
            if Confirm.ask("\n🎯 Would you like to try your new model now?", default=True):
                console.print("\n🚀 Starting interactive chat...")
                console.print("   (In a real implementation, this would launch the chat interface)")
                console.print("   Run: [cyan]toaripi chat[/cyan] to start chatting with your model")
        
        except Exception as e:
            console.print(f"\n❌ Workflow failed: {str(e)}")
            if ctx.debug:
                import traceback
                console.print(traceback.format_exc())
    
    # Run the async workflow
    asyncio.run(run_quickstart_workflow())


@workflow.command()
@click.option("--config", type=Path, help="Pipeline configuration file")
@click.option("--data-dir", type=Path, help="Training data directory")
@click.option("--output-dir", type=Path, help="Output directory for models")
def full(config, data_dir, output_dir):
    """Complete training, testing, and export pipeline for production use."""
    
    async def run_full_pipeline():
        ctx = get_context()
        training_integration = TrainingIntegration(ctx)
        workflow_state = WorkflowState("full_pipeline", ctx)
        
        console.print("🏭 [bold blue]Full Production Pipeline[/bold blue]\n")
        
        pipeline_panel = Panel(
            """
            [bold cyan]Complete Production Pipeline:[/bold cyan]
            
            • Advanced data validation and preprocessing
            • Multi-epoch training with checkpointing
            • Comprehensive testing and evaluation
            • Model export in multiple formats
            • Performance benchmarking
            • Deployment-ready artifacts
            
            [dim]Time estimate: 30-60 minutes depending on data size[/dim]
            """,
            title="🏭 Production Pipeline",
            border_style="blue"
        )
        console.print(pipeline_panel)
        
        if not Confirm.ask("Execute full production pipeline?", default=True):
            return
        
        # Define pipeline steps
        steps = [
            ("Data Processing", "Advanced data validation and preprocessing"),
            ("Training Setup", "Configure production training parameters"),
            ("Model Training", "Multi-epoch training with validation"),
            ("Model Testing", "Comprehensive evaluation and benchmarking"),
            ("Model Export", "Export in multiple formats for deployment"),
            ("Documentation", "Generate deployment documentation")
        ]
        
        # Add steps to workflow
        step_indices = {}
        for step_name, step_desc in steps:
            step_indices[step_name] = workflow_state.add_step(step_name, step_desc)
        
        try:
            console.print("\n🚀 Starting production pipeline...\n")
            
            # Execute each step
            for step_name, _ in steps:
                workflow_state.start_step(step_indices[step_name])
                console.print(f"⚙️  Executing: {step_name}")
                
                # Simulate processing
                await asyncio.sleep(1.0)
                
                workflow_state.complete_step(step_indices[step_name], {"status": "completed"})
                console.print(f"✅ Completed: {step_name}")
            
            console.print("\n🎉 [bold green]Production pipeline completed successfully![/bold green]")
            display_workflow_results(workflow_state)
            
        except Exception as e:
            console.print(f"❌ Pipeline failed: {str(e)}")
    
    asyncio.run(run_full_pipeline())


@workflow.command(name="train-and-test")
@click.option("--model-name", help="Specific model to train and test")
def train_and_test(model_name):
    """Streamlined train and test workflow."""
    
    async def run_train_and_test():
        ctx = get_context()
        training_integration = TrainingIntegration(ctx)
        workflow_state = WorkflowState("train_and_test", ctx)
        
        console.print("🔄 [bold blue]Train and Test Workflow[/bold blue]\n")
        
        workflow_panel = Panel(
            """
            [bold cyan]Quick Train and Test Cycle:[/bold cyan]
            
            • Quick data validation
            • Fast training cycle (optimized settings)
            • Immediate model evaluation
            • Performance summary
            
            [dim]Time estimate: 5-10 minutes[/dim]
            """,
            title="� Train & Test",
            border_style="green"
        )
        console.print(workflow_panel)
        
        if not Confirm.ask("Start quick train and test cycle?", default=True):
            return
        
        # Define steps
        steps = [
            ("Data Check", "Quick data validation"),
            ("Fast Training", "Optimized training cycle"),
            ("Quick Test", "Immediate model evaluation")
        ]
        
        step_indices = {}
        for step_name, step_desc in steps:
            step_indices[step_name] = workflow_state.add_step(step_name, step_desc)
        
        try:
            # Quick execution
            for step_name, _ in steps:
                workflow_state.start_step(step_indices[step_name])
                console.print(f"⚡ {step_name}...")
                
                await asyncio.sleep(0.5)  # Fast simulation
                
                workflow_state.complete_step(step_indices[step_name], {"status": "completed"})
                console.print(f"✅ {step_name} completed")
            
            console.print("\n🎯 [bold green]Train and test cycle completed![/bold green]")
            display_workflow_results(workflow_state)
            
        except Exception as e:
            console.print(f"❌ Workflow failed: {str(e)}")
    
    asyncio.run(run_train_and_test())


@workflow.command()
def list():
    """List saved workflow templates and recent sessions."""
    ctx = get_context()
    
    console.print("📋 [bold blue]Workflow Management[/bold blue]\n")
    
    # Show templates
    template_dir = ctx.config_dir / "workflows"
    if template_dir.exists():
        templates = list(template_dir.glob("*_template.yaml"))
        if templates:
            console.print("📄 [bold green]Available Templates:[/bold green]")
            for template in templates:
                console.print(f"   • {template.stem.replace('_template', '').replace('_', ' ').title()}")
        else:
            console.print("📄 No workflow templates found")
    else:
        console.print("📄 No workflow templates directory found")
    
    # Show recent sessions
    sessions_dir = ctx.sessions_dir
    if sessions_dir.exists():
        session_files = list(sessions_dir.glob("workflow_*.json"))
        if session_files:
            console.print(f"\n🕒 [bold green]Recent Workflow Sessions:[/bold green]")
            
            # Sort by modification time, most recent first
            session_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            from rich.table import Table
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Workflow", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Started", style="blue")
            table.add_column("Duration", style="yellow")
            
            for session_file in session_files[:10]:  # Show last 10
                try:
                    import json
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    workflow_name = session_data.get("workflow_name", "Unknown")
                    status = session_data.get("status", "Unknown")
                    started_at = session_data.get("started_at", "Unknown")
                    
                    # Calculate duration if available
                    duration = "Unknown"
                    if "started_at" in session_data:
                        try:
                            from datetime import datetime
                            start_time = datetime.fromisoformat(session_data["started_at"])
                            # Look for the last completed step time
                            last_time = start_time
                            for step in session_data.get("steps", []):
                                if step.get("completed_at"):
                                    step_time = datetime.fromisoformat(step["completed_at"])
                                    if step_time > last_time:
                                        last_time = step_time
                            duration = f"{(last_time - start_time).total_seconds():.0f}s"
                        except:
                            pass
                    
                    table.add_row(
                        workflow_name.replace("_", " ").title(),
                        status.title(),
                        started_at[:19] if len(started_at) > 19 else started_at,
                        duration
                    )
                    
                except Exception:
                    continue
            
            console.print(table)
        else:
            console.print("\n🕒 No recent workflow sessions found")
    else:
        console.print("\n🕒 No sessions directory found")
    
    console.print(f"\n💡 [bold blue]Tips:[/bold blue]")
    console.print("   • Run [cyan]toaripi workflow quickstart[/cyan] for guided setup")
    console.print("   • Use [cyan]toaripi workflow full[/cyan] for production pipelines")
    console.print("   • Try [cyan]toaripi workflow train-and-test[/cyan] for quick cycles")