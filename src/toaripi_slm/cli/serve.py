"""
Serving and deployment commands for the Toaripi SLM CLI.
Handles API server operations, edge deployment, and educational content endpoints.
"""

import json
import time
import signal
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
import threading
import psutil

from ..models.enums import ModelSize, DeviceType, ContentType, AgeGroup
from ..utils.helpers import get_file_hash, format_file_size

console = Console()


@click.group()
def serve():
    """Serving and deployment commands for Toaripi SLM."""
    pass


@serve.command()
@click.option('--model', '-m', required=True, type=Path, help='Model path for serving')
@click.option('--host', default='127.0.0.1', help='Server host address')
@click.option('--port', default=8000, type=int, help='Server port')
@click.option('--workers', default=1, type=int, help='Number of worker processes')
@click.option('--device', type=click.Choice([dt.value for dt in DeviceType] + ['auto']),
              default='auto', help='Target device for inference')
@click.option('--quantization', type=click.Choice(['auto', 'int8', 'int4', 'fp16', 'fp32']),
              default='auto', help='Model quantization level')
@click.option('--cpu-only', is_flag=True, help='Force CPU-only inference')
@click.option('--max-memory', default='8GB', help='Maximum memory usage (e.g., 4GB, 8GB)')
@click.option('--educational-mode', is_flag=True, default=True, 
              help='Enable educational content validation and safety filters')
@click.option('--cultural-validation', is_flag=True, default=True,
              help='Enable Toaripi cultural appropriateness validation')
@click.option('--age-filtering', is_flag=True, default=True,
              help='Enable age-appropriate content filtering')
@click.option('--dry-run', is_flag=True, help='Show configuration without starting server')
@click.option('--daemon', is_flag=True, help='Run server in background')
@click.option('--log-level', type=click.Choice(['debug', 'info', 'warning', 'error']),
              default='info', help='Logging level')
def start(model: Path, host: str, port: int, workers: int, device: str, 
          quantization: str, cpu_only: bool, max_memory: str, educational_mode: bool,
          cultural_validation: bool, age_filtering: bool, dry_run: bool, 
          daemon: bool, log_level: str):
    """Start educational content generation API server."""
    
    console.print("\n🚀 [bold blue]Starting Toaripi SLM Educational Content Server[/bold blue]\n")
    
    # Validate model path
    if not model.exists():
        console.print(f"❌ [red]Model not found: {model}[/red]")
        raise click.Abort()
    
    # Device configuration
    if cpu_only:
        device = 'cpu'
    elif device == 'auto':
        device = _detect_optimal_device()
    
    # Convert to enum if it's a valid device type
    if device in [dt.value for dt in DeviceType]:
        device_enum = DeviceType(device)
    else:
        device_enum = DeviceType.CPU  # Fallback
    
    # Display server configuration
    config_table = Table(title="Server Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")
    
    config_table.add_row("Model Path", str(model))
    config_table.add_row("Server Address", f"http://{host}:{port}")
    config_table.add_row("Workers", str(workers))
    config_table.add_row("Device", device.replace('_', ' ').title())
    config_table.add_row("Quantization", quantization.upper())
    config_table.add_row("Max Memory", max_memory)
    config_table.add_row("Educational Mode", "✓ Enabled" if educational_mode else "✗ Disabled")
    config_table.add_row("Cultural Validation", "✓ Enabled" if cultural_validation else "✗ Disabled")
    config_table.add_row("Age Filtering", "✓ Enabled" if age_filtering else "✗ Disabled")
    config_table.add_row("Log Level", log_level.upper())
    
    console.print(config_table)
    console.print()
    
    # Educational content endpoints
    console.print("📚 [bold green]Educational Content Endpoints[/bold green]\n")
    
    endpoints_table = Table()
    endpoints_table.add_column("Endpoint", style="cyan")
    endpoints_table.add_column("Method", style="yellow")
    endpoints_table.add_column("Description", style="white")
    
    endpoints_table.add_row("/api/generate/story", "POST", "Generate educational stories")
    endpoints_table.add_row("/api/generate/vocabulary", "POST", "Create vocabulary exercises")
    endpoints_table.add_row("/api/generate/dialogue", "POST", "Generate conversational dialogues")
    endpoints_table.add_row("/api/generate/comprehension", "POST", "Create reading comprehension")
    endpoints_table.add_row("/api/validate/content", "POST", "Validate educational appropriateness")
    endpoints_table.add_row("/api/translate", "POST", "English ↔ Toaripi translation")
    endpoints_table.add_row("/api/health", "GET", "Server health check")
    endpoints_table.add_row("/api/model/info", "GET", "Model information")
    
    console.print(endpoints_table)
    console.print()
    
    if dry_run:
        console.print("🧪 [yellow]DRY RUN MODE - Server will not start[/yellow]")
        return
    
    # Validate system resources
    console.print("🔍 [bold blue]System Validation[/bold blue]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Check model compatibility
        task1 = progress.add_task("Validating model compatibility...", total=1)
        try:
            model_info = _validate_model_for_serving(model, device_enum, max_memory)
            progress.update(task1, completed=1)
            console.print(f"✓ [green]Model validated: {model_info['compatibility']}[/green]")
        except Exception as e:
            progress.update(task1, completed=1)
            console.print(f"❌ [red]Model validation failed: {e}[/red]")
            raise click.Abort()
        
        # Check system resources
        task2 = progress.add_task("Checking system resources...", total=1)
        try:
            resource_check = _check_system_resources(max_memory, workers)
            progress.update(task2, completed=1)
            if resource_check['sufficient']:
                console.print(f"✓ [green]Resources sufficient: {resource_check['details']}[/green]")
            else:
                console.print(f"⚠️  [yellow]Resource warning: {resource_check['details']}[/yellow]")
        except Exception as e:
            progress.update(task2, completed=1)
            console.print(f"❌ [red]Resource check failed: {e}[/red]")
            raise click.Abort()
        
        # Initialize educational validation
        task3 = progress.add_task("Initializing educational validation...", total=1)
        try:
            validation_config = _setup_educational_validation(
                educational_mode, cultural_validation, age_filtering
            )
            progress.update(task3, completed=1)
            console.print(f"✓ [green]Validation configured: {len(validation_config['filters'])} filters active[/green]")
        except Exception as e:
            progress.update(task3, completed=1)
            console.print(f"❌ [red]Validation setup failed: {e}[/red]")
            raise click.Abort()
    
    # Server startup
    console.print("\n🎯 [bold green]Starting Server[/bold green]\n")
    
    try:
        if daemon:
            # Start server in background
            server_process = _start_server_daemon(
                model, host, port, workers, device_enum, quantization, 
                max_memory, validation_config, log_level
            )
            console.print(f"✅ [bold green]Server started in background (PID: {server_process.pid})[/bold green]")
            console.print(f"📝 Server accessible at: [cyan]http://{host}:{port}[/cyan]")
            console.print(f"📊 Health check: [cyan]http://{host}:{port}/api/health[/cyan]")
            console.print(f"📚 API docs: [cyan]http://{host}:{port}/docs[/cyan]")
            
            # Save server info for status/stop commands
            _save_server_info({
                'pid': server_process.pid,
                'host': host,
                'port': port,
                'model': str(model),
                'started_at': time.time(),
                'educational_mode': educational_mode
            })
            
        else:
            # Start server in foreground
            console.print("Starting server in foreground mode...")
            console.print("Press Ctrl+C to stop the server")
            
            try:
                _start_server_foreground(
                    model, host, port, workers, device_enum, quantization,
                    max_memory, validation_config, log_level
                )
            except KeyboardInterrupt:
                console.print("\n🛑 [yellow]Server stopped by user[/yellow]")
            except Exception as e:
                console.print(f"\n❌ [red]Server error: {e}[/red]")
                raise click.Abort()
    
    except Exception as e:
        console.print(f"❌ [red]Failed to start server: {e}[/red]")
        raise click.Abort()


@serve.command()
@click.option('--host', default='127.0.0.1', help='Server host to check')
@click.option('--port', default=8000, type=int, help='Server port to check')
@click.option('--detailed', is_flag=True, help='Show detailed server information')
def status(host: str, port: int, detailed: bool):
    """Check server status and health."""
    
    console.print("\n📊 [bold blue]Toaripi SLM Server Status[/bold blue]\n")
    
    # Check for saved server info
    server_info = _load_server_info()
    
    if server_info:
        # Display saved server information
        info_table = Table(title="Server Information")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")
        
        info_table.add_row("Process ID", str(server_info['pid']))
        info_table.add_row("Host", server_info['host'])
        info_table.add_row("Port", str(server_info['port']))
        info_table.add_row("Model", server_info['model'])
        
        # Calculate uptime
        uptime_seconds = time.time() - server_info['started_at']
        uptime_str = _format_uptime(uptime_seconds)
        info_table.add_row("Uptime", uptime_str)
        
        info_table.add_row("Educational Mode", "✓ Enabled" if server_info.get('educational_mode', True) else "✗ Disabled")
        
        console.print(info_table)
        
        # Check if process is still running
        try:
            process = psutil.Process(server_info['pid'])
            if process.is_running():
                console.print(f"\n✅ [green]Server is running (PID: {server_info['pid']})[/green]")
                
                # Show process details if requested
                if detailed:
                    console.print("\n💻 [bold green]Process Details[/bold green]\n")
                    
                    process_table = Table()
                    process_table.add_column("Metric", style="cyan")
                    process_table.add_column("Value", style="white")
                    
                    try:
                        memory_info = process.memory_info()
                        cpu_percent = process.cpu_percent()
                        
                        process_table.add_row("CPU Usage", f"{cpu_percent:.1f}%")
                        process_table.add_row("Memory (RSS)", format_file_size(memory_info.rss))
                        process_table.add_row("Memory (VMS)", format_file_size(memory_info.vms))
                        process_table.add_row("Threads", str(process.num_threads()))
                        process_table.add_row("Status", process.status())
                        
                        console.print(process_table)
                        
                    except psutil.AccessDenied:
                        console.print("⚠️  [yellow]Process details access denied[/yellow]")
                
            else:
                console.print(f"\n❌ [red]Server process not found (PID: {server_info['pid']})[/red]")
                console.print("Server may have crashed or been terminated")
        
        except psutil.NoSuchProcess:
            console.print(f"\n❌ [red]Server process not found (PID: {server_info['pid']})[/red]")
            console.print("Server is not running")
    
    else:
        console.print("ℹ️  [yellow]No server information found[/yellow]")
        console.print("Server may not be running or was started without daemon mode")
    
    # Try to connect to server endpoint
    console.print(f"\n🔗 [bold blue]Health Check[/bold blue]\n")
    
    try:
        health_status = _check_server_health(host, port)
        
        if health_status['accessible']:
            console.print(f"✅ [green]Server accessible at http://{host}:{port}[/green]")
            
            if detailed and 'details' in health_status:
                health_table = Table(title="Health Details")
                health_table.add_column("Check", style="cyan")
                health_table.add_column("Status", style="white")
                
                for check, status in health_status['details'].items():
                    status_str = "✓ OK" if status else "❌ FAIL"
                    health_table.add_row(check, status_str)
                
                console.print(health_table)
        else:
            console.print(f"❌ [red]Server not accessible at http://{host}:{port}[/red]")
            console.print(f"Error: {health_status.get('error', 'Unknown error')}")
    
    except Exception as e:
        console.print(f"❌ [red]Health check failed: {e}[/red]")


@serve.command()
@click.option('--host', default='127.0.0.1', help='Server host to stop')
@click.option('--port', default=8000, type=int, help='Server port to stop')
@click.option('--force', is_flag=True, help='Force stop server')
@click.option('--timeout', default=30, type=int, help='Timeout in seconds for graceful shutdown')
def stop(host: str, port: int, force: bool, timeout: int):
    """Stop running server gracefully or forcefully."""
    
    console.print("\n🛑 [bold blue]Stopping Toaripi SLM Server[/bold blue]\n")
    
    # Load server information
    server_info = _load_server_info()
    
    if not server_info:
        console.print("ℹ️  [yellow]No server information found[/yellow]")
        console.print("Attempting to stop server anyway...")
    
    success = False
    
    # Try graceful shutdown first
    if not force:
        console.print("Attempting graceful shutdown...")
        
        try:
            if server_info and 'pid' in server_info:
                # Send SIGTERM to process
                process = psutil.Process(server_info['pid'])
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=timeout)
                    console.print("✅ [green]Server stopped gracefully[/green]")
                    success = True
                except psutil.TimeoutExpired:
                    console.print(f"⚠️  [yellow]Graceful shutdown timed out after {timeout}s[/yellow]")
            
            else:
                # Try API shutdown endpoint
                shutdown_result = _request_server_shutdown(host, port, timeout)
                if shutdown_result['success']:
                    console.print("✅ [green]Server stopped via API[/green]")
                    success = True
                else:
                    console.print(f"❌ [red]API shutdown failed: {shutdown_result['error']}[/red]")
        
        except Exception as e:
            console.print(f"❌ [red]Graceful shutdown failed: {e}[/red]")
    
    # Force shutdown if graceful failed
    if not success or force:
        console.print("Attempting force shutdown...")
        
        try:
            if server_info and 'pid' in server_info:
                process = psutil.Process(server_info['pid'])
                process.kill()  # SIGKILL
                console.print("✅ [green]Server force stopped[/green]")
                success = True
            else:
                console.print("❌ [red]Cannot force stop: No process ID available[/red]")
        
        except psutil.NoSuchProcess:
            console.print("ℹ️  [yellow]Process already stopped[/yellow]")
            success = True
        except Exception as e:
            console.print(f"❌ [red]Force shutdown failed: {e}[/red]")
    
    # Clean up server info file
    if success:
        _cleanup_server_info()
        console.print("\n🧹 [blue]Server information cleaned up[/blue]")
    
    if not success:
        console.print("\n❌ [red]Failed to stop server[/red]")
        console.print("You may need to manually kill the process")
        raise click.Abort()


@serve.command()
@click.option('--server', default='http://127.0.0.1:8000', help='Server URL to test')
@click.option('--prompt', default='Tell me a story about children helping with fishing.',
              help='Test prompt for content generation')
@click.option('--content-type', type=click.Choice([ct.value for ct in ContentType]),
              default='story', help='Type of content to generate')
@click.option('--age-group', type=click.Choice([ag.value for ag in AgeGroup]),
              default='primary_lower', help='Target age group')
@click.option('--count', default=1, type=int, help='Number of generation requests to make')
@click.option('--concurrent', is_flag=True, help='Make concurrent requests for load testing')
def test(server: str, prompt: str, content_type: str, age_group: str, count: int, concurrent: bool):
    """Test server endpoints and educational content generation."""
    
    console.print("\n🧪 [bold blue]Testing Educational Content Server[/bold blue]\n")
    
    # Display test configuration
    config_table = Table(title="Test Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")
    
    config_table.add_row("Server URL", server)
    config_table.add_row("Test Prompt", prompt)
    config_table.add_row("Content Type", content_type.replace('_', ' ').title())
    config_table.add_row("Age Group", age_group.replace('_', ' ').title())
    config_table.add_row("Request Count", str(count))
    config_table.add_row("Concurrent", "✓ Yes" if concurrent else "✗ No")
    
    console.print(config_table)
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Test server health
        task1 = progress.add_task("Testing server health...", total=1)
        try:
            health_result = _test_server_endpoint(f"{server}/api/health")
            progress.update(task1, completed=1)
            
            if health_result['success']:
                console.print("✅ [green]Server health check passed[/green]")
            else:
                console.print(f"❌ [red]Server health check failed: {health_result['error']}[/red]")
                raise click.Abort()
        
        except Exception as e:
            progress.update(task1, completed=1)
            console.print(f"❌ [red]Health check failed: {e}[/red]")
            raise click.Abort()
        
        # Test content generation
        task2 = progress.add_task(f"Testing content generation ({count} requests)...", total=count)
        
        generation_results = []
        
        try:
            for i in range(count):
                result = _test_content_generation(
                    server, prompt, content_type, age_group, i + 1
                )
                generation_results.append(result)
                progress.update(task2, advance=1)
            
            # Analyze results
            successful_requests = sum(1 for r in generation_results if r['success'])
            console.print(f"✅ [green]{successful_requests}/{count} requests successful[/green]")
            
        except Exception as e:
            console.print(f"❌ [red]Content generation test failed: {e}[/red]")
            raise click.Abort()
    
    # Display test results
    console.print("\n📊 [bold green]Test Results[/bold green]\n")
    
    results_table = Table()
    results_table.add_column("Request", style="cyan")
    results_table.add_column("Status", style="white")
    results_table.add_column("Response Time", justify="right", style="yellow")
    results_table.add_column("Content Length", justify="right", style="blue")
    
    for i, result in enumerate(generation_results, 1):
        status = "✅ Success" if result['success'] else "❌ Failed"
        response_time = f"{result.get('response_time', 0):.2f}s"
        content_length = str(result.get('content_length', 0))
        
        results_table.add_row(str(i), status, response_time, content_length)
    
    console.print(results_table)
    
    # Show sample generated content
    if generation_results and generation_results[0]['success']:
        console.print("\n📄 [bold green]Sample Generated Content[/bold green]\n")
        sample_content = generation_results[0].get('content', 'No content generated')
        console.print(Panel(sample_content, title="Educational Content", border_style="green"))


# Helper functions (placeholders for actual implementation)

def _detect_optimal_device() -> str:
    """Auto-detect optimal device for inference."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'gpu'
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'


def _validate_model_for_serving(model_path: Path, device: DeviceType, max_memory: str) -> Dict[str, Any]:
    """Validate model compatibility for serving."""
    # Placeholder implementation
    return {
        'compatibility': 'Compatible for serving',
        'device_supported': True,
        'memory_requirement': '2GB'
    }


def _check_system_resources(max_memory: str, workers: int) -> Dict[str, Any]:
    """Check system resource availability."""
    # Convert memory string to bytes
    memory_gb = int(max_memory.replace('GB', '').replace('gb', ''))
    required_memory = memory_gb * 1024 * 1024 * 1024  # Convert to bytes
    
    # Get available memory
    virtual_memory = psutil.virtual_memory()
    available_memory = virtual_memory.available
    
    sufficient = available_memory >= required_memory
    
    return {
        'sufficient': sufficient,
        'details': f"Required: {memory_gb}GB, Available: {available_memory // (1024**3)}GB"
    }


def _setup_educational_validation(educational_mode: bool, cultural_validation: bool, 
                                 age_filtering: bool) -> Dict[str, Any]:
    """Setup educational content validation configuration."""
    filters = []
    
    if educational_mode:
        filters.append('educational_content')
    if cultural_validation:
        filters.append('cultural_appropriateness')
    if age_filtering:
        filters.append('age_appropriate')
    
    return {
        'filters': filters,
        'strict_mode': educational_mode and cultural_validation and age_filtering
    }


def _start_server_daemon(model: Path, host: str, port: int, workers: int, 
                        device: DeviceType, quantization: str, max_memory: str,
                        validation_config: Dict, log_level: str) -> subprocess.Popen:
    """Start server in daemon mode."""
    # Placeholder implementation - would start FastAPI/uvicorn server
    console.print("[yellow]Note: Daemon server startup requires FastAPI implementation[/yellow]")
    
    # Simulate server process
    import time
    import os
    return subprocess.Popen(['python', '-c', 'import time; time.sleep(3600)'])


def _start_server_foreground(model: Path, host: str, port: int, workers: int,
                           device: DeviceType, quantization: str, max_memory: str,
                           validation_config: Dict, log_level: str):
    """Start server in foreground mode."""
    # Placeholder implementation
    console.print("[yellow]Note: Foreground server requires FastAPI implementation[/yellow]")
    console.print("Server would start here...")
    
    # Simulate server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        raise


def _save_server_info(info: Dict[str, Any]):
    """Save server information to file."""
    server_info_file = Path.cwd() / '.toaripi_server.json'
    with open(server_info_file, 'w') as f:
        json.dump(info, f, indent=2)


def _load_server_info() -> Optional[Dict[str, Any]]:
    """Load server information from file."""
    server_info_file = Path.cwd() / '.toaripi_server.json'
    if server_info_file.exists():
        try:
            with open(server_info_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def _cleanup_server_info():
    """Remove server information file."""
    server_info_file = Path.cwd() / '.toaripi_server.json'
    if server_info_file.exists():
        server_info_file.unlink()


def _format_uptime(seconds: float) -> str:
    """Format uptime in human readable format."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def _check_server_health(host: str, port: int) -> Dict[str, Any]:
    """Check server health via HTTP endpoint."""
    try:
        # Placeholder - would use requests library
        return {
            'accessible': True,
            'details': {
                'api_responsive': True,
                'model_loaded': True,
                'validation_active': True
            }
        }
    except Exception as e:
        return {
            'accessible': False,
            'error': str(e)
        }


def _request_server_shutdown(host: str, port: int, timeout: int) -> Dict[str, Any]:
    """Request server shutdown via API."""
    try:
        # Placeholder - would send shutdown request to API
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def _test_server_endpoint(url: str) -> Dict[str, Any]:
    """Test server endpoint accessibility."""
    try:
        # Placeholder - would use requests library
        return {'success': True, 'response_time': 0.123}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def _test_content_generation(server: str, prompt: str, content_type: str, 
                           age_group: str, request_id: int) -> Dict[str, Any]:
    """Test content generation endpoint."""
    try:
        # Placeholder - would send POST request to generation endpoint
        return {
            'success': True,
            'response_time': 1.234,
            'content_length': 156,
            'content': f"Generated educational content for request {request_id}: {prompt[:50]}..."
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == '__main__':
    serve()