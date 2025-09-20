#!/usr/bin/env python3
"""
🌟 Toaripi SLM Universal Launcher
Cross-platform launcher that works on Windows, Linux, and macOS

Usage:
    python start_toaripi.py
    python3 start_toaripi.py
"""

import os
import sys
import subprocess
import platform
import shutil
import time
import threading
from pathlib import Path

# Colors and symbols
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

class Symbols:
    CHECK = "✅"
    CROSS = "❌"
    WARN = "⚠️"
    ROCKET = "🚀"
    GEAR = "⚙️"
    PYTHON = "🐍"

def print_colored(text, color=Colors.NC):
    """Print colored text (Windows-safe)"""
    if platform.system() == "Windows":
        # Simple fallback for Windows
        text = (text.replace('✅', '[OK]').replace('❌', '[ERROR]')
                   .replace('⚠️', '[WARN]').replace('🚀', '[ROCKET]')
                   .replace('⚙️', '[GEAR]').replace('🐍', '[PYTHON]'))
        print(text)
    else:
        print(f"{color}{text}{Colors.NC}")

def print_step(text):
    print_colored(f"{Symbols.GEAR} {text}", Colors.PURPLE)

def print_success(text):
    print_colored(f"{Symbols.CHECK} {text}", Colors.GREEN)

def print_warning(text):
    print_colored(f"{Symbols.WARN} {text}", Colors.YELLOW)

def print_error(text):
    print_colored(f"{Symbols.CROSS} {text}", Colors.RED)

def print_info(text):
    print_colored(f"ℹ️  {text}", Colors.CYAN)

def show_progress_bar(duration=30, message="Installing dependencies"):
    """Show a progress bar for long-running operations"""
    if platform.system() == "Windows":
        # Simple dots for Windows compatibility
        print_info(f"{message}...")
        for i in range(duration):
            print(".", end="", flush=True)
            time.sleep(1)
        print()
    else:
        # Fancy progress bar for Unix systems
        print_info(message)
        bar_length = 40
        for i in range(duration):
            percent = (i + 1) / duration
            filled_length = int(bar_length * percent)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            elapsed_time = i + 1
            print(f"\r{Colors.PURPLE}[{bar}] {percent:.1%} ({elapsed_time}s){Colors.NC}", end="", flush=True)
            time.sleep(1)
        print()

def show_spinner(stop_event, message="Processing"):
    """Show a spinner while a process is running"""
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧']
    if platform.system() == "Windows":
        spinner_chars = ['|', '/', '-', '\\']  # Windows-safe characters
    
    i = 0
    while not stop_event.is_set():
        char = spinner_chars[i % len(spinner_chars)]
        print(f"\r{Colors.CYAN}{char} {message}...{Colors.NC}", end="", flush=True)
        time.sleep(0.1)
        i += 1

def run_command_with_progress(cmd, quiet=False, shell=False, estimated_time=None):
    """Run a command with progress indication"""
    if estimated_time and estimated_time > 10:
        # For long operations, show estimated progress
        print_info(f"Estimated time: {estimated_time} seconds")
        
        # Start spinner in background
        stop_event = threading.Event()
        spinner_thread = threading.Thread(target=show_spinner, args=(stop_event, "Installing"))
        spinner_thread.start()
        
        try:
            result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
            success = result.returncode == 0
            stop_event.set()
            spinner_thread.join()
            print(f"\r{' ' * 50}\r", end="")  # Clear spinner line
            
            return success, result.stdout, result.stderr
        except Exception as e:
            stop_event.set()
            spinner_thread.join()
            print(f"\r{' ' * 50}\r", end="")  # Clear spinner line
            return False, "", str(e)
    else:
        # For quick operations, use regular command execution
        return run_command(cmd, quiet, shell)

def check_toaripi_installation(python_exe):
    """Check if Toaripi SLM is already installed and functional."""
    print_info("Checking existing installation...")
    
    # Check if package is installed
    check_cmd = [str(python_exe), '-c', 'import toaripi_slm; print("package_installed")']
    package_installed, _, _ = run_command(check_cmd, quiet=True)
    
    if package_installed:
        print_success("Toaripi SLM package found")
        
        # Check if CLI is functional
        cli_check_cmd = [str(python_exe), '-c', 'from src.toaripi_slm.cli import cli; print("cli_functional")']
        cli_functional, _, _ = run_command(cli_check_cmd, quiet=True)
        
        if cli_functional:
            print_success("CLI is functional")
            return "ready"
        else:
            print_warning("CLI needs updating")
            return "needs_update"
    else:
        print_info("Package not installed")
        return "needs_install"

def check_dependencies(python_exe):
    """Check which major dependencies are already installed."""
    print_info("Checking dependency status...")
    
    deps_to_check = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("accelerate", "Accelerate"),
        ("peft", "PEFT"),
        ("click", "Click"),
        ("rich", "Rich")
    ]
    
    installed_deps = []
    missing_deps = []
    
    for module, name in deps_to_check:
        check_cmd = [str(python_exe), '-c', f'import {module}; print("installed")']
        success, _, _ = run_command(check_cmd, quiet=True)
        
        if success:
            installed_deps.append(name)
        else:
            missing_deps.append(name)
    
    if installed_deps:
        print_success(f"Found existing dependencies: {', '.join(installed_deps)}")
    
    if missing_deps:
        print_info(f"Need to install: {', '.join(missing_deps)}")
    
    return len(missing_deps) == 0

def install_toaripi_slm(python_exe):
    """Install Toaripi SLM with detailed progress."""
    print_info("Installing Toaripi SLM package...")
    
    # Check dependencies first
    all_deps_present = check_dependencies(python_exe)
    
    if all_deps_present:
        print_success("All major dependencies already present - installation should be fast")
        estimated_time = 15  # Fast install
    else:
        print_info("Installing missing dependencies - this may take several minutes...")
        estimated_time = 180  # 3 minutes for full dependency install
    
    # Try quiet install first with progress
    print_step("Starting installation process...")
    success, _, stderr = run_command_with_progress(
        [str(python_exe), '-m', 'pip', 'install', '-e', '.'], 
        quiet=True, 
        estimated_time=estimated_time
    )
    
    if success:
        print_success("Installation completed successfully")
        return True
    
    # If quiet install failed, try verbose
    print_warning("Quiet installation had issues, trying with detailed output...")
    print_info("This may take a few minutes for large dependencies...")
    
    # Show progress bar for verbose install
    print_step("Retrying installation with detailed output...")
    success, stdout, stderr = run_command_with_progress(
        [str(python_exe), '-m', 'pip', 'install', '-e', '.', '--verbose'], 
        estimated_time=estimated_time + 30
    )
    
    if success:
        print_success("Installation completed successfully")
        return True
    else:
        print_error("Installation failed")
        print(f"Error details: {stderr}")
        print("\nTroubleshooting suggestions:")
        print("1. Check internet connection")
        print("2. Try upgrading pip: python -m pip install --upgrade pip")
        print("3. Check available disk space")
        print("4. Try installing without cache: pip install -e . --no-cache-dir")
        return False

def run_command(cmd, quiet=False, shell=False):
    """Run a command and return success status"""
    try:
        if quiet:
            result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, shell=shell)
            return result.returncode == 0, "", ""
    except Exception as e:
        return False, "", str(e)

def find_python():
    """Find the best Python executable"""
    python_candidates = ['python3', 'python', 'py']
    
    for candidate in python_candidates:
        if shutil.which(candidate):
            try:
                result = subprocess.run([candidate, '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    version = result.stdout.strip().split()[1]
                    major, minor = map(int, version.split('.')[:2])
                    if major >= 3 and minor >= 8:
                        return candidate, version
            except:
                continue
    
    return None, None

def main():
    """Main launcher function"""
    print_colored(f"{Symbols.ROCKET} Welcome to Toaripi SLM!", Colors.BLUE)
    print_colored("Setting up your environment automatically...\n", Colors.CYAN)
    
    # Check if we're in the right directory
    if not (Path("setup.py").exists() and Path("src/toaripi_slm").exists()):
        print_error("Not in the Toaripi SLM project directory!")
        print("Please run this script from the project root directory.")
        print("Expected to find: setup.py and src/toaripi_slm/")
        sys.exit(1)
    
    print_success("Found Toaripi SLM project!")
    
    # Step 1: Check Python
    print_step("Step 1: Checking Python installation")
    python_cmd, python_version = find_python()
    
    if not python_cmd:
        print_error("Python 3.8+ not found! Please install Python 3.8 or newer")
        sys.exit(1)
    
    print_success(f"Python {python_version} found ({python_cmd})")
    
    # Step 2: Setup virtual environment
    print_step("Step 2: Setting up virtual environment")
    venv_name = "toaripi_env"
    venv_path = Path(venv_name)
    
    if venv_path.exists():
        print_success(f"Virtual environment '{venv_name}' exists")
    else:
        print_step(f"Creating virtual environment '{venv_name}'...")
        success, _, _ = run_command([python_cmd, '-m', 'venv', venv_name])
        if not success:
            print_error("Failed to create virtual environment")
            sys.exit(1)
        print_success("Virtual environment created")
    
    # Step 3: Determine activation script
    print_step("Step 3: Preparing virtual environment activation")
    system = platform.system().lower()
    
    if system == "windows":
        activate_script = venv_path / "Scripts" / "activate.bat"
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:
        activate_script = venv_path / "bin" / "activate"
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    if not python_exe.exists():
        print_error(f"Virtual environment Python not found at {python_exe}")
        sys.exit(1)
    
    print_success("Virtual environment ready")
    
    # Step 4: Upgrade pip
    print_step("Step 4: Upgrading pip")
    success, _, _ = run_command([str(python_exe), '-m', 'pip', 'install', '--upgrade', 'pip'], quiet=True)
    if success:
        print_success("Pip upgraded")
    else:
        print_warning("Pip upgrade had issues, continuing...")
    
    # Step 5: Check and Install Dependencies
    print_step("Step 5: Checking and installing Toaripi SLM")
    
    install_status = check_toaripi_installation(python_exe)
    
    if install_status == "ready":
        print_success("Toaripi SLM is already installed and ready")
    elif install_status == "needs_update":
        print_info("Updating Toaripi SLM installation...")
        if not install_toaripi_slm(python_exe):
            print_error("Failed to update installation")
            sys.exit(1)
    else:  # needs_install
        print_info("Installing Toaripi SLM for the first time...")
        if not install_toaripi_slm(python_exe):
            print_error("Failed to install Toaripi SLM")
            sys.exit(1)
    
    # Step 6: Verify installation
    print_step("Step 6: Verifying installation")
    
    # Try to import and check CLI
    try:
        # Test if we can run the CLI
        if system == "windows":
            toaripi_exe = venv_path / "Scripts" / "toaripi.exe"
        else:
            toaripi_exe = venv_path / "bin" / "toaripi"
        
        if toaripi_exe.exists():
            success, _, _ = run_command([str(toaripi_exe), '--help'], quiet=True)
            if success:
                print_success("Toaripi CLI installed and ready!")
            else:
                print_warning("CLI installed but may have issues")
        else:
            # Fallback: try running as module
            success, _, _ = run_command([str(python_exe), '-m', 'toaripi_slm.cli', '--help'], quiet=True)
            if success:
                print_success("Toaripi CLI available as Python module")
            else:
                print_error("CLI verification failed")
                sys.exit(1)
    
    except Exception as e:
        print_error(f"Verification failed: {e}")
        sys.exit(1)
    
    # Step 7: Run initial diagnostics
    print_step("Step 7: Running system diagnostics")
    print()
    
    try:
        if toaripi_exe.exists():
            subprocess.run([str(toaripi_exe), 'doctor'], check=False)
        else:
            subprocess.run([str(python_exe), '-m', 'toaripi_slm.cli', 'doctor'], check=False)
    except:
        print_warning("Diagnostics had issues, but CLI should work")
    
    # Success message
    print()
    print_colored(f"{Symbols.ROCKET} Setup Complete! {Symbols.ROCKET}", Colors.GREEN)
    print_colored("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", Colors.CYAN)
    print_colored("You can now use these commands:", Colors.BLUE)
    print("  toaripi status     - Check system status")
    print("  toaripi doctor     - Run diagnostics")
    print("  toaripi train      - Start training")
    print("  toaripi test       - Test your model")
    print("  toaripi interact   - Chat with your model")
    print("  toaripi --help     - Show all commands")
    print()
    print_colored("Quick start: toaripi status", Colors.PURPLE)
    print_colored("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", Colors.CYAN)
    
    # Auto-run status
    print_colored(f"{Symbols.PYTHON} Running initial status check...", Colors.BLUE)
    print()
    
    try:
        if toaripi_exe.exists():
            subprocess.run([str(toaripi_exe), 'status'], check=False)
        else:
            subprocess.run([str(python_exe), '-m', 'toaripi_slm.cli', 'status'], check=False)
    except:
        print_warning("Status check failed, but you can run 'toaripi status' manually")
    
    print()
    print_colored(f"{Symbols.ROCKET} Ready to go! The Toaripi CLI is now active.", Colors.GREEN)
    print_colored(f"Virtual environment: {venv_name}", Colors.CYAN)
    
    if system == "windows":
        print_colored(f"To reactivate later: {venv_name}\\Scripts\\activate.bat", Colors.CYAN)
    else:
        print_colored(f"To reactivate later: source {venv_name}/bin/activate", Colors.CYAN)

if __name__ == "__main__":
    main()