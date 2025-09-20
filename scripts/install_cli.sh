#!/bin/bash
# Cross-platform installation and testing script for Toaripi SLM CLI

set -e  # Exit on any error

echo "🚀 Toaripi SLM CLI Installation and Testing"
echo "=========================================="

# Function to print colored output
print_status() {
    case $1 in
        "info") echo -e "\033[0;34mℹ️  $2\033[0m" ;;
        "success") echo -e "\033[0;32m✅ $2\033[0m" ;;
        "warning") echo -e "\033[0;33m⚠️  $2\033[0m" ;;
        "error") echo -e "\033[0;31m❌ $2\033[0m" ;;
    esac
}

# Detect platform
PLATFORM="unknown"
case "$OSTYPE" in
    linux*) PLATFORM="linux" ;;
    darwin*) PLATFORM="macos" ;;
    msys*|cygwin*|win*) PLATFORM="windows" ;;
esac

print_status "info" "Detected platform: $PLATFORM"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>/dev/null || python --version 2>/dev/null || echo "Not found")
print_status "info" "Python version: $PYTHON_VERSION"

if [[ ! $PYTHON_VERSION =~ Python\ 3\.(1[0-9]|[1-9][0-9]) ]]; then
    print_status "error" "Python 3.10+ required. Please upgrade Python."
    exit 1
fi

print_status "success" "Python version is compatible"

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_status "warning" "Not in a virtual environment. Consider creating one:"
    print_status "info" "  python -m venv toaripi_env"
    print_status "info" "  source toaripi_env/bin/activate  # Linux/macOS"
    print_status "info" "  toaripi_env\\Scripts\\activate.bat  # Windows"
    
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_status "success" "Running in virtual environment: $VIRTUAL_ENV"
fi

# Install the package
print_status "info" "Installing Toaripi SLM CLI..."

# Install in development mode
if pip install -e .; then
    print_status "success" "Package installed successfully"
else
    print_status "error" "Package installation failed"
    exit 1
fi

# Install dependencies
print_status "info" "Installing dependencies..."
if pip install -r requirements.txt; then
    print_status "success" "Dependencies installed successfully"
else
    print_status "error" "Dependency installation failed"
    exit 1
fi

# Test CLI availability
print_status "info" "Testing CLI availability..."

if command -v toaripi &> /dev/null; then
    print_status "success" "CLI command 'toaripi' is available"
else
    print_status "warning" "CLI command 'toaripi' not found in PATH"
    print_status "info" "Trying alternative method..."
    
    if python -m toaripi_slm.cli --help &> /dev/null; then
        print_status "success" "CLI available via 'python -m toaripi_slm.cli'"
        # Create alias function
        alias toaripi='python -m toaripi_slm.cli'
    else
        print_status "error" "CLI not accessible"
        exit 1
    fi
fi

# Run basic tests
print_status "info" "Running basic CLI tests..."

# Test help command
if toaripi --help > /dev/null 2>&1; then
    print_status "success" "Help command works"
else
    print_status "error" "Help command failed"
    exit 1
fi

# Test status command
print_status "info" "Testing status command..."
if toaripi status > /dev/null 2>&1; then
    print_status "success" "Status command works"
else
    print_status "warning" "Status command had issues (may be expected)"
fi

# Test doctor command
print_status "info" "Testing doctor command..."
if toaripi doctor > /dev/null 2>&1; then
    print_status "success" "Doctor command works"
else
    print_status "warning" "Doctor command had issues (may be expected)"
fi

# Create sample directory structure
print_status "info" "Creating sample directory structure..."

mkdir -p data/{raw,processed,samples}
mkdir -p models/{cache,hf,gguf}
mkdir -p configs/{training,data}
mkdir -p training_sessions
mkdir -p chat_sessions

print_status "success" "Directory structure created"

# Create sample configuration if it doesn't exist
if [[ ! -f "configs/training/test_config.yaml" ]]; then
    print_status "info" "Creating sample configuration..."
    
    cat > configs/training/test_config.yaml << 'EOF'
# Test configuration for Toaripi SLM
model:
  name: "microsoft/DialoGPT-medium"
  cache_dir: "./models/cache"

training:
  epochs: 1
  learning_rate: 2e-5
  batch_size: 2
  gradient_accumulation_steps: 2

lora:
  enabled: true
  r: 8
  lora_alpha: 16

output:
  checkpoint_dir: "./models/checkpoints"

logging:
  use_wandb: false
EOF
    
    print_status "success" "Sample configuration created"
fi

# Test dry run
print_status "info" "Testing training dry run..."
if toaripi train --dry-run --config configs/training/test_config.yaml > /dev/null 2>&1; then
    print_status "success" "Training dry run works"
else
    print_status "warning" "Training dry run failed (expected without data)"
fi

# Final status check
print_status "info" "Running final system check..."
toaripi doctor --detailed

echo
print_status "success" "Installation and testing completed!"
echo
echo "🎉 Next steps:"
echo "  1. Check system status: toaripi status --detailed"
echo "  2. Prepare your data: toaripi-prepare-data"
echo "  3. Train a model: toaripi train --interactive"
echo "  4. Test the model: toaripi test"
echo "  5. Interactive chat: toaripi interact"
echo
echo "📚 For more information:"
echo "  - CLI Guide: docs/CLI_GUIDE.md"
echo "  - Help: toaripi --help"
echo "  - System diagnostics: toaripi doctor"
echo

# Platform-specific notes
case $PLATFORM in
    "windows")
        echo "🪟 Windows-specific notes:"
        echo "  - Use PowerShell or Command Prompt"
        echo "  - Path separators use backslashes"
        echo "  - Consider using WSL for better compatibility"
        ;;
    "linux")
        echo "🐧 Linux-specific notes:"
        echo "  - All features should work normally"
        echo "  - GPU support requires CUDA installation"
        echo "  - Use 'sudo' for system-wide installation if needed"
        ;;
    "macos")
        echo "🍎 macOS-specific notes:"
        echo "  - Use Homebrew for additional dependencies"
        echo "  - GPU training not supported (use CPU or cloud)"
        echo "  - Xcode command line tools may be required"
        ;;
esac