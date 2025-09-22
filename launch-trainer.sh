#!/bin/bash

# Toaripi SLM Trainer Launcher for Linux/macOS
# Educational Content Generation with Cultural Sensitivity

echo "==============================================="
echo "   Toaripi SLM Educational Content Trainer"
echo "==============================================="
echo
echo "Starting system validation..."
echo

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found"
    echo
    echo "The Toaripi SLM trainer requires Python 3.10 or newer."
    echo "Please install Python 3 using your package manager:"
    echo
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install python3.11"
    echo "  macOS:         brew install python@3.11"
    echo "  CentOS/RHEL:   sudo yum install python3.11"
    echo
    echo "After installation, restart this launcher."
    echo
    read -p "Press Enter to exit..."
    exit 1
fi

# Display Python version found
PYTHON_VERSION=$(python3 --version 2>&1)
echo "Found $PYTHON_VERSION"

# Change to project directory (in case launcher is run from elsewhere)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if we're in the right directory
if [ ! -d "src/toaripi_slm" ]; then
    echo "ERROR: Cannot find Toaripi SLM source code"
    echo
    echo "This launcher must be run from the toaripi-slm project directory."
    echo "Please ensure you're in the correct folder and try again."
    echo
    read -p "Press Enter to exit..."
    exit 1
fi

# Run launcher validation
echo "Running system validation..."
python3 launcher/launcher.py --platform unix
if [ $? -ne 0 ]; then
    echo
    echo "Launch failed. Please follow the guidance above to resolve issues."
    echo
    read -p "Press Enter to exit..."
    exit 1
fi

# Launch successful - show success message
echo
echo "✅ System validation passed!"
echo "Starting Toaripi SLM Trainer in interactive mode..."
echo

# Activate virtual environment and start trainer
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated."
    echo
else
    echo "Warning: Virtual environment not found, using system Python."
    echo
fi

# Start the trainer CLI in interactive mode
toaripi-slm train interactive --beginner
if [ $? -ne 0 ]; then
    echo
    echo "Training session ended with an error."
    echo "Please check the messages above for troubleshooting information."
    echo
else
    echo
    echo "Training session completed successfully!"
    echo
fi

echo "Press Enter to exit..."
read