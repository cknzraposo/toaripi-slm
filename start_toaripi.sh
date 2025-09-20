#!/bin/bash

# 🌟 Toaripi SLM Smart Launcher
# Just run: ./start_toaripi.sh or bash start_toaripi.sh

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Unicode symbols
CHECK="✅"
CROSS="❌"
WARN="⚠️"
ROCKET="🚀"
GEAR="⚙️"
PYTHON="🐍"

echo -e "${BLUE}${ROCKET} Welcome to Toaripi SLM!${NC}"
echo -e "${CYAN}Setting up your environment automatically...${NC}\n"

# Function to print step headers
print_step() {
    echo -e "${PURPLE}${GEAR} $1${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}${CHECK} $1${NC}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}${WARN} $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}${CROSS} $1${NC}"
}

# Function to show progress bar
show_progress_bar() {
    local duration=${1:-30}
    local message=${2:-"Installing dependencies"}
    
    echo -e "${CYAN}${GEAR} $message${NC}"
    
    local bar_length=40
    for ((i=1; i<=duration; i++)); do
        local percent=$((i * 100 / duration))
        local filled_length=$((bar_length * i / duration))
        local bar=""
        
        # Build progress bar
        for ((j=1; j<=filled_length; j++)); do
            bar+="█"
        done
        for ((j=filled_length+1; j<=bar_length; j++)); do
            bar+="░"
        done
        
        printf "\r${PURPLE}[%s] %3d%% (%ds)${NC}" "$bar" "$percent" "$i"
        sleep 1
    done
    echo ""
}

# Function to show spinner
show_spinner() {
    local message=${1:-"Processing"}
    local pid=$2
    
    local spinner_chars=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧')
    local i=0
    
    while kill -0 "$pid" 2>/dev/null; do
        printf "\r${CYAN}%s %s...${NC}" "${spinner_chars[i]}" "$message"
        i=$(( (i + 1) % ${#spinner_chars[@]} ))
        sleep 0.1
    done
    printf "\r%50s\r" ""  # Clear spinner line
}

# Function to run command with progress
run_with_progress() {
    local estimated_time=$1
    shift
    local command=("$@")
    
    if [[ $estimated_time -gt 10 ]]; then
        echo -e "${CYAN}ℹ️  Estimated time: ${estimated_time} seconds${NC}"
        
        # Run command in background and show spinner
        "${command[@]}" &>/dev/null &
        local cmd_pid=$!
        
        show_spinner "Installing" $cmd_pid
        
        # Wait for command to finish
        wait $cmd_pid
        return $?
    else
        # For quick operations, run normally
        "${command[@]}"
        return $?
    fi
}

# Check if we're in the right directory
if [[ ! -f "setup.py" ]] || [[ ! -d "src/toaripi_slm" ]]; then
    print_error "Not in the Toaripi SLM project directory!"
    echo "Please run this script from the project root directory."
    echo "Expected to find: setup.py and src/toaripi_slm/"
    exit 1
fi

print_success "Found Toaripi SLM project!"

# Step 1: Check Python
print_step "Step 1: Checking Python installation"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python ${PYTHON_VERSION} found"
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_success "Python ${PYTHON_VERSION} found"
    PYTHON_CMD="python"
else
    print_error "Python not found! Please install Python 3.8+"
    exit 1
fi

# Step 2: Check/Create Virtual Environment
print_step "Step 2: Setting up virtual environment"
VENV_NAME="toaripi_env"

if [[ -d "$VENV_NAME" ]]; then
    print_success "Virtual environment '$VENV_NAME' exists"
else
    print_step "Creating virtual environment '$VENV_NAME'..."
    $PYTHON_CMD -m venv $VENV_NAME
    print_success "Virtual environment created"
fi

# Step 3: Activate Virtual Environment
print_step "Step 3: Activating virtual environment"
source $VENV_NAME/bin/activate
print_success "Virtual environment activated"

# Step 4: Upgrade pip
print_step "Step 4: Upgrading pip"
pip install --upgrade pip --quiet
print_success "Pip upgraded"

# Function to check dependencies
check_dependencies() {
    print_step "Checking dependency status..."
    
    local deps=("torch:PyTorch" "transformers:Transformers" "datasets:Datasets" "accelerate:Accelerate" "peft:PEFT" "click:Click" "rich:Rich")
    local installed=()
    local missing=()
    
    for dep in "${deps[@]}"; do
        module="${dep%%:*}"
        name="${dep##*:}"
        
        if $PYTHON_CMD -c "import $module" 2>/dev/null; then
            installed+=("$name")
        else
            missing+=("$name")
        fi
    done
    
    if [ ${#installed[@]} -gt 0 ]; then
        print_success "Found existing dependencies: ${installed[*]}"
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        print_warning "Need to install: ${missing[*]}"
        return 1
    else
        return 0
    fi
}

# Function to check if Toaripi SLM is installed
check_toaripi_installation() {
    print_step "Checking existing installation..."
    
    # Check if package is installed
    if $PYTHON_CMD -c "import toaripi_slm" 2>/dev/null; then
        print_success "Toaripi SLM package found"
        
        # Check if CLI is functional
        if $PYTHON_CMD -c "from src.toaripi_slm.cli import cli" 2>/dev/null; then
            print_success "CLI is functional"
            echo "ready"
        else
            print_warning "CLI needs updating"
            echo "needs_update"
        fi
    else
        print_warning "Package not installed"
        echo "needs_install"
    fi
}

# Function to install Toaripi SLM
install_toaripi_slm() {
    print_step "Installing Toaripi SLM package..."
    
    # Check dependencies first
    local estimated_time=15  # Default for fast install
    if check_dependencies; then
        print_success "All major dependencies already present - installation should be fast"
        estimated_time=15
    else
        print_warning "Installing missing dependencies - this may take several minutes..."
        estimated_time=180  # 3 minutes for full dependency install
    fi
    
    # Try quiet install first with progress
    print_step "Starting installation process..."
    if run_with_progress $estimated_time pip install -e . --quiet; then
        print_success "Installation completed successfully"
        return 0
    fi
    
    # If quiet install failed, try verbose
    print_warning "Quiet installation had issues, trying with detailed output..."
    print_step "Retrying installation with detailed output..."
    
    # Show progress for verbose install
    local verbose_time=$((estimated_time + 30))
    if run_with_progress $verbose_time pip install -e .; then
        print_success "Installation completed successfully"
        return 0
    else
        print_error "Installation failed"
        echo "Troubleshooting suggestions:"
        echo "1. Check internet connection"
        echo "2. Try upgrading pip: python -m pip install --upgrade pip"
        echo "3. Check available disk space"
        echo "4. Try installing without cache: pip install -e . --no-cache-dir"
        return 1
    fi
}

# Step 5: Install Dependencies
print_step "Step 5: Checking and installing Toaripi SLM"

INSTALL_STATUS=$(check_toaripi_installation)

if [[ "$INSTALL_STATUS" == "ready" ]]; then
    print_success "Toaripi SLM is already installed and ready"
elif [[ "$INSTALL_STATUS" == "needs_update" ]]; then
    print_step "Updating Toaripi SLM installation..."
    if ! install_toaripi_slm; then
        print_error "Failed to update installation"
        exit 1
    fi
else
    print_step "Installing Toaripi SLM for the first time..."
    if ! install_toaripi_slm; then
        print_error "Failed to install Toaripi SLM"
        exit 1
    fi
fi

# Step 6: Verify Installation
print_step "Step 6: Verifying installation"
if command -v toaripi &> /dev/null; then
    print_success "Toaripi CLI installed and ready!"
else
    print_error "CLI installation failed"
    echo "Trying to diagnose the issue..."
    which python
    pip list | grep toaripi || echo "Toaripi package not found in pip list"
    exit 1
fi

# Step 7: Run System Check
print_step "Step 7: Running system diagnostics"
echo ""
toaripi doctor 2>/dev/null || {
    print_warning "Some system checks failed, but CLI is working"
    echo "Run 'toaripi doctor' for detailed diagnostics"
}

echo ""
echo -e "${GREEN}${ROCKET} Setup Complete! ${ROCKET}${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}You can now use these commands:${NC}"
echo -e "  ${YELLOW}toaripi status${NC}     - Check system status"
echo -e "  ${YELLOW}toaripi doctor${NC}     - Run diagnostics"
echo -e "  ${YELLOW}toaripi train${NC}      - Start training"
echo -e "  ${YELLOW}toaripi test${NC}       - Test your model"
echo -e "  ${YELLOW}toaripi interact${NC}   - Chat with your model"
echo -e "  ${YELLOW}toaripi --help${NC}     - Show all commands"
echo ""
echo -e "${PURPLE}Quick start: ${YELLOW}toaripi status${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Auto-start status check
echo -e "${BLUE}${PYTHON} Running initial status check...${NC}"
echo ""
toaripi status

echo ""
echo -e "${GREEN}${ROCKET} Ready to go! The Toaripi CLI is now active in this terminal.${NC}"
echo -e "${CYAN}Virtual environment: ${YELLOW}$VENV_NAME${NC}"
echo -e "${CYAN}To reactivate later: ${YELLOW}source $VENV_NAME/bin/activate${NC}"