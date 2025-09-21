#!/bin/bash

# 🌟 Toaripi SLM Alias Setup
# Source this file to add convenient aliases for Toaripi SLM
# Usage: source setup_aliases.sh

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}🌟 Setting up Toaripi SLM aliases...${NC}"

# Get the current directory (project root)
TOARIPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$TOARIPI_ROOT/toaripi_env"

# Function to activate Toaripi environment and run commands
toaripi() {
    if [[ -d "$VENV_PATH" ]]; then
        # Activate the virtual environment in a subshell to avoid affecting current shell
        (source "$VENV_PATH/bin/activate" && command toaripi "$@")
    else
        echo -e "${RED}❌ Toaripi virtual environment not found at: $VENV_PATH${NC}"
        echo "Run the setup script first: ./start_toaripi.sh"
        return 1
    fi
}

# Create convenient aliases
alias start-toaripi="cd '$TOARIPI_ROOT' && ./start_toaripi.sh"
alias toaripi-status="toaripi status"
alias toaripi-train="toaripi train"
alias toaripi-test="toaripi test"
alias toaripi-chat="toaripi interact"
alias toaripi-doctor="toaripi doctor"

echo -e "${GREEN}✅ Toaripi SLM aliases configured!${NC}"
echo -e "${CYAN}Available aliases:${NC}"
echo -e "  ${PURPLE}toaripi${NC}        - Run any Toaripi command"
echo -e "  ${PURPLE}start-toaripi${NC}  - Run the setup script"
echo -e "  ${PURPLE}toaripi-status${NC} - Check system status"
echo -e "  ${PURPLE}toaripi-train${NC}  - Start training"
echo -e "  ${PURPLE}toaripi-test${NC}   - Test models"
echo -e "  ${PURPLE}toaripi-chat${NC}   - Interactive chat"
echo -e "  ${PURPLE}toaripi-doctor${NC} - System diagnostics"
echo ""
echo -e "${BLUE}Quick start: ${PURPLE}toaripi-status${NC}"
echo -e "${CYAN}Virtual environment: ${PURPLE}$VENV_PATH${NC}"