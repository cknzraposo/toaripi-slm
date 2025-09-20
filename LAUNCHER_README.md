# 🌟 Toaripi SLM Launcher Scripts

This directory contains multiple launcher scripts to make getting started with Toaripi SLM as easy as possible. Choose the one that works best for your environment:

## Quick Start Options

### 🐍 Universal Python Launcher (Recommended)
```bash
python start_toaripi.py
```
- **Platform**: Windows, Linux, macOS, WSL
- **Requirements**: Python 3.8+
- **Best for**: Everyone, most reliable option

### 🐧 Linux/WSL/macOS Shell Script
```bash
./start_toaripi.sh
```
- **Platform**: Linux, macOS, WSL
- **Requirements**: Bash shell
- **Best for**: Unix-like systems, fastest startup

### 🪟 Windows Batch Script
```batch
start_toaripi.bat
```
- **Platform**: Windows Command Prompt
- **Requirements**: Windows with Python
- **Best for**: Windows users who prefer .bat files

## What These Scripts Do

All launcher scripts perform the same automated setup:

1. **✅ Verify Environment** - Check you're in the right directory
2. **🐍 Find Python** - Locate Python 3.8+ installation  
3. **📦 Setup Virtual Environment** - Create/activate `toaripi_env`
4. **⬆️ Upgrade Pip** - Ensure latest package installer
5. **🔧 Install Toaripi SLM** - Install in development mode
6. **🩺 Run Diagnostics** - Verify everything works
7. **🚀 Launch CLI** - Show available commands and run status

## First Time Setup

1. **Navigate to project directory**:
   ```bash
   cd /path/to/toaripi-slm
   ```

2. **Choose your launcher**:
   ```bash
   # Option 1: Universal (recommended)
   python start_toaripi.py
   
   # Option 2: Linux/Mac
   ./start_toaripi.sh
   
   # Option 3: Windows
   start_toaripi.bat
   ```

3. **Follow the guided setup** - The script will handle everything automatically!

## After Setup

Once setup is complete, you'll have access to these commands:

```bash
toaripi status      # Check system status
toaripi doctor      # Run comprehensive diagnostics  
toaripi train       # Start interactive training
toaripi test        # Evaluate your models
toaripi interact    # Chat with trained models
toaripi --help      # Show all available commands
```

## Reactivating Later

If you close your terminal and want to use Toaripi SLM again:

**Linux/Mac/WSL:**
```bash
cd /path/to/toaripi-slm
source toaripi_env/bin/activate
toaripi status
```

**Windows:**
```batch
cd C:\path\to\toaripi-slm
toaripi_env\Scripts\activate.bat
toaripi status
```

## Troubleshooting

### Python Not Found
- Install Python 3.8+ from [python.org](https://python.org)
- On Ubuntu/Debian: `sudo apt install python3 python3-venv`
- On macOS: `brew install python3`

### Permission Denied (Linux/Mac)
```bash
chmod +x start_toaripi.sh
./start_toaripi.sh
```

### Virtual Environment Issues
Delete the environment and start fresh:
```bash
rm -rf toaripi_env
python start_toaripi.py
```

### CLI Not Found After Setup
Manually activate and check:
```bash
source toaripi_env/bin/activate  # Linux/Mac
# or
toaripi_env\Scripts\activate.bat  # Windows

pip list | grep toaripi
which toaripi  # Linux/Mac
where toaripi  # Windows
```

## Getting Help

- Run `toaripi doctor` for comprehensive system diagnostics
- Check `toaripi --help` for all available commands
- See `/docs/setup/` for detailed setup guides
- Review `/specs/` for project specifications

---

**🎯 Goal**: Get you from zero to training Toaripi language models in under 5 minutes!

**🚀 Quick Start**: Just run `python start_toaripi.py` and follow the prompts.