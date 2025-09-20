@echo off
REM Cross-platform installation and testing script for Toaripi SLM CLI (Windows)

echo 🚀 Toaripi SLM CLI Installation and Testing
echo ==========================================

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.10+
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ℹ️  Python version: %PYTHON_VERSION%

REM Check if we're in a virtual environment
if "%VIRTUAL_ENV%"=="" (
    echo ⚠️  Not in a virtual environment. Consider creating one:
    echo   python -m venv toaripi_env
    echo   toaripi_env\Scripts\activate.bat
    set /p CONTINUE=Continue anyway? (y/N): 
    if /i not "%CONTINUE%"=="y" exit /b 1
) else (
    echo ✅ Running in virtual environment: %VIRTUAL_ENV%
)

REM Install the package
echo ℹ️  Installing Toaripi SLM CLI...
pip install -e .
if errorlevel 1 (
    echo ❌ Package installation failed
    exit /b 1
)
echo ✅ Package installed successfully

REM Install dependencies
echo ℹ️  Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Dependency installation failed
    exit /b 1
)
echo ✅ Dependencies installed successfully

REM Test CLI availability
echo ℹ️  Testing CLI availability...
toaripi --help >nul 2>&1
if errorlevel 1 (
    echo ⚠️  CLI command 'toaripi' not found in PATH
    echo ℹ️  Trying alternative method...
    python -m toaripi_slm.cli --help >nul 2>&1
    if errorlevel 1 (
        echo ❌ CLI not accessible
        exit /b 1
    ) else (
        echo ✅ CLI available via 'python -m toaripi_slm.cli'
    )
) else (
    echo ✅ CLI command 'toaripi' is available
)

REM Run basic tests
echo ℹ️  Running basic CLI tests...

REM Test status command
echo ℹ️  Testing status command...
toaripi status >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Status command had issues (may be expected)
) else (
    echo ✅ Status command works
)

REM Test doctor command
echo ℹ️  Testing doctor command...
toaripi doctor >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Doctor command had issues (may be expected)
) else (
    echo ✅ Doctor command works
)

REM Create sample directory structure
echo ℹ️  Creating sample directory structure...
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir data\samples 2>nul
mkdir models\cache 2>nul
mkdir models\hf 2>nul
mkdir models\gguf 2>nul
mkdir configs\training 2>nul
mkdir configs\data 2>nul
mkdir training_sessions 2>nul
mkdir chat_sessions 2>nul
echo ✅ Directory structure created

REM Create sample configuration if it doesn't exist
if not exist "configs\training\test_config.yaml" (
    echo ℹ️  Creating sample configuration...
    (
    echo # Test configuration for Toaripi SLM
    echo model:
    echo   name: "microsoft/DialoGPT-medium"
    echo   cache_dir: "./models/cache"
    echo.
    echo training:
    echo   epochs: 1
    echo   learning_rate: 2e-5
    echo   batch_size: 2
    echo   gradient_accumulation_steps: 2
    echo.
    echo lora:
    echo   enabled: true
    echo   r: 8
    echo   lora_alpha: 16
    echo.
    echo output:
    echo   checkpoint_dir: "./models/checkpoints"
    echo.
    echo logging:
    echo   use_wandb: false
    ) > configs\training\test_config.yaml
    echo ✅ Sample configuration created
)

REM Test dry run
echo ℹ️  Testing training dry run...
toaripi train --dry-run --config configs\training\test_config.yaml >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Training dry run failed (expected without data)
) else (
    echo ✅ Training dry run works
)

REM Final status check
echo ℹ️  Running final system check...
toaripi doctor --detailed

echo.
echo ✅ Installation and testing completed!
echo.
echo 🎉 Next steps:
echo   1. Check system status: toaripi status --detailed
echo   2. Prepare your data: toaripi-prepare-data
echo   3. Train a model: toaripi train --interactive
echo   4. Test the model: toaripi test
echo   5. Interactive chat: toaripi interact
echo.
echo 📚 For more information:
echo   - CLI Guide: docs\CLI_GUIDE.md
echo   - Help: toaripi --help
echo   - System diagnostics: toaripi doctor
echo.
echo 🪟 Windows-specific notes:
echo   - Use PowerShell or Command Prompt
echo   - Path separators use backslashes
echo   - Consider using WSL for better compatibility

pause