@echo off
title Toaripi SLM Trainer Launcher

echo ===============================================
echo    Toaripi SLM Educational Content Trainer
echo ===============================================
echo.
echo Starting system validation...
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo.
    echo The Toaripi SLM trainer requires Python 3.10 or newer.
    echo Please install Python from: https://python.org/downloads/
    echo.
    echo After installation, restart this launcher.
    echo.
    pause
    exit /b 1
)

:: Display Python version found
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%

:: Change to project directory (in case launcher is run from elsewhere)
cd /d "%~dp0"

:: Check if we're in the right directory
if not exist "src\toaripi_slm" (
    echo ERROR: Cannot find Toaripi SLM source code
    echo.
    echo This launcher must be run from the toaripi-slm project directory.
    echo Please ensure you're in the correct folder and try again.
    echo.
    pause
    exit /b 1
)

:: Run launcher validation
echo Running system validation...
python launcher\launcher.py --platform windows
if errorlevel 1 (
    echo.
    echo Launch failed. Please follow the guidance above to resolve issues.
    echo.
    pause
    exit /b 1
)

:: Launch successful - show success message
echo.
echo ✅ System validation passed!
echo Starting Toaripi SLM Trainer in interactive mode...
echo.

:: Activate virtual environment and start trainer
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo Virtual environment activated.
    echo.
) else (
    echo Warning: Virtual environment not found, using system Python.
    echo.
)

:: Start the trainer CLI in interactive mode
toaripi-slm train interactive --beginner
if errorlevel 1 (
    echo.
    echo Training session ended with an error.
    echo Please check the messages above for troubleshooting information.
    echo.
) else (
    echo.
    echo Training session completed successfully!
    echo.
)

echo Press any key to exit...
pause >nul