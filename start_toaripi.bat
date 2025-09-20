@echo off
REM 🌟 Toaripi SLM Smart Launcher for Windows
REM Just run: start_toaripi.bat

echo 🚀 Welcome to Toaripi SLM!
echo Setting up your environment automatically...
echo.

REM Check if we're in the right directory
if not exist "setup.py" (
    echo ❌ Not in the Toaripi SLM project directory!
    echo Please run this script from the project root directory.
    echo Expected to find: setup.py
    pause
    exit /b 1
)

if not exist "src\toaripi_slm" (
    echo ❌ Not in the Toaripi SLM project directory!
    echo Please run this script from the project root directory.
    echo Expected to find: src\toaripi_slm\
    pause
    exit /b 1
)

echo ✅ Found Toaripi SLM project!

REM Step 1: Check Python
echo ⚙️ Step 1: Checking Python installation
python --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
    echo ✅ Python !PYTHON_VERSION! found
    set PYTHON_CMD=python
) else (
    py --version >nul 2>&1
    if %errorlevel% equ 0 (
        for /f "tokens=2" %%v in ('py --version 2^>^&1') do set PYTHON_VERSION=%%v
        echo ✅ Python !PYTHON_VERSION! found
        set PYTHON_CMD=py
    ) else (
        echo ❌ Python not found! Please install Python 3.8+
        pause
        exit /b 1
    )
)

REM Step 2: Check/Create Virtual Environment
echo ⚙️ Step 2: Setting up virtual environment
set VENV_NAME=toaripi_env

if exist "%VENV_NAME%" (
    echo ✅ Virtual environment '%VENV_NAME%' exists
) else (
    echo ⚙️ Creating virtual environment '%VENV_NAME%'...
    %PYTHON_CMD% -m venv %VENV_NAME%
    echo ✅ Virtual environment created
)

REM Step 3: Activate Virtual Environment
echo ⚙️ Step 3: Activating virtual environment
call %VENV_NAME%\Scripts\activate.bat
echo ✅ Virtual environment activated

REM Step 4: Upgrade pip
echo ⚙️ Step 4: Upgrading pip
pip install --upgrade pip --quiet
echo ✅ Pip upgraded

REM Step 5: Check and Install Dependencies
echo ⚙️ Step 5: Checking and installing Toaripi SLM

REM Check if package is already installed
echo ⚙️ Checking existing installation...
%PYTHON_CMD% -c "import toaripi_slm" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Toaripi SLM package found
    
    REM Check if CLI is functional
    %PYTHON_CMD% -c "from src.toaripi_slm.cli import cli" >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ CLI is functional - Toaripi SLM is ready
        goto :SkipInstall
    ) else (
        echo ⚠️ CLI needs updating
        goto :InstallUpdate
    )
) else (
    echo ⚠️ Package not installed
    goto :InstallNew
)

:InstallNew
echo ⚙️ Installing Toaripi SLM for the first time...
goto :CheckDeps

:InstallUpdate
echo ⚙️ Updating Toaripi SLM installation...
goto :CheckDeps

:CheckDeps
echo ⚙️ Checking dependency status...

REM Check for major dependencies
set DEPS_PRESENT=0
set ESTIMATED_TIME=15

%PYTHON_CMD% -c "import torch" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ PyTorch found
    set /a DEPS_PRESENT+=1
)

%PYTHON_CMD% -c "import transformers" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Transformers found
    set /a DEPS_PRESENT+=1
)

%PYTHON_CMD% -c "import datasets" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Datasets found
    set /a DEPS_PRESENT+=1
)

if %DEPS_PRESENT% gtr 0 (
    echo ✅ Some dependencies already present - installation should be faster
    set ESTIMATED_TIME=15
) else (
    echo ⚠️ Installing missing dependencies - this may take several minutes...
    set ESTIMATED_TIME=180
)

goto :DoInstall

:DoInstall
echo ⚙️ Installing Toaripi SLM package...
echo ℹ️ Estimated time: %ESTIMATED_TIME% seconds

REM Show progress indicator
echo ⚙️ Starting installation process...
for /l %%i in (1,1,5) do (
    echo   Progress: %%i/5 - Installing...
    timeout /t 1 >nul 2>&1
)

REM Try quiet install first
pip install -e . --quiet >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Installation completed successfully
    goto :SkipInstall
)

REM If quiet install failed, try verbose
echo ⚠️ Quiet installation had issues, trying with detailed output...
echo ⚙️ Retrying installation with detailed output...

REM Show extended progress for verbose install
for /l %%i in (1,1,10) do (
    echo   Retry Progress: %%i/10 - Installing with verbose output...
    timeout /t 2 >nul 2>&1
)

pip install -e .
if %errorlevel% equ 0 (
    echo ✅ Installation completed successfully
) else (
    echo ❌ Installation failed
    echo Troubleshooting suggestions:
    echo 1. Check internet connection
    echo 2. Try upgrading pip: python -m pip install --upgrade pip
    echo 3. Check available disk space
    echo 4. Try installing without cache: pip install -e . --no-cache-dir
    pause
    exit /b 1
)

:SkipInstall

REM Step 6: Verify Installation
echo ⚙️ Step 6: Verifying installation
toaripi --help >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Toaripi CLI installed and ready!
) else (
    echo ❌ CLI installation failed
    echo Trying to diagnose the issue...
    where python
    pip list | findstr toaripi
    pause
    exit /b 1
)

REM Step 7: Run System Check
echo ⚙️ Step 7: Running system diagnostics
echo.
toaripi doctor 2>nul
if %errorlevel% neq 0 (
    echo ⚠️ Some system checks failed, but CLI is working
    echo Run 'toaripi doctor' for detailed diagnostics
)

echo.
echo 🚀 Setup Complete! 🚀
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo You can now use these commands:
echo   toaripi status     - Check system status
echo   toaripi doctor     - Run diagnostics
echo   toaripi train      - Start training
echo   toaripi test       - Test your model
echo   toaripi interact   - Chat with your model
echo   toaripi --help     - Show all commands
echo.
echo Quick start: toaripi status
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REM Auto-start status check
echo 🐍 Running initial status check...
echo.
toaripi status

echo.
echo 🚀 Ready to go! The Toaripi CLI is now active in this terminal.
echo Virtual environment: %VENV_NAME%
echo To reactivate later: %VENV_NAME%\Scripts\activate.bat
pause