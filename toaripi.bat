@echo off
REM Toaripi SLM CLI Windows Helper Script
REM This script provides enhanced Windows support for the Toaripi CLI

setlocal enabledelayedexpansion

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

REM Check if toaripi command is available
toaripi --version >nul 2>&1
if errorlevel 1 (
    echo Error: toaripi command not found
    echo Please install the package with: pip install -e .
    pause
    exit /b 1
)

REM If no arguments provided, show interactive menu
if "%~1"=="" (
    goto :interactive_menu
)

REM Otherwise, pass all arguments to toaripi
toaripi %*
goto :end

:interactive_menu
cls
echo.
echo ================================================================
echo             🌴 Toaripi SLM - Windows Command Center
echo ================================================================
echo.
echo Quick Actions:
echo   1. Check system status
echo   2. Setup project (guided)
echo   3. Train model (guided)
echo   4. Test model (guided)
echo   5. Interactive chat
echo   6. Manage models
echo   7. Troubleshoot issues
echo   8. Open command prompt
echo   9. Exit
echo.
set /p choice="Select an option (1-9): "

if "%choice%"=="1" (
    toaripi --status
    goto :pause_and_menu
)
if "%choice%"=="2" (
    toaripi setup --guided
    goto :pause_and_menu
)
if "%choice%"=="3" (
    toaripi train --guided
    goto :pause_and_menu
)
if "%choice%"=="4" (
    toaripi test --guided
    goto :pause_and_menu
)
if "%choice%"=="5" (
    toaripi interact --guided
    goto :pause_and_menu
)
if "%choice%"=="6" (
    toaripi models
    echo.
    echo Available model commands:
    echo   toaripi models list
    echo   toaripi models info MODEL_NAME
    echo   toaripi models convert MODEL_NAME --to-gguf
    goto :pause_and_menu
)
if "%choice%"=="7" (
    toaripi troubleshoot --report
    goto :pause_and_menu
)
if "%choice%"=="8" (
    echo.
    echo Starting command prompt. Type 'toaripi --help' for available commands.
    echo Type 'exit' to return to Windows.
    echo.
    cmd
    goto :interactive_menu
)
if "%choice%"=="9" (
    goto :end
)

echo Invalid choice. Please select 1-9.
goto :pause_and_menu

:pause_and_menu
echo.
pause
goto :interactive_menu

:end
endlocal