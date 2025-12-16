@echo off
cd /d "%~dp0"

echo ==============================================
echo   Student Success Dashboard Launcher
echo ==============================================

REM Check if python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not found. Please install Python and add it to your PATH.
    pause
    exit /b
)

echo.
echo [1/2] Checking dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Warning: Some dependencies might have failed to install. attempting to proceed...
)

echo.
echo [2/2] Launching Dashboard...
echo.
python -m streamlit run predictive_model/dashboard.py

if %errorlevel% neq 0 (
    echo.
    echo The application closed with an error. 
    pause
)
