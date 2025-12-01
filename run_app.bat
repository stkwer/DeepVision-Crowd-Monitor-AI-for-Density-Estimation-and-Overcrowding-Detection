@echo off
REM People Counter AI - Windows Launcher
REM Quick start script for the Streamlit application

setlocal enabledelayedexpansion

cls
echo ============================================================
echo  PEOPLE COUNTER AI - PRESENTATION MODE
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check requirements.txt
if not exist requirements.txt (
    echo Error: requirements.txt not found
    pause
    exit /b 1
)

echo Installing dependencies...
python -m pip install -r requirements.txt -q
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Launching People Counter AI...
echo.
echo Web App URL: http://localhost:8501
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

python -m streamlit run streamlit\app.py --logger.level=info

pause
