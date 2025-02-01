@echo off
setlocal enabledelayedexpansion

:: Set current directory
cd /d "%~dp0"

:: Setup Python virtual environment
if not exist .env (
    echo Python virtual environment not found. Please run start_api_service.bat first.
    exit /b 1
)

:: Activate virtual environment
call "%~dp0.env\Scripts\activate.bat"

:: Start API service
echo Starting API service...

:: Set Python path for imports
echo [INFO] Setting up Python path...
set PYTHONPATH=%~dp0..;%PYTHONPATH%

python main.py

:: If error occurs
if errorlevel 1 (
    echo Failed to start API service
    exit /b 1
)

endlocal
