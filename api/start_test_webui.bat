@echo off
setlocal enabledelayedexpansion

:: Set title
title CosyVoice API Test WebUI

:: Set color
color 0A

echo ====================================
echo    CosyVoice API Test WebUI
echo ====================================
echo.

:: Check if .env exists
set ENV_DIR=%~dp0.env
if not exist "%ENV_DIR%" (
    echo [ERROR] Conda environment not found in .env directory.
    echo [ERROR] Please run start_api_service.bat first to set up the environment.
    pause
    exit /b 1
)

:: Check if API is running
echo [INFO] Checking if API service is running...
curl -s http://localhost:8507 > nul
if %errorlevel% neq 0 (
    echo [ERROR] API service is not running on port 8507.
    echo [ERROR] Please start the API service first using start_api_service.bat
    pause
    exit /b 1
)

:: Set Python path for imports
echo [INFO] Setting up Python path...
set PYTHONPATH=%~dp0..;%PYTHONPATH%

:: Activate conda environment
echo [INFO] Activating conda environment...
call conda activate "%ENV_DIR%"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate conda environment.
    echo [ERROR] Try running start_api_service.bat to fix the environment.
    pause
    exit /b 1
)

:: Start the test WebUI
echo.
echo [SUCCESS] Environment activated successfully.
echo [INFO] Starting API Test WebUI...
echo [INFO] Web interface will be available at http://localhost:8508
echo [INFO] Press Ctrl+C to stop the service
echo.

python api_test_webui.py

:: Wait for user confirmation if service stops
echo.
echo [INFO] Service stopped.
pause

:: Deactivate conda environment
call conda deactivate
