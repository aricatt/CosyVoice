@echo off
setlocal enabledelayedexpansion

:: Set title
title CosyVoice API Service

:: Set color
color 0A

echo ====================================
echo    CosyVoice API Service Launcher
echo ====================================
echo.

:: Check Conda installation
echo [INFO] Checking conda installation...
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Conda not found. Please install Miniconda or Anaconda.
    pause
    exit /b 1
)

:: Check CUDA environment
echo [INFO] Checking CUDA environment...
nvidia-smi > nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] NVIDIA GPU not found. Service will run on CPU mode.
    echo [WARNING] Speech generation may be slower without GPU acceleration.
    timeout /t 3 > nul
) else (
    echo [INFO] NVIDIA GPU detected.
)

:: Initialize conda for batch script
echo [INFO] Initializing conda environment...
call conda deactivate
call conda activate base

:: Set environment path
set ENV_DIR=%~dp0.env
set PYTHONPATH=%~dp0..;%~dp0..\third_party\Matcha-TTS

:: Check and create local conda environment
if not exist "%ENV_DIR%" (
    echo [INFO] Creating local conda environment in .env...
    echo [INFO] This may take a few minutes...
    call conda create -y -p "%ENV_DIR%" python=3.10
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create conda environment.
        echo [ERROR] Please check your conda installation and internet connection.
        pause
        exit /b 1
    )
    echo [SUCCESS] Conda environment created successfully.
)

:: Activate local conda environment
echo [INFO] Activating conda environment...
call conda activate "%ENV_DIR%"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate conda environment.
    echo [ERROR] Try removing the .env directory and run this script again.
    pause
    exit /b 1
)

:: Install conda dependencies
echo [INFO] Installing conda dependencies...
call conda install -y -c conda-forge pynini==2.1.5
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install conda dependencies.
    echo [ERROR] Please check your internet connection and try again.
    pause
    exit /b 1
)

:: Install pip dependencies
echo [INFO] Installing API dependencies...
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install API dependencies.
    echo [ERROR] Please check your internet connection and try again.
    pause
    exit /b 1
)

echo [INFO] Installing CosyVoice dependencies...
pip install -r ..\requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install CosyVoice dependencies.
    echo [ERROR] Please check your internet connection and try again.
    pause
    exit /b 1
)

:: Check model files
if not exist "..\pretrained_models\CosyVoice2-0.5B" (
    echo [WARNING] Model files not found in ..\pretrained_models\CosyVoice2-0.5B
    echo [WARNING] Please make sure you have downloaded the model files.
    echo [WARNING] You can download them from ModelScope or follow the instructions in README.
    echo [WARNING] Press any key to continue anyway...
    pause > nul
)

:: Set Python path for imports
echo [INFO] Setting up Python path...
set PYTHONPATH=%~dp0..;%PYTHONPATH%

:: Start service
echo.
echo [SUCCESS] Environment setup completed.
echo [INFO] Starting API service...
echo [INFO] API documentation will be available at http://localhost:8507/docs
echo [INFO] Press Ctrl+C to stop the service
echo.

python main.py

:: Wait for user confirmation if service stops
echo.
echo [INFO] Service stopped.
pause

:: Deactivate conda environment
call conda deactivate
