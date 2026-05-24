@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "HUMAN_COLLECT_PASSWORD=123"
set "HUMAN_COLLECT_POLICY=td3"
set "CONDA_ENV_NAME=ppo"

echo Starting human collection web system...
echo Project folder: %CD%
echo Access password: %HUMAN_COLLECT_PASSWORD%
echo.

if exist "D:\Anaconda\condabin\conda.bat" (
    call "D:\Anaconda\condabin\conda.bat" activate "%CONDA_ENV_NAME%"
) else if exist "D:\Anaconda\Scripts\activate.bat" (
    call "D:\Anaconda\Scripts\activate.bat" "%CONDA_ENV_NAME%"
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%CONDA_ENV_NAME%"
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%CONDA_ENV_NAME%"
) else if exist "%ProgramData%\anaconda3\Scripts\activate.bat" (
    call "%ProgramData%\anaconda3\Scripts\activate.bat" "%CONDA_ENV_NAME%"
) else if exist "%ProgramData%\miniconda3\Scripts\activate.bat" (
    call "%ProgramData%\miniconda3\Scripts\activate.bat" "%CONDA_ENV_NAME%"
) else if exist "%LOCALAPPDATA%\anaconda3\Scripts\activate.bat" (
    call "%LOCALAPPDATA%\anaconda3\Scripts\activate.bat" "%CONDA_ENV_NAME%"
) else if exist "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat" (
    call "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat" "%CONDA_ENV_NAME%"
) else if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
    call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate "%CONDA_ENV_NAME%"
) else if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
    call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate "%CONDA_ENV_NAME%"
) else (
    echo Conda activation script was not found.
    echo Please run this file from Anaconda Prompt, or edit CONDA_ENV_NAME / conda path in this script.
    pause
    exit /b 1
)

if errorlevel 1 (
    echo Failed to activate conda environment "%CONDA_ENV_NAME%". Please check whether the ppo environment exists.
    pause
    exit /b 1
)

python real_System_remake\human_collect_web.py

echo.
echo Human collection web system has stopped.
pause
