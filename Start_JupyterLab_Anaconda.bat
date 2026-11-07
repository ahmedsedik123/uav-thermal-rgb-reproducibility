@echo off
setlocal
cd /d "%~dp0"
where conda >nul 2>nul
if errorlevel 1 (
  echo Open Anaconda Prompt, change to this code folder, and run this launcher again.
  echo See JUPYTERLAB_START_HERE.md for the initial environment setup.
  pause
  exit /b 1
)
call conda activate uav-all-modalities
if errorlevel 1 (
  echo Create the environment first: conda env create -f environment.yml
  echo See JUPYTERLAB_START_HERE.md.
  pause
  exit /b 1
)
python launch_jupyterlab.py
if errorlevel 1 (
  echo See JUPYTERLAB_START_HERE.md for environment and kernel setup.
  pause
  exit /b 1
)
endlocal
