@echo off
setlocal
cd /d "%~dp0"
where conda >nul 2>nul
if errorlevel 1 (
  echo Open Anaconda Prompt, change to this folder, and run this launcher again.
  echo See ANACONDA_START_HERE.md for the initial environment setup.
  pause
  exit /b 1
)
call conda activate uav-all-modalities
if errorlevel 1 (
  echo Create the environment first: conda env create -f environment.yml
  echo See ANACONDA_START_HERE.md.
  pause
  exit /b 1
)
python -m notebook "UAV_Revision_Study.ipynb"
endlocal
