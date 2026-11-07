@echo off
setlocal
cd /d "%~dp0"
where conda >nul 2>nul
if errorlevel 1 (
  echo Open Anaconda Prompt and run this file from the extracted project folder.
  exit /b 1
)
call conda run -n uav-all-modalities python -c "import sys" >nul 2>nul
if errorlevel 1 (
  call conda env create -f environment.yml
  if errorlevel 1 exit /b 1
)
call conda activate uav-all-modalities
if errorlevel 1 exit /b 1
set "UAV_TORCH_INDEX=https://download.pytorch.org/whl/cu118"
if /I "%~1"=="CPU" set "UAV_TORCH_INDEX=https://download.pytorch.org/whl/cpu"
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url %UAV_TORCH_INDEX%
if errorlevel 1 exit /b 1
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
python -m ipykernel install --user --name uav-all-modalities --display-name "Python (UAV All Modalities)"
if errorlevel 1 exit /b 1
python launch_jupyterlab.py --check
if errorlevel 1 exit /b 1
echo Environment ready. Run Start_JupyterLab_Anaconda.bat
endlocal
