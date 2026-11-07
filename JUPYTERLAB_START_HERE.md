# Run the modality study in Anaconda JupyterLab

Extract the complete project, including `modalities`, `weights`, `uav_review` and `third_party`.

In Anaconda Prompt:

```bat
cd /d C:\Research\UAV_Full_Project
Create_Environment_Anaconda.bat
Start_JupyterLab_Anaconda.bat
```

For a CPU-only environment, use `Create_Environment_Anaconda.bat CPU` in the second line. Select **Python (UAV All Modalities)**, open `UAV_Revision_Study.ipynb` and run cells in order. Check the data/status tables before training. RGB and thermal can run now; fusion requires real aligned pairs described in `README.md`.

If using an existing compatible environment, install `requirements.txt` there and register that environment's kernel with `python -m ipykernel install --user --name uav-all-modalities --display-name "Python (UAV All Modalities)"`.

The notebook default schedules 60 primary jobs with the available two unmatched datasets. A valid paired three-modality holdout schedules 90 jobs. The full study is intended for the CUDA workstation. CPU engineering checks in `validation` do not substitute for the primary experiments.

The package can be moved before its first run. Preparation creates correct paths for that computer; old source YAML files are not used. If a completed experiment is moved or its recipe changes, use a new `STUDY_NAME` to preserve earlier results.
