# Run the modality study in Anaconda JupyterLab

Extract the code and records, images and annotations, and initial weights archives from [image benchmark v1.0.0](https://github.com/ahmedsedik123/uav-thermal-rgb-reproducibility/releases/tag/v1.0.0) into the same project directory.

In Anaconda Prompt:

```bat
cd /d C:\Research\UAV_Full_Project
Create_Environment_Anaconda.bat
Start_JupyterLab_Anaconda.bat
```

For a CPU-only environment, use `Create_Environment_Anaconda.bat CPU` in the second line. Select **Python (UAV All Modalities)**, open `UAV_Revision_Study.ipynb` and run cells in order. Check the data/status tables before training. The thermal and RGB experiments in the paper use separate image cohorts.

If using an existing compatible environment, install `requirements.txt` there and register that environment's kernel with `python -m ipykernel install --user --name uav-all-modalities --display-name "Python (UAV All Modalities)"`.

The paper reports 60 training runs: ten detectors, two image cohorts, and seeds 17, 42 and 93, with 100 epochs per run. The optional paired fusion workflow was not evaluated. The reported training used a CUDA workstation. CPU engineering checks do not substitute for the reported experiments.

The package can be moved before its first run. Preparation creates correct paths for that computer; old source YAML files are not used. If a completed experiment is moved or its recipe changes, use a new `STUDY_NAME` to preserve earlier results.
