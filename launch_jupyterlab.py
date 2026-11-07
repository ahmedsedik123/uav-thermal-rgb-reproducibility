"""Open the local study in JupyterLab using this Python environment."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="Launch the full local UAV study in JupyterLab.")
    parser.add_argument("--check", action="store_true", help="Check launch requirements without starting a server")
    parser.add_argument("--no-browser", action="store_true", help="Start the server without opening a browser")
    args = parser.parse_args(argv)
    notebook = PACKAGE_ROOT / "UAV_Revision_Study.ipynb"
    config_path = PACKAGE_ROOT / "multimodal_config.json"
    if not notebook.is_file() or not config_path.is_file():
        parser.exit(2, "Extract the whole code ZIP before launching JupyterLab.\n")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing = [name for name in ["jupyterlab", "ipykernel"] if importlib.util.find_spec(name) is None]
    if missing:
        parser.exit(2, f"Missing {', '.join(missing)} in {sys.executable}. Activate uav-all-modalities and follow JUPYTERLAB_START_HERE.md.\n")
    command = [sys.executable, "-m", "jupyterlab",
               f"--ServerApp.root_dir={PACKAGE_ROOT}",
               "--LabApp.default_url=/lab/tree/UAV_Revision_Study.ipynb"]
    if args.no_browser:
        command.append("--no-browser")
    if args.check:
        print(json.dumps({"python_executable": sys.executable, "package_root": str(PACKAGE_ROOT),
                          "notebook": str(notebook), "configured_models": len(config["models"]),
                          "model_keys": config["models"], "modalities": config["modalities"],
                          "launch_command": command}, indent=2))
        return 0
    return subprocess.call(command, cwd=PACKAGE_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
