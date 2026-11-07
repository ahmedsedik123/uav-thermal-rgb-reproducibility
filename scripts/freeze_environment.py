"""Save exact installed versions after validating the target training environment."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from uav_review.core import environment

target = root / "environment-lock"
target.mkdir(exist_ok=True)
result = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
(target / "requirements.lock.txt").write_text(result.stdout)
(target / "environment.json").write_text(json.dumps(environment(), indent=2))
print(target)
