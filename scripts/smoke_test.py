from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
command = [
    sys.executable,
    "-m",
    "pohang_donor_models.cli",
    "all",
    "--quick",
    "--run-name",
    "smoke_test",
]
print("RUN:", " ".join(command))
raise SystemExit(subprocess.call(command, cwd=PROJECT_ROOT))
