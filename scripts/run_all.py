from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pohang_donor_models.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["all", *sys.argv[1:]]))
