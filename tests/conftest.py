"""Shared pytest setup: make the project's scripts importable from the tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The project is plain scripts rather than a package, so add the two folders
# that contain them to Python's import path.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))
