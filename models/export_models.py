"""
Standalone script to train, evaluate, and export Iris model artifacts to disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from models import main as run_models_main


if __name__ == "__main__":
    run_models_main()
