"""Pytest configuration and shared fixtures for competitive-velocity-ml-engine."""

import sys
from pathlib import Path

# Ensure root package directory is always on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
