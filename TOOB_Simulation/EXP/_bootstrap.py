from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def ensure_project_root(required_modules: tuple[str, ...] = ()) -> None:
    """Add the TOOB_Simulation directory to sys.path so ``toob.*`` is importable."""
    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    for mod in required_modules:
        if importlib.util.find_spec(f"toob.{mod}") is None:
            raise ImportError(
                f"Required subpackage toob.{mod} not found under {project_root}. "
                "Make sure the toob package is complete."
            )
