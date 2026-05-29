"""Verify that all dependencies required by the TOOB pipeline are importable."""

from __future__ import annotations

import sys

from _bootstrap import ensure_project_root

ensure_project_root(required_modules=("burst", "cluster", "data", "detector", "generator", "losses", "pseudo", "train"))


def main() -> None:
    import numpy  # noqa: F401
    import torch  # noqa: F401
    import tqdm  # noqa: F401

    from toob import burst, cluster, data, detector, generator, losses, pseudo, train  # noqa: F401

    print(f"Python : {sys.executable}")
    print(f"numpy  : {numpy.__version__}")
    print(f"torch  : {torch.__version__}")
    print(f"tqdm   : {tqdm.__version__}")
    print("All TOOB imports OK.")


if __name__ == "__main__":
    main()
