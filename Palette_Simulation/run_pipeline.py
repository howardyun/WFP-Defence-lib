#!/usr/bin/env python3
"""Run the Palette simulation pipeline.

This is the Python equivalent of ``run_pipeline.sh``. It keeps the working
directory at ``Palette_Simulation/palette`` so the existing relative paths in
the Palette scripts continue to work.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PIPELINE_STEPS = (
    ("extract", "Extract TAM Features", ("extract-list.py",)),
    ("cluster", "Website Clustering", ("cluster.py",)),
    ("refinement", "Super-Matrix Refinement", ("refinement.py",)),
    ("regularization", "Regularization (Defense Simulation)", ("regularization.py",)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Palette simulation pipeline.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to run each pipeline step.",
    )
    parser.add_argument(
        "--from-step",
        choices=[name for name, _, _ in PIPELINE_STEPS],
        help="Start from this step and run the remaining steps.",
    )
    parser.add_argument(
        "--only",
        choices=[name for name, _, _ in PIPELINE_STEPS],
        help="Run only one pipeline step.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip feature extraction.",
    )
    parser.add_argument(
        "--skip-cluster",
        action="store_true",
        help="Skip website clustering.",
    )
    parser.add_argument(
        "--skip-refinement",
        action="store_true",
        help="Skip super-matrix refinement.",
    )
    parser.add_argument(
        "--skip-regularization",
        action="store_true",
        help="Skip regularization and defense trace generation.",
    )
    return parser.parse_args()


def selected_steps(args: argparse.Namespace) -> list[tuple[str, str, tuple[str, ...]]]:
    steps = list(PIPELINE_STEPS)

    if args.only:
        return [step for step in steps if step[0] == args.only]

    if args.from_step:
        start_idx = next(idx for idx, step in enumerate(steps) if step[0] == args.from_step)
        steps = steps[start_idx:]

    skip_flags = {
        "extract": args.skip_extract,
        "cluster": args.skip_cluster,
        "refinement": args.skip_refinement,
        "regularization": args.skip_regularization,
    }
    return [step for step in steps if not skip_flags[step[0]]]


def create_alias_dir(root_dir: Path) -> Path:
    alias_dir = root_dir / f".palette_pipeline_{os.getpid()}"
    simulation_dir = alias_dir / "Simulation"
    simulation_dir.mkdir(parents=True, exist_ok=True)
    (alias_dir / ".generated_by_run_pipeline").write_text("", encoding="utf-8")
    return alias_dir


def remove_alias_dir(alias_dir: Path) -> None:
    marker = alias_dir / ".generated_by_run_pipeline"
    if marker.exists():
        shutil.rmtree(alias_dir, ignore_errors=True)


def build_child_env(root_dir: Path, alias_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    simulation_dir = alias_dir / "Simulation"
    (simulation_dir / "__init__.py").write_text(
        f"__path__ = [{str(root_dir)!r}]\n",
        encoding="utf-8",
    )

    python_paths = [str(alias_dir), str(root_dir.parent)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env


def run_step(
    python_exe: str,
    palette_dir: Path,
    env: dict[str, str],
    title: str,
    command: tuple[str, ...],
) -> None:
    print(f"========== {title} ==========", flush=True)
    subprocess.run([python_exe, *command], cwd=palette_dir, env=env, check=True)


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    palette_dir = root_dir / "palette"

    if not palette_dir.is_dir():
        raise FileNotFoundError(f"Palette directory not found: {palette_dir}")

    steps = selected_steps(args)
    if not steps:
        print("No pipeline steps selected.")
        return 0

    alias_dir = create_alias_dir(root_dir)
    try:
        env = build_child_env(root_dir, alias_dir)
        for _, title, command in steps:
            run_step(args.python, palette_dir, env, title, command)
    finally:
        remove_alias_dir(alias_dir)

    print("========== Pipeline Complete ==========", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
