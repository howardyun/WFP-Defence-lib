from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect TOOB training-tuning metrics.")
    parser.add_argument("--tune-dir", help="Directory containing run_* subdirectories.")
    parser.add_argument("--output-csv", help="CSV summary path.")
    parser.add_argument("--output-json", help="JSON summary path.")
    parser.add_argument("--best-csv", help="Best-per-target CSV path.")
    parser.add_argument("--best-json", help="Best-per-target JSON path.")
    parser.add_argument("--budget-slack", type=float, default=0.02, help="Allowed overhead overshoot when choosing best rows.")
    parser.add_argument("--select-metric", default="accuracy", help="Metric minimized when selecting best rows.")

    parser.add_argument("--write-config", help="Write one run_config.json and exit.")
    parser.add_argument("--run-name")
    parser.add_argument("--overhead-threshold", type=float)
    parser.add_argument("--lambda-overhead", type=float)
    parser.add_argument("--overhead-loss")
    parser.add_argument("--overhead-tolerance", type=float)
    parser.add_argument("--attack-loss")
    parser.add_argument("--soft-projection-tau", type=float)
    parser.add_argument("--set-size", type=int)
    parser.add_argument("--cluster-count", type=int)
    parser.add_argument("--mapping")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--noise-dim", type=int)
    parser.add_argument("--output-npz")
    parser.add_argument("--metrics-json")
    return parser.parse_args()


def write_config(args: argparse.Namespace) -> int:
    path = Path(args.write_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "run_name": args.run_name,
        "overhead_threshold": args.overhead_threshold,
        "lambda_overhead": args.lambda_overhead,
        "overhead_loss": args.overhead_loss,
        "overhead_tolerance": args.overhead_tolerance,
        "attack_loss": args.attack_loss,
        "soft_projection_tau": args.soft_projection_tau,
        "set_size": args.set_size,
        "cluster_count": args.cluster_count,
        "mapping": args.mapping,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "noise_dim": args.noise_dim,
        "output_npz": args.output_npz,
        "metrics_json": args.metrics_json,
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return 0


def flatten_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    overhead = summary.get("overhead") or {}
    return {
        "accuracy": metrics.get("accuracy"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "f1_score_macro": metrics.get("f1_score_macro"),
        "precision_weighted": metrics.get("precision_weighted"),
        "recall_weighted": metrics.get("recall_weighted"),
        "f1_score_weighted": metrics.get("f1_score_weighted"),
        "overhead_mean": overhead.get("mean"),
        "overhead_median": overhead.get("median"),
        "overhead_max": overhead.get("max"),
        "num_samples": summary.get("num_samples"),
        "num_dropped": summary.get("num_dropped"),
        "eval_average": summary.get("average"),
    }


def collect(tune_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for metrics_path in sorted(tune_dir.glob("run_*/defense_eval_metrics.json")):
        run_dir = metrics_path.parent
        config_path = run_dir / "run_config.json"
        config = {}
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))

        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "run_name": config.get("run_name", run_dir.name),
            "overhead_threshold": config.get("overhead_threshold"),
            "lambda_overhead": config.get("lambda_overhead"),
            "overhead_loss": config.get("overhead_loss"),
            "overhead_tolerance": config.get("overhead_tolerance"),
            "attack_loss": config.get("attack_loss"),
            "soft_projection_tau": config.get("soft_projection_tau"),
            "set_size": config.get("set_size"),
            "cluster_count": config.get("cluster_count"),
            "mapping": config.get("mapping"),
            "epochs": config.get("epochs"),
            "batch_size": config.get("batch_size"),
            "noise_dim": config.get("noise_dim"),
            "output_npz": config.get("output_npz"),
            "metrics_json": str(metrics_path),
        }
        row.update(flatten_metrics(summary))
        rows.append(row)
    return rows


def write_outputs(rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = [
        "run_name",
        "overhead_threshold",
        "lambda_overhead",
        "overhead_loss",
        "overhead_tolerance",
        "attack_loss",
        "soft_projection_tau",
        "set_size",
        "cluster_count",
        "mapping",
        "epochs",
        "batch_size",
        "noise_dim",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_score_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_score_weighted",
        "overhead_mean",
        "overhead_median",
        "overhead_max",
        "num_samples",
        "num_dropped",
        "eval_average",
        "output_npz",
        "metrics_json",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def select_best_by_target(
    rows: list[dict[str, Any]],
    *,
    budget_slack: float,
    select_metric: str,
) -> list[dict[str, Any]]:
    def number(value: Any, default: float = 1e9) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        target = row.get("overhead_threshold")
        if target is None:
            continue
        grouped.setdefault(float(target), []).append(row)

    best_rows = []
    for target, candidates in sorted(grouped.items()):
        eligible = [
            row for row in candidates
            if row.get("overhead_mean") is not None and float(row["overhead_mean"]) <= target + budget_slack
        ]
        if eligible:
            selected = min(eligible, key=lambda row: number(row.get(select_metric)))
            reason = "within_budget"
        else:
            selected = min(
                candidates,
                key=lambda row: abs(number(row.get("overhead_mean")) - target),
            )
            reason = "closest_overhead"
        copied = dict(selected)
        copied["target"] = target
        copied["selection_reason"] = reason
        copied["budget_slack"] = budget_slack
        copied["select_metric"] = select_metric
        best_rows.append(copied)
    return best_rows


def write_best_outputs(rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.write_config:
        return write_config(args)
    if not args.tune_dir:
        raise ValueError("--tune-dir is required unless --write-config is used")

    tune_dir = Path(args.tune_dir)
    csv_path = Path(args.output_csv) if args.output_csv else tune_dir / "summary.csv"
    json_path = Path(args.output_json) if args.output_json else tune_dir / "summary.json"
    best_csv_path = Path(args.best_csv) if args.best_csv else tune_dir / "best_by_target.csv"
    best_json_path = Path(args.best_json) if args.best_json else tune_dir / "best_by_target.json"
    rows = collect(tune_dir)
    write_outputs(rows, csv_path, json_path)
    best_rows = select_best_by_target(rows, budget_slack=args.budget_slack, select_metric=args.select_metric)
    write_best_outputs(best_rows, best_csv_path, best_json_path)
    print(f"collected {len(rows)} runs")
    print(f"saved csv: {csv_path}")
    print(f"saved json: {json_path}")
    print(f"saved best csv: {best_csv_path}")
    print(f"saved best json: {best_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
