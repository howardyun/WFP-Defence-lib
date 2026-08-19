from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from _bootstrap import ensure_project_root

ensure_project_root(required_modules=("burst", "data", "detector", "generator", "pseudo"))
from toob.burst import (
    apply_burst_perturbation,
    burst_to_direction,
    direction_to_burst,
    direction_to_sign_sequence,
    overhead_ratio,
)
from toob.data import labels_to_int, load_npz_dataset
from toob.detector import format_detector_input, load_detector
from toob.generator import build_generator_from_checkpoint
from toob.pseudo import load_website_to_pseudo_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detector performance on clean or TOOB-defended data.")
    parser.add_argument("--input-npz", required=True, help="Validation/test .npz file to evaluate.")
    parser.add_argument("--input-kind", choices=("direction", "burst"), default="direction")
    parser.add_argument("--data-key", default="data")
    parser.add_argument("--labels-key", default="labels")
    parser.add_argument("--limit", type=int, help="Optional sample limit for quick checks.")
    parser.add_argument("--exclude-labels", nargs="*", type=int, default=(), help="Labels to drop before defense/evaluation.")

    parser.add_argument("--defense", choices=("none", "toob"), default="none")
    parser.add_argument("--generator-dir", help="Directory containing generator_pseudo_*.pt files.")
    parser.add_argument("--pseudo-json", help="JSON containing website_to_pseudo_label mapping.")
    parser.add_argument("--drop-unmapped", action="store_true", help="Drop labels missing from pseudo mapping.")
    parser.add_argument("--max-bursts", type=int, default=2000)
    parser.add_argument("--max-trace-len", type=int, default=5000)
    parser.add_argument("--round", action="store_true", help="Round defended burst counts.")

    parser.add_argument("--detector-builder", required=True, help="module:function or file.py:function")
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--detector-state-dict-key")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--detector-input-kind", choices=("direction", "burst"), default="direction")
    parser.add_argument("--detector-input-layout", choices=("nl", "ncl", "nchw"), default="ncl")

    parser.add_argument("--metrics", nargs="+", choices=("accuracy", "precision", "recall", "f1"), default=("accuracy", "precision", "recall", "f1"))
    parser.add_argument("--average", choices=("macro", "micro", "weighted"), default="macro")
    parser.add_argument("--include-all-classes", action="store_true", help="Average over 0..num_classes-1 instead of observed classes.")
    parser.add_argument("--include-per-class", action="store_true", help="Store per-class precision/recall/f1/support.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--predictions-output", help="Optional .npz file for labels/predictions/pseudo labels.")

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_generators(generator_dir: str | Path, device: torch.device) -> dict[int, torch.nn.Module]:
    models = {}
    for path in Path(generator_dir).glob("generator_pseudo_*.pt"):
        checkpoint = torch.load(path, map_location=device)
        pseudo_label = int(checkpoint["pseudo_label"])
        model = build_generator_from_checkpoint(checkpoint, map_location=device)
        model.eval()
        models[pseudo_label] = model
    if not models:
        raise FileNotFoundError(f"No generator_pseudo_*.pt files found in {generator_dir}")
    return models


def load_optional_array(path: str | Path, key: str) -> np.ndarray | None:
    with np.load(path, allow_pickle=True) as npz:
        if key not in npz:
            return None
        return np.asarray(npz[key])


def map_pseudo_labels(
    labels: np.ndarray,
    mapping: dict[int, int],
    *,
    drop_unmapped: bool,
) -> tuple[np.ndarray, np.ndarray]:
    pseudo_labels = []
    keep_indices = []
    for index, label in enumerate(labels_to_int(labels)):
        pseudo_label = mapping.get(int(label))
        if pseudo_label is None:
            if drop_unmapped:
                continue
            raise KeyError(f"No pseudo label mapping found for website label {int(label)}")
        keep_indices.append(index)
        pseudo_labels.append(pseudo_label)
    return np.asarray(pseudo_labels, dtype=np.int64), np.asarray(keep_indices, dtype=np.int64)


def deploy_toob_defense(
    *,
    data: np.ndarray,
    labels: np.ndarray,
    input_kind: str,
    generator_dir: str | Path,
    pseudo_json: str | Path,
    drop_unmapped: bool,
    max_bursts: int,
    round_output: bool,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mapping = load_website_to_pseudo_label(pseudo_json)
    pseudo_labels, keep_indices = map_pseudo_labels(labels, mapping, drop_unmapped=drop_unmapped)
    labels = labels[keep_indices]
    data = data[keep_indices]

    if input_kind == "direction":
        bursts = direction_to_burst(data, max_bursts=max_bursts)
    else:
        bursts = np.asarray(data, dtype=np.float32)

    generators = load_generators(generator_dir, device)
    defended = np.zeros_like(bursts, dtype=np.float32)
    overhead_values = np.zeros((bursts.shape[0],), dtype=np.float32)

    for pseudo_label in np.unique(pseudo_labels):
        pseudo_label = int(pseudo_label)
        if pseudo_label not in generators:
            raise KeyError(f"Missing generator for pseudo label {pseudo_label}")
        model = generators[pseudo_label]
        indices = np.flatnonzero(pseudo_labels == pseudo_label)
        for start in tqdm(range(0, len(indices), batch_size), desc=f"defense pseudo={pseudo_label}"):
            batch_indices = indices[start:start + batch_size]
            x = torch.from_numpy(bursts[batch_indices].astype(np.float32)).to(device)
            with torch.no_grad():
                delta = model(x)
                adv = apply_burst_perturbation(x, delta, round_output=round_output)
                batch_overhead = overhead_ratio(x, adv).detach().cpu().numpy()
            defended[batch_indices] = adv.detach().cpu().numpy()
            overhead_values[batch_indices] = batch_overhead

    return defended, labels, pseudo_labels, keep_indices, overhead_values


def prepare_detector_data(
    data: np.ndarray,
    *,
    current_kind: str,
    detector_input_kind: str,
    max_bursts: int,
    max_trace_len: int,
) -> np.ndarray:
    if current_kind == detector_input_kind:
        if detector_input_kind == "direction":
            return direction_to_sign_sequence(data, max_trace_len=max_trace_len)
        return np.asarray(data, dtype=np.float32)
    if current_kind == "direction" and detector_input_kind == "burst":
        return direction_to_burst(data, max_bursts=max_bursts)
    if current_kind == "burst" and detector_input_kind == "direction":
        return burst_to_direction(data, max_trace_len=max_trace_len)
    raise ValueError(f"Cannot convert {current_kind!r} to {detector_input_kind!r}")


def predict_detector(
    detector: torch.nn.Module,
    data: np.ndarray,
    *,
    labels: np.ndarray,
    detector_input_kind: str,
    detector_input_layout: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    predictions = []
    for start in tqdm(range(0, len(labels), batch_size), desc="evaluate detector"):
        batch = torch.from_numpy(data[start:start + batch_size].astype(np.float32)).to(device)
        x = format_detector_input(batch, kind=detector_input_kind, layout=detector_input_layout)
        with torch.no_grad():
            logits = detector(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            pred = torch.argmax(logits, dim=1)
        predictions.append(pred.detach().cpu().numpy())
    if not predictions:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(predictions).astype(np.int64)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    metric_names: list[str],
    average: str,
    num_classes: int,
    include_all_classes: bool,
    include_per_class: bool,
) -> dict:
    labels = labels_to_int(labels)
    predictions = labels_to_int(predictions)
    result: dict[str, object] = {}

    if len(labels) == 0:
        for metric in metric_names:
            key = "f1_score" if metric == "f1" else metric
            result[key if metric == "accuracy" else f"{key}_{average}"] = 0.0
        return result

    if include_all_classes:
        classes = np.arange(num_classes, dtype=np.int64)
    else:
        classes = np.union1d(labels, predictions).astype(np.int64)

    tp_values = []
    fp_values = []
    fn_values = []
    support_values = []
    per_class = {}
    for cls in classes:
        cls = int(cls)
        true_mask = labels == cls
        pred_mask = predictions == cls
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        support = int(np.sum(true_mask))
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)

        tp_values.append(tp)
        fp_values.append(fp)
        fn_values.append(fn)
        support_values.append(support)
        if include_per_class:
            per_class[str(cls)] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": support,
            }

    precision_values = np.asarray([
        safe_divide(tp, tp + fp)
        for tp, fp in zip(tp_values, fp_values)
    ], dtype=np.float64)
    recall_values = np.asarray([
        safe_divide(tp, tp + fn)
        for tp, fn in zip(tp_values, fn_values)
    ], dtype=np.float64)
    f1_values = np.asarray([
        safe_divide(2.0 * p * r, p + r)
        for p, r in zip(precision_values, recall_values)
    ], dtype=np.float64)
    support = np.asarray(support_values, dtype=np.float64)

    def average_values(values: np.ndarray) -> float:
        if average == "macro":
            return float(np.mean(values)) if len(values) else 0.0
        if average == "weighted":
            total = float(np.sum(support))
            return float(np.sum(values * support) / total) if total > 0 else 0.0
        if average == "micro":
            tp_sum = float(np.sum(tp_values))
            fp_sum = float(np.sum(fp_values))
            fn_sum = float(np.sum(fn_values))
            if values is precision_values:
                return safe_divide(tp_sum, tp_sum + fp_sum)
            if values is recall_values:
                return safe_divide(tp_sum, tp_sum + fn_sum)
            p_micro = safe_divide(tp_sum, tp_sum + fp_sum)
            r_micro = safe_divide(tp_sum, tp_sum + fn_sum)
            return safe_divide(2.0 * p_micro * r_micro, p_micro + r_micro)
        raise ValueError(f"Unknown average: {average}")

    if "accuracy" in metric_names:
        result["accuracy"] = float(np.mean(labels == predictions))
    if "precision" in metric_names:
        result[f"precision_{average}"] = average_values(precision_values)
    if "recall" in metric_names:
        result[f"recall_{average}"] = average_values(recall_values)
    if "f1" in metric_names:
        result[f"f1_score_{average}"] = average_values(f1_values)
    if include_per_class:
        result["per_class"] = per_class
    return result


def summarize_overhead(values: np.ndarray | None) -> dict | None:
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    data, labels = load_npz_dataset(args.input_npz, data_key=args.data_key, labels_key=args.labels_key)
    labels = labels_to_int(labels)
    if args.limit:
        data = data[:args.limit]
        labels = labels[:args.limit]
    original_sample_count = int(len(labels))

    pseudo_labels = load_optional_array(args.input_npz, "pseudo_labels")
    overhead_values = load_optional_array(args.input_npz, "overhead")
    if pseudo_labels is not None and args.limit:
        pseudo_labels = pseudo_labels[:args.limit]
    if overhead_values is not None and args.limit:
        overhead_values = overhead_values[:args.limit]

    if args.exclude_labels:
        keep_mask = ~np.isin(labels, np.asarray(args.exclude_labels, dtype=np.int64))
        data = data[keep_mask]
        labels = labels[keep_mask]
        if pseudo_labels is not None:
            pseudo_labels = pseudo_labels[keep_mask]
        if overhead_values is not None:
            overhead_values = overhead_values[keep_mask]

    keep_indices = np.arange(len(labels), dtype=np.int64)
    current_kind = args.input_kind

    if args.defense == "toob":
        if not args.generator_dir:
            raise ValueError("--generator-dir is required when --defense toob")
        if not args.pseudo_json:
            raise ValueError("--pseudo-json is required when --defense toob")
        data, labels, pseudo_labels, keep_indices, overhead_values = deploy_toob_defense(
            data=data,
            labels=labels,
            input_kind=args.input_kind,
            generator_dir=args.generator_dir,
            pseudo_json=args.pseudo_json,
            drop_unmapped=args.drop_unmapped,
            max_bursts=args.max_bursts,
            round_output=args.round,
            batch_size=args.batch_size,
            device=device,
        )
        current_kind = "burst"
    else:
        if pseudo_labels is not None:
            pseudo_labels = labels_to_int(pseudo_labels)

    detector_data = prepare_detector_data(
        data,
        current_kind=current_kind,
        detector_input_kind=args.detector_input_kind,
        max_bursts=args.max_bursts,
        max_trace_len=args.max_trace_len,
    )

    detector = load_detector(
        builder_spec=args.detector_builder,
        checkpoint_path=args.detector_checkpoint,
        num_classes=args.num_classes,
        device=device,
        state_dict_key=args.detector_state_dict_key,
    )
    predictions = predict_detector(
        detector,
        detector_data,
        labels=labels,
        detector_input_kind=args.detector_input_kind,
        detector_input_layout=args.detector_input_layout,
        batch_size=args.batch_size,
        device=device,
    )
    metrics = compute_metrics(
        labels,
        predictions,
        metric_names=list(args.metrics),
        average=args.average,
        num_classes=args.num_classes,
        include_all_classes=args.include_all_classes,
        include_per_class=args.include_per_class,
    )

    summary = {
        "input_npz": str(args.input_npz),
        "input_kind": args.input_kind,
        "exclude_labels": [int(label) for label in args.exclude_labels],
        "defense": {
            "type": args.defense,
            "generator_dir": str(args.generator_dir) if args.generator_dir else None,
            "pseudo_json": str(args.pseudo_json) if args.pseudo_json else None,
            "round": bool(args.round),
        },
        "detector": {
            "builder": args.detector_builder,
            "checkpoint": str(args.detector_checkpoint),
            "num_classes": int(args.num_classes),
            "input_kind": args.detector_input_kind,
            "input_layout": args.detector_input_layout,
        },
        "metrics": metrics,
        "average": args.average,
        "num_samples": int(len(labels)),
        "num_predictions": int(len(predictions)),
        "num_dropped": int(original_sample_count - len(labels)),
        "overhead": summarize_overhead(overhead_values),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.predictions_output:
        arrays = {
            "labels": labels,
            "predictions": predictions,
        }
        if pseudo_labels is not None:
            arrays["pseudo_labels"] = labels_to_int(pseudo_labels)
        if overhead_values is not None:
            arrays["overhead"] = np.asarray(overhead_values)
        np.savez_compressed(args.predictions_output, **arrays)

    print("evaluation metrics:")
    for key, value in metrics.items():
        if key == "per_class":
            continue
        print(f"  {key}: {float(value):.6f}")
    overhead = summary["overhead"]
    if overhead is not None:
        print(f"  overhead_mean: {overhead['mean']:.6f}")
        print(f"  overhead_median: {overhead['median']:.6f}")
    print(f"saved metrics: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
