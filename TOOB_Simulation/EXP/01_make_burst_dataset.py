from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toob.burst import direction_to_burst, summarize_bursts
from toob.data import load_npz_dataset, save_npz_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert direction-sequence dataset to burst dataset.")
    parser.add_argument("--input", required=True, help="Input .npz with data and labels.")
    parser.add_argument("--output", required=True, help="Output burst .npz.")
    parser.add_argument("--data-key", default="data")
    parser.add_argument("--labels-key", default="labels")
    parser.add_argument("--max-bursts", type=int, default=2000)
    parser.add_argument("--limit", type=int, help="Optional sample limit for smoke tests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data, labels = load_npz_dataset(args.input, data_key=args.data_key, labels_key=args.labels_key)
    if args.limit:
        data = data[:args.limit]
        labels = labels[:args.limit]
    bursts = direction_to_burst(data, max_bursts=args.max_bursts)
    save_npz_dataset(args.output, data=bursts, labels=labels)
    stats = summarize_bursts(bursts)
    print(f"saved: {args.output}")
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
