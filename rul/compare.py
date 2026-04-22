from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one comparison table for all trained models.")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--output", default="outputs/metrics/model_comparison.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(Path(args.metrics_dir).glob("*_test_metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))

    if not rows:
        raise FileNotFoundError(
            f"No test metric files found in {args.metrics_dir}. Run rul.evaluate first."
        )

    table = pd.DataFrame(rows).sort_values("test_rmse")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(f"Saved comparison table to {output}")


if __name__ == "__main__":
    main()
