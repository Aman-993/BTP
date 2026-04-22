from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FD001-only results with FD001-FD004 results.")
    parser.add_argument("--before", default="outputs/metrics/model_comparison.csv")
    parser.add_argument("--after", default="outputs_combined/metrics/model_comparison.csv")
    parser.add_argument("--output", default="outputs_combined/metrics/before_after_comparison.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before = pd.read_csv(args.before).add_prefix("fd001_")
    after = pd.read_csv(args.after).add_prefix("fdall_")

    merged = before.merge(after, left_on="fd001_model", right_on="fdall_model", how="inner")
    merged.insert(0, "model", merged["fd001_model"])
    merged = merged.drop(columns=["fd001_model", "fdall_model"])

    if {"fd001_test_rmse", "fdall_test_rmse"}.issubset(merged.columns):
        merged["rmse_change"] = merged["fdall_test_rmse"] - merged["fd001_test_rmse"]
        merged["rmse_change_pct"] = 100.0 * merged["rmse_change"] / merged["fd001_test_rmse"]

    if {"fd001_test_score", "fdall_test_score"}.issubset(merged.columns):
        merged["score_change"] = merged["fdall_test_score"] - merged["fd001_test_score"]
        merged["score_change_pct"] = 100.0 * merged["score_change"] / merged["fd001_test_score"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(merged.to_string(index=False))
    print(f"Saved before/after comparison to {output}")


if __name__ == "__main__":
    main()
