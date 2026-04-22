from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine FD001-FD004 into one train/test/RUL C-MAPSS dataset."
    )
    parser.add_argument(
        "--source-dir",
        default="data/raw",
        help="Folder containing train_FD001.txt ... RUL_FD004.txt, or subfolders with those files.",
    )
    parser.add_argument("--output-dir", default="data/raw/FDALL")
    parser.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    return parser.parse_args()


def find_file(source_dir: Path, filename: str) -> Path:
    direct = source_dir / filename
    if direct.exists():
        return direct

    matches = sorted(source_dir.glob(f"**/{filename}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {source_dir}")
    raise FileExistsError(f"Found multiple copies of {filename}: {matches}")


def read_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None)


def offset_units(df: pd.DataFrame, unit_offset: int) -> pd.DataFrame:
    out = df.copy()
    out.iloc[:, 0] = out.iloc[:, 0] + unit_offset
    return out


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_train = []
    combined_test = []
    combined_rul = []
    train_offset = 0
    test_offset = 0
    summary = []

    for subset in args.subsets:
        train_path = find_file(source_dir, f"train_{subset}.txt")
        test_path = find_file(source_dir, f"test_{subset}.txt")
        rul_path = find_file(source_dir, f"RUL_{subset}.txt")

        train_df = read_split(train_path)
        test_df = read_split(test_path)
        rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None)

        train_units = int(train_df.iloc[:, 0].max())
        test_units = int(test_df.iloc[:, 0].max())
        if len(rul_df) != test_units:
            raise ValueError(
                f"{rul_path} has {len(rul_df)} rows, but {test_path} has {test_units} test units."
            )

        combined_train.append(offset_units(train_df, train_offset))
        combined_test.append(offset_units(test_df, test_offset))
        combined_rul.append(rul_df)
        summary.append(
            {
                "subset": subset,
                "train_rows": len(train_df),
                "train_units": train_units,
                "test_rows": len(test_df),
                "test_units": test_units,
            }
        )
        train_offset += train_units
        test_offset += test_units

    train_all = pd.concat(combined_train, ignore_index=True)
    test_all = pd.concat(combined_test, ignore_index=True)
    rul_all = pd.concat(combined_rul, ignore_index=True)

    train_all.to_csv(output_dir / "train_FDALL.txt", sep=" ", header=False, index=False)
    test_all.to_csv(output_dir / "test_FDALL.txt", sep=" ", header=False, index=False)
    rul_all.to_csv(output_dir / "RUL_FDALL.txt", sep=" ", header=False, index=False)

    print("Created combined dataset:")
    print(f"  {output_dir / 'train_FDALL.txt'} rows={len(train_all)} units={train_offset}")
    print(f"  {output_dir / 'test_FDALL.txt'} rows={len(test_all)} units={test_offset}")
    print(f"  {output_dir / 'RUL_FDALL.txt'} rows={len(rul_all)}")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()


