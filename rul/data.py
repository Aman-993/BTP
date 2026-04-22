from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


INDEX_COLUMNS = ["unit", "cycle"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS


@dataclass
class PreparedData:
    train_dataset: "SequenceDataset"
    val_dataset: "SequenceDataset"
    test_dataset: "SequenceDataset"
    input_size: int
    feature_columns: list[str]
    scaler: MinMaxScaler


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, unit_ids: np.ndarray | None = None) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        self.y = y_tensor.view(-1, 1) if y_tensor.ndim == 1 else y_tensor
        self.unit_ids = unit_ids

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class RolloutSequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, future: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.future = torch.tensor(future, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.future[idx]


def read_cmapss_file(path: str | Path) -> pd.DataFrame:
    """Read whitespace-separated C-MAPSS files with trailing spaces."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLUMNS)


def find_cmapss_split_files(data_dir: str | Path) -> tuple[Path, Path, Path]:
    data_dir = Path(data_dir)
    train_files = sorted(data_dir.glob("train_*.txt"))
    test_files = sorted(data_dir.glob("test_*.txt"))
    rul_files = sorted(data_dir.glob("RUL_*.txt"))

    if len(train_files) != 1 or len(test_files) != 1 or len(rul_files) != 1:
        raise FileNotFoundError(
            f"{data_dir} must contain exactly one train_*.txt, one test_*.txt, "
            "and one RUL_*.txt file. For example: train_FD001.txt, "
            "test_FD001.txt, RUL_FD001.txt."
        )

    return train_files[0], test_files[0], rul_files[0]


def add_train_rul(df: pd.DataFrame, rul_cap: int) -> pd.DataFrame:
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = (max_cycle - out["cycle"]).clip(upper=rul_cap)
    return out


def add_test_rul(df: pd.DataFrame, rul_path: str | Path, rul_cap: int) -> pd.DataFrame:
    out = df.copy()
    final_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["final_rul"])
    final_rul["unit"] = np.arange(1, len(final_rul) + 1)
    last_cycle = out.groupby("unit")["cycle"].max().rename("last_cycle")
    out = out.merge(final_rul, on="unit", how="left").merge(last_cycle, on="unit", how="left")
    out["rul"] = (out["last_cycle"] - out["cycle"] + out["final_rul"]).clip(upper=rul_cap)
    return out.drop(columns=["final_rul", "last_cycle"])


def choose_feature_columns(train_df: pd.DataFrame) -> list[str]:
    candidates = SETTING_COLUMNS + SENSOR_COLUMNS
    # FD001 contains several constant columns; removing them makes scaling stable.
    return [col for col in candidates if train_df[col].std() > 1e-10]


def make_train_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    unit_list: list[int] = []

    for unit, group in df.groupby("unit"):
        group = group.sort_values("cycle")
        features = group[feature_columns].to_numpy(dtype=np.float32)
        targets = group["rul"].to_numpy(dtype=np.float32)
        if len(group) < sequence_length:
            continue
        for end in range(sequence_length, len(group) + 1):
            start = end - sequence_length
            x_list.append(features[start:end])
            y_list.append(targets[end - 1])
            unit_list.append(unit)

    return np.asarray(x_list), np.asarray(y_list), np.asarray(unit_list)


def make_last_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    unit_list: list[int] = []

    for unit, group in df.groupby("unit"):
        group = group.sort_values("cycle")
        features = group[feature_columns].to_numpy(dtype=np.float32)
        targets = group["rul"].to_numpy(dtype=np.float32)
        if len(group) < sequence_length:
            pad_len = sequence_length - len(group)
            pad = np.repeat(features[:1], repeats=pad_len, axis=0)
            features = np.concatenate([pad, features], axis=0)
        x_list.append(features[-sequence_length:])
        y_list.append(targets[-1])
        unit_list.append(unit)

    return np.asarray(x_list), np.asarray(y_list), np.asarray(unit_list)


def make_rollout_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_list: list[np.ndarray] = []
    future_list: list[np.ndarray] = []

    for _, group in df.groupby("unit"):
        group = group.sort_values("cycle")
        features = group[feature_columns].to_numpy(dtype=np.float32)
        if len(group) < sequence_length + horizon:
            continue
        for start in range(0, len(group) - sequence_length - horizon + 1):
            input_end = start + sequence_length
            future_end = input_end + horizon
            x_list.append(features[start:input_end])
            future_list.append(features[input_end:future_end])

    return np.asarray(x_list), np.asarray(future_list)


def prepare_datasets(
    data_dir: str | Path,
    sequence_length: int = 30,
    rul_cap: int = 125,
    val_size: float = 0.2,
    seed: int = 42,
) -> PreparedData:
    data_dir = Path(data_dir)
    train_path, test_path, rul_path = find_cmapss_split_files(data_dir)
    train_df = add_train_rul(read_cmapss_file(train_path), rul_cap=rul_cap)
    test_df = add_test_rul(read_cmapss_file(test_path), rul_path, rul_cap=rul_cap)

    feature_columns = choose_feature_columns(train_df)
    float_dtypes = {col: np.float32 for col in feature_columns}
    train_df = train_df.astype(float_dtypes)
    test_df = test_df.astype(float_dtypes)

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(train_df, groups=train_df["unit"]))
    train_split = train_df.iloc[train_idx].copy()
    val_split = train_df.iloc[val_idx].copy()

    scaler = MinMaxScaler()
    scaler.fit(train_split[feature_columns])

    for frame in (train_split, val_split, test_df):
        scaled = scaler.transform(frame[feature_columns]).astype(np.float32)
        for idx, col in enumerate(feature_columns):
            frame[col] = scaled[:, idx]

    x_train, y_train, train_units = make_train_sequences(train_split, feature_columns, sequence_length)
    x_val, y_val, val_units = make_last_sequences(val_split, feature_columns, sequence_length)
    x_test, y_test, test_units = make_last_sequences(test_df, feature_columns, sequence_length)

    return PreparedData(
        train_dataset=SequenceDataset(x_train, y_train, train_units),
        val_dataset=SequenceDataset(x_val, y_val, val_units),
        test_dataset=SequenceDataset(x_test, y_test, test_units),
        input_size=len(feature_columns),
        feature_columns=feature_columns,
        scaler=scaler,
    )


def prepare_rollout_dataset(
    data_dir: str | Path,
    sequence_length: int = 30,
    horizon: int = 5,
    split: str = "test",
    val_size: float = 0.2,
    seed: int = 42,
) -> RolloutSequenceDataset:
    """Prepare observed feature windows for K-step rollout evaluation.

    This is separate from RUL evaluation because rollout uses future feature
    vectors as ground truth, not scalar RUL labels.
    """
    data_dir = Path(data_dir)
    train_path, test_path, _ = find_cmapss_split_files(data_dir)
    train_df = read_cmapss_file(train_path)
    eval_df = read_cmapss_file(test_path if split == "test" else train_path)

    feature_columns = choose_feature_columns(train_df)
    float_dtypes = {col: np.float32 for col in feature_columns}
    train_df = train_df.astype(float_dtypes)
    eval_df = eval_df.astype(float_dtypes)

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, _ = next(splitter.split(train_df, groups=train_df["unit"]))
    scaler_train = train_df.iloc[train_idx].copy()

    scaler = MinMaxScaler()
    scaler.fit(scaler_train[feature_columns])
    scaled = scaler.transform(eval_df[feature_columns]).astype(np.float32)
    for idx, col in enumerate(feature_columns):
        eval_df[col] = scaled[:, idx]

    x, future = make_rollout_sequences(eval_df, feature_columns, sequence_length, horizon)
    return RolloutSequenceDataset(x, future)


def prepare_next_step_datasets(
    data_dir: str | Path,
    sequence_length: int = 30,
    val_size: float = 0.2,
    seed: int = 42,
) -> PreparedData:
    """Prepare datasets for next-feature-vector prediction.

    The target is the next scaled feature vector after each input window.
    This is the training task needed for K-step rollout evaluation.
    """
    data_dir = Path(data_dir)
    train_path, test_path, _ = find_cmapss_split_files(data_dir)
    train_df = read_cmapss_file(train_path)
    test_df = read_cmapss_file(test_path)

    feature_columns = choose_feature_columns(train_df)
    float_dtypes = {col: np.float32 for col in feature_columns}
    train_df = train_df.astype(float_dtypes)
    test_df = test_df.astype(float_dtypes)

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(train_df, groups=train_df["unit"]))
    train_split = train_df.iloc[train_idx].copy()
    val_split = train_df.iloc[val_idx].copy()

    scaler = MinMaxScaler()
    scaler.fit(train_split[feature_columns])

    for frame in (train_split, val_split, test_df):
        scaled = scaler.transform(frame[feature_columns]).astype(np.float32)
        for idx, col in enumerate(feature_columns):
            frame[col] = scaled[:, idx]

    x_train, future_train = make_rollout_sequences(train_split, feature_columns, sequence_length, horizon=1)
    x_val, future_val = make_rollout_sequences(val_split, feature_columns, sequence_length, horizon=1)
    x_test, future_test = make_rollout_sequences(test_df, feature_columns, sequence_length, horizon=1)

    y_train = future_train[:, 0, :]
    y_val = future_val[:, 0, :]
    y_test = future_test[:, 0, :]

    return PreparedData(
        train_dataset=SequenceDataset(x_train, y_train),
        val_dataset=SequenceDataset(x_val, y_val),
        test_dataset=SequenceDataset(x_test, y_test),
        input_size=len(feature_columns),
        feature_columns=feature_columns,
        scaler=scaler,
    )
