# data preparation code
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def get_column_names() -> list[str]:
    return (
        ["unit_id", "time_cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )


def _load_txt_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    return df


def add_train_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = train_df.groupby("unit_id")["time_cycle"].max().reset_index()
    max_cycle.columns = ["unit_id", "max_cycle"]

    out = train_df.merge(max_cycle, on="unit_id", how="left")
    out["RUL"] = out["max_cycle"] - out["time_cycle"]
    out = out.drop(columns=["max_cycle"])
    return out


def load_fd001_data(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load FD001 train, test, and RUL files from data/raw.
    """
    data_dir = Path(data_dir)

    train_path = data_dir / "train_FD001.txt"
    test_path = data_dir / "test_FD001.txt"
    rul_path = data_dir / "RUL_FD001.txt"

    train = _load_txt_file(train_path)
    test = _load_txt_file(test_path)
    rul = _load_txt_file(rul_path)

    col_names = get_column_names()
    train.columns = col_names
    test.columns = col_names
    rul.columns = ["RUL"]

    train = add_train_rul(train)

    return train, test, rul


def run_data_prep(data_dir: str | Path = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test, rul = load_fd001_data(data_dir)
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("RUL shape:", rul.shape)
    return train, test, rul
