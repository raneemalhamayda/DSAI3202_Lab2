from __future__ import annotations

from typing import Tuple

import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute


def build_train_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build rolling windows for train data.
    Each window gets the RUL at its end cycle.
    """
    all_windows = []
    all_targets = []

    df = df.sort_values(["unit_id", "time_cycle"]).copy()

    for unit_id, g in df.groupby("unit_id"):
        g = g.reset_index(drop=True)

        if len(g) < window:
            continue

        for end_idx in range(window - 1, len(g)):
            current_cycle = int(g.loc[end_idx, "time_cycle"])
            current_rul = float(g.loc[end_idx, "RUL"])

            w = g.loc[end_idx - window + 1 : end_idx, ["time_cycle"] + feature_cols].copy()
            window_id = f"{unit_id}_{current_cycle}"
            w.insert(0, "window_id", window_id)

            all_windows.append(w)
            all_targets.append(
                {
                    "window_id": window_id,
                    "unit_id": int(unit_id),
                    "time_cycle": current_cycle,
                    "RUL": current_rul,
                }
            )

    windows_df = pd.concat(all_windows, ignore_index=True)
    targets_df = pd.DataFrame(all_targets)

    return windows_df, targets_df


def build_test_windows(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    feature_cols: list[str],
    window: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each test engine, take only the last available window.
    Match it with the provided test RUL.
    """
    all_windows = []
    all_targets = []

    test_df = test_df.sort_values(["unit_id", "time_cycle"]).copy()

    for idx, (unit_id, g) in enumerate(test_df.groupby("unit_id"), start=1):
        g = g.reset_index(drop=True)

        if len(g) < window:
            continue

        end_idx = len(g) - 1
        current_cycle = int(g.loc[end_idx, "time_cycle"])
        true_rul = float(rul_df.iloc[idx - 1, 0])

        w = g.loc[end_idx - window + 1 : end_idx, ["time_cycle"] + feature_cols].copy()
        window_id = f"{unit_id}_{current_cycle}"
        w.insert(0, "window_id", window_id)

        all_windows.append(w)
        all_targets.append(
            {
                "window_id": window_id,
                "unit_id": int(unit_id),
                "time_cycle": current_cycle,
                "RUL": true_rul,
            }
        )

    windows_df = pd.concat(all_windows, ignore_index=True)
    targets_df = pd.DataFrame(all_targets)

    return windows_df, targets_df


def extract_tsfresh_features(
    windows_df: pd.DataFrame,
    n_jobs: int = 1,
) -> pd.DataFrame:
    X = extract_features(
        windows_df,
        column_id="window_id",
        column_sort="time_cycle",
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=n_jobs,
    )
    impute(X)
    X = X.reset_index().rename(columns={"index": "window_id"})
    return X


def merge_targets_and_features(
    targets_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    return targets_df.merge(features_df, on="window_id", how="inner")
