from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_setting_and_sensor_columns(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    setting_cols = [c for c in df.columns if c.startswith("op_setting_")]
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    return setting_cols, sensor_cols


def basic_validation_report(train: pd.DataFrame, test: pd.DataFrame, rul: pd.DataFrame) -> dict:
    report = {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "rul_shape": rul.shape,
        "train_missing": int(train.isna().sum().sum()),
        "test_missing": int(test.isna().sum().sum()),
        "rul_missing": int(rul.isna().sum().sum()),
    }
    return report


def low_variation_columns(train: pd.DataFrame) -> list[str]:
    setting_cols, sensor_cols = get_setting_and_sensor_columns(train)
    cols = setting_cols + sensor_cols
    return [col for col in cols if train[col].nunique() <= 2]


def scale_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, list[str]]:
    """
    Scale only operating settings and sensor columns.
    Do NOT scale unit_id, time_cycle, or RUL.
    """
    setting_cols, sensor_cols = get_setting_and_sensor_columns(train)
    scale_cols = setting_cols + sensor_cols

    scaler = StandardScaler()

    train_scaled = train.copy()
    test_scaled = test.copy()

    train_scaled[scale_cols] = scaler.fit_transform(train_scaled[scale_cols])
    test_scaled[scale_cols] = scaler.transform(test_scaled[scale_cols])

    return train_scaled, test_scaled, scaler, scale_cols
