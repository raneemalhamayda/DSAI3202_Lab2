from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression


META_COLS = ["window_id", "unit_id", "time_cycle", "RUL"]


def split_xy(feature_table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = feature_table.drop(columns=[c for c in META_COLS if c in feature_table.columns])
    y = feature_table["RUL"]
    return X, y


def apply_variance_threshold(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    selector = VarianceThreshold(threshold=threshold)
    X_train_sel = selector.fit_transform(X_train)
    X_test_sel = selector.transform(X_test)

    selected_cols = X_train.columns[selector.get_support()].tolist()

    X_train_out = pd.DataFrame(X_train_sel, columns=selected_cols, index=X_train.index)
    X_test_out = pd.DataFrame(X_test_sel, columns=selected_cols, index=X_test.index)

    return X_train_out, X_test_out, selected_cols


def apply_correlation_filter(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.95,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    kept_cols = [c for c in X_train.columns if c not in to_drop]

    return X_train[kept_cols].copy(), X_test[kept_cols].copy(), kept_cols, to_drop


def apply_mutual_information(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    top_k: int = 50,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    mi = mutual_info_regression(X_train, y_train, random_state=random_state)
    mi_series = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)

    selected_cols = mi_series.head(min(top_k, len(mi_series))).index.tolist()

    return X_train[selected_cols].copy(), X_test[selected_cols].copy(), mi_series, selected_cols
