from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred, squared=False))


def save_selected_features(features: list[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for feat in features:
            f.write(f"{feat}\n")


def save_metrics(metrics: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_results_table(results: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([results])
    df.to_csv(output_path, index=False)
