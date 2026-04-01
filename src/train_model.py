from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


ModelName = Literal["gradient_boosting", "random_forest"]


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: ModelName = "gradient_boosting",
    random_state: int = 42,
):
    if model_name == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=random_state)
    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model.fit(X_train, y_train)
    return model


def predict_model(model, X: pd.DataFrame):
    return model.predict(X)
