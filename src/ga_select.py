from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def _safe_register_creators() -> None:
    if not hasattr(creator, "FitnessMinLab5"):
        creator.create("FitnessMinLab5", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualLab5"):
        creator.create("IndividualLab5", list, fitness=creator.FitnessMinLab5)


def select_features_ga(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    population_size: int = 20,
    generations: int = 10,
    cxpb: float = 0.7,
    mutpb: float = 0.2,
    random_state: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Fitness = RMSE + alpha * feature_ratio
    Lower is better.
    """
    _safe_register_creators()
    random.seed(random_state)
    np.random.seed(random_state)

    n_features = X_train.shape[1]
    feature_names = X_train.columns.tolist()

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.IndividualLab5,
        toolbox.attr_bool,
        n=n_features,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    cache: dict[tuple[int, ...], tuple[float]] = {}

    def evaluate(individual: list[int]) -> tuple[float]:
        key = tuple(individual)
        if key in cache:
            return cache[key]

        selected_idx = [i for i, bit in enumerate(individual) if bit == 1]

        if len(selected_idx) == 0:
            fitness = (1e9,)
            cache[key] = fitness
            return fitness

        selected_cols = [feature_names[i] for i in selected_idx]

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train[selected_cols], y_train)
        preds = model.predict(X_val[selected_cols])

        rmse = mean_squared_error(y_val, preds, squared=False)
        feature_ratio = len(selected_cols) / n_features
        score = rmse + alpha * feature_ratio

        fitness = (float(score),)
        cache[key] = fitness
        return fitness

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )

    best_individual = hof[0]
    best_idx = [i for i, bit in enumerate(best_individual) if bit == 1]
    best_features = [feature_names[i] for i in best_idx]
    best_score = float(best_individual.fitness.values[0])

    return {
        "best_features": best_features,
        "best_score": best_score,
        "num_selected_features": len(best_features),
        "population_size": population_size,
        "generations": generations,
    }
