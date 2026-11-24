# -*- coding: utf-8 -*-
"""Одиночные методы машинного обучения: KNN, SVR, дерево решений (п. 2.1–2.3)."""

from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.data_loader import SupervisedData
from src.evaluation import regression_metrics


def _xy(data: SupervisedData, scaled: bool):
    if scaled:
        return data.X_train_s, data.X_val_s, data.X_test_s
    return data.X_train, data.X_val, data.X_test


def _select_by_validation(candidates, X_train, y_train, X_val, y_val):
    """Перебирает кандидатов и выбирает модель с минимальной MSE на валидации."""
    best = None
    log = []
    for params, factory in candidates:
        model = factory()
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        metrics = regression_metrics(y_val, pred)
        log.append({"params": params, "val_mse": metrics["mse"], "val_mae": metrics["mae"]})
        key = (metrics["mse"], metrics["mae"])
        if best is None or key < best[0]:
            best = (key, params, model)
    return best[1], best[2], sorted(log, key=lambda item: item["val_mse"])[:5]


def _predictions(model, data: SupervisedData, scaled: bool) -> dict[str, np.ndarray]:
    X_train, X_val, X_test = _xy(data, scaled)
    return {
        "train": model.predict(X_train),
        "val": model.predict(X_val),
        "test": model.predict(X_test),
    }


def tune_knn(data: SupervisedData, scaled: bool):
    X_train, X_val, _, y_train, y_val, _ = (
        *_xy(data, scaled),
        data.y_train[:, 0],
        data.y_val[:, 0],
        data.y_test[:, 0],
    )
    candidates = []
    for k, weights, p in product([1, 3, 5, 7, 9, 11, 15, 21, 31], ["uniform", "distance"], [1, 2]):
        params = {"n_neighbors": k, "weights": weights, "p": p}
        candidates.append(
            (
                params,
                lambda params=params: KNeighborsRegressor(**params, n_jobs=1),
            )
        )
    best_params, model, log = _select_by_validation(candidates, X_train, y_train, X_val, y_val)
    return best_params, model, log


def tune_svr(data: SupervisedData, scaled: bool):
    X_train, X_val, _, y_train, y_val, _ = (
        *_xy(data, scaled),
        data.y_train[:, 0],
        data.y_val[:, 0],
        data.y_test[:, 0],
    )
    candidates = []
    for C, gamma, epsilon in product(
        [1.0, 10.0, 100.0, 300.0, 1000.0],
        ["scale", 0.01, 0.03, 0.1, 0.3],
        [0.01, 0.05, 0.1],
    ):
        params = {"C": C, "gamma": gamma, "epsilon": epsilon, "kernel": "rbf"}
        candidates.append((params, lambda params=params: SVR(**params)))
    best_params, model, log = _select_by_validation(candidates, X_train, y_train, X_val, y_val)
    return best_params, model, log


def tune_tree(data: SupervisedData):
    X_train, X_val, _, y_train, y_val, _ = (
        *_xy(data, True),
        data.y_train[:, 0],
        data.y_val[:, 0],
        data.y_test[:, 0],
    )
    candidates = []
    for depth, leaf, features in product(
        [3, 5, 7, 9, 12, None], [1, 2, 3, 5], [None, "sqrt"]
    ):
        params = {
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "max_features": features,
            "random_state": 42,
        }
        candidates.append((params, lambda params=params: DecisionTreeRegressor(**params)))
    best_params, model, log = _select_by_validation(candidates, X_train, y_train, X_val, y_val)
    return best_params, model, log


def fit_single_models(data: SupervisedData, scaled: bool = True) -> tuple[dict, dict, dict]:
    """Обучает KNN, SVR и дерево решений, подбирая гиперпараметры на валидации."""
    y_train = data.y_train[:, 0]
    y_val = data.y_val[:, 0]
    y_test = data.y_test[:, 0]

    results: dict[str, dict] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    models: dict[str, object] = {}

    specs = [
        ("knn", "2.1. Метод K ближайших соседей (KNN)", tune_knn),
        ("svr", "2.2. Машина опорных векторов (SVR RBF)", tune_svr),
    ]
    for key, label, tuner in specs:
        best_params, model, log = tuner(data, scaled)
        pred = _predictions(model, data, scaled)
        results[key] = {
            "label": label,
            "params": best_params,
            "metrics": regression_metrics(y_test, pred["test"]),
            "val_metrics": regression_metrics(y_val, pred["val"]),
            "train_metrics": regression_metrics(y_train, pred["train"]),
            "top_trials": log,
        }
        predictions[key] = pred
        models[key] = model

    best_params, model, log = tune_tree(data)
    pred = _predictions(model, data, True)
    results["tree"] = {
        "label": "2.3. Дерево принятия решений (Decision Tree)",
        "params": best_params,
        "metrics": regression_metrics(y_test, pred["test"]),
        "val_metrics": regression_metrics(y_val, pred["val"]),
        "train_metrics": regression_metrics(y_train, pred["train"]),
        "top_trials": log,
    }
    predictions["tree"] = pred
    models["tree"] = model
    return results, predictions, models
