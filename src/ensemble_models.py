# -*- coding: utf-8 -*-
"""Ансамблевые методы: Random Forest, Bagging, Gradient Boosting (п. 3.1–3.3)."""

from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from src.data_loader import SupervisedData
from src.evaluation import regression_metrics


def _select(candidates, X_train, y_train, X_val, y_val):
    best = None
    log = []
    for params, factory in candidates:
        model = factory()
        model.fit(X_train, y_train)
        metrics = regression_metrics(y_val, model.predict(X_val))
        log.append({"params": params, "val_mse": metrics["mse"], "val_mae": metrics["mae"]})
        key = (metrics["mse"], metrics["mae"])
        if best is None or key < best[0]:
            best = (key, params, model)
    return best[1], best[2], sorted(log, key=lambda item: item["val_mse"])[:5]


def _predict(model, X_train, X_val, X_test):
    return {
        "train": model.predict(X_train),
        "val": model.predict(X_val),
        "test": model.predict(X_test),
    }


def _tune_rf(data: SupervisedData):
    X_train, X_val = data.X_train_s, data.X_val_s
    y_train, y_val = data.y_train[:, 0], data.y_val[:, 0]
    candidates = []
    for depth, features, leaf in product([5, 7, 9, 12], [3, 4, 5], [1, 2]):
        params = {
            "n_estimators": 250,
            "max_depth": depth,
            "max_features": features,
            "min_samples_leaf": leaf,
            "random_state": 42,
            "n_jobs": 1,
        }
        candidates.append((params, lambda params=params: RandomForestRegressor(**params)))
    return _select(candidates, X_train, y_train, X_val, y_val)


def _tune_bagging(data: SupervisedData):
    X_train, X_val = data.X_train_s, data.X_val_s
    y_train, y_val = data.y_train[:, 0], data.y_val[:, 0]
    candidates = []
    for depth, features, leaf in product([5, 7, 9, None], [0.6, 0.8, 1.0], [1, 2]):
        params = {
            "n_estimators": 180,
            "max_samples": 0.8,
            "max_features": features,
            "bootstrap": True,
            "bootstrap_features": False,
            "random_state": 42,
            "n_jobs": 1,
            "estimator__max_depth": depth,
            "estimator__min_samples_leaf": leaf,
        }
        candidates.append(
            (
                params,
                lambda params=params: BaggingRegressor(
                    estimator=DecisionTreeRegressor(
                        max_depth=params["estimator__max_depth"],
                        min_samples_leaf=params["estimator__min_samples_leaf"],
                        random_state=42,
                    ),
                    n_estimators=params["n_estimators"],
                    max_samples=params["max_samples"],
                    max_features=params["max_features"],
                    bootstrap=params["bootstrap"],
                    bootstrap_features=params["bootstrap_features"],
                    random_state=params["random_state"],
                    n_jobs=params["n_jobs"],
                ),
            )
        )
    return _select(candidates, X_train, y_train, X_val, y_val)


def _tune_gb(data: SupervisedData):
    X_train, X_val = data.X_train_s, data.X_val_s
    y_train, y_val = data.y_train[:, 0], data.y_val[:, 0]
    candidates = []
    for n_estimators, learning_rate, depth, subsample in product(
        [120, 220], [0.05, 0.1], [2, 3, 4], [0.8, 1.0]
    ):
        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": depth,
            "subsample": subsample,
            "random_state": 42,
        }
        candidates.append((params, lambda params=params: GradientBoostingRegressor(**params)))
    return _select(candidates, X_train, y_train, X_val, y_val)


def _clone_params(params: dict) -> dict:
    cleaned = {k: v for k, v in params.items() if not k.startswith("estimator__")}
    return cleaned


def _build(key: str, params: dict):
    if key == "rf":
        return RandomForestRegressor(**_clone_params(params))
    if key == "bagging":
        return BaggingRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=params["estimator__max_depth"],
                min_samples_leaf=params["estimator__min_samples_leaf"],
                random_state=42,
            ),
            **_clone_params(params),
        )
    return GradientBoostingRegressor(**_clone_params(params))


def fit_ensemble_models(data: SupervisedData) -> tuple[dict, dict, dict, dict]:
    """Подбирает гиперпараметры и обучает три ансамбля в двух режимах."""
    y_train = data.y_train[:, 0]
    y_val = data.y_val[:, 0]
    y_test = data.y_test[:, 0]

    results: dict[str, dict] = {}
    predictions_scaled: dict[str, dict[str, np.ndarray]] = {}
    predictions_raw: dict[str, dict[str, np.ndarray]] = {}
    models: dict[str, object] = {}

    for key, label, tuner in [
        ("rf", "3.1. Случайный лес (Random Forest)", _tune_rf),
        ("bagging", "3.2. Бэггинг (Bagging)", _tune_bagging),
        ("gb", "3.3. Градиентный бустинг (Gradient Boosting)", _tune_gb),
    ]:
        best_params, model_scaled, log = tuner(data)
        model_raw = _build(key, best_params).fit(data.X_train, y_train)

        pred_s = _predict(model_scaled, data.X_train_s, data.X_val_s, data.X_test_s)
        pred_r = _predict(model_raw, data.X_train, data.X_val, data.X_test)

        results[key] = {
            "label": label,
            "params": dict(best_params),
            "metrics_scaled": regression_metrics(y_test, pred_s["test"]),
            "metrics_raw": regression_metrics(y_test, pred_r["test"]),
            "val_metrics_scaled": regression_metrics(y_val, pred_s["val"]),
            "val_metrics_raw": regression_metrics(y_val, pred_r["val"]),
            "train_metrics_scaled": regression_metrics(y_train, pred_s["train"]),
            "top_trials": log,
        }
        predictions_scaled[key] = pred_s
        predictions_raw[key] = pred_r
        models[key] = model_scaled

        if hasattr(model_scaled, "feature_importances_"):
            results[key]["feature_importance"] = {
                name: float(value)
                for name, value in zip("abcde", model_scaled.feature_importances_)
            }

    return results, predictions_scaled, predictions_raw, models
