# -*- coding: utf-8 -*-
"""Парная, множественная и нелинейная регрессия (п. 1.4–1.6)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from src.data_loader import SupervisedData
from src.evaluation import regression_metrics

POLY_NAMES = (
    ["a", "b", "c", "d", "e"]
    + ["a^2", "b^2", "c^2", "d^2", "e^2"]
    + [
        "a*b",
        "a*c",
        "a*d",
        "a*e",
        "b*c",
        "b*d",
        "b*e",
        "c*d",
        "c*e",
        "d*e",
    ]
)


def poly2_features(X: np.ndarray) -> np.ndarray:
    """Полный квадратичный базис: 5 линейных, 5 квадратов и 10 парных членов."""
    X = np.asarray(X, dtype=float)
    a, b, c, d, e = X.T
    return np.column_stack(
        [
            a,
            b,
            c,
            d,
            e,
            a * a,
            b * b,
            c * c,
            d * d,
            e * e,
            a * b,
            a * c,
            a * d,
            a * e,
            b * c,
            b * d,
            b * e,
            c * d,
            c * e,
            d * e,
        ]
    )


def _nonlinear_function(X, *params):
    """Квадратичная аппроксимирующая функция для curve_fit."""
    X = np.asarray(X, dtype=float)
    return params[0] + poly2_features(X) @ np.asarray(params[1:])


def _predict_nonlinear(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    return params[0] + poly2_features(X) @ np.asarray(params[1:])


def _fit_curve(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, bool, str]:
    """Подгонка curve_fit с начальным приближением МНК."""
    design = np.column_stack([np.ones(len(X)), poly2_features(X)])
    p0, *_ = np.linalg.lstsq(design, y, rcond=None)
    try:
        popt, _ = curve_fit(
            _nonlinear_function,
            X,
            y,
            p0=p0,
            maxfev=60000,
            method="trf",
            x_scale="jac",
        )
        return popt, True, "curve_fit (trf, x_scale=jac)"
    except Exception as exc:  # pragma: no cover - резервный вариант
        return p0, False, f"МНК-инициализация (curve_fit не сошёлся: {exc})"


def fit_base_models(data: SupervisedData) -> tuple[dict, dict[str, np.ndarray], dict[str, object]]:
    """Обучает три базовые модели и возвращает результаты, прогнозы и модели."""
    X_train, y_train = data.X_train, data.y_train[:, 0]
    train_corr = {
        name: abs(float(np.corrcoef(X_train[:, i], y_train)[0, 1]))
        for i, name in enumerate("abcde")
    }
    best_feature = max(train_corr, key=train_corr.get)
    best_index = "abcde".index(best_feature)

    # 1.4. Парная линейная регрессия
    pair = LinearRegression().fit(X_train[:, [best_index]], y_train)
    pair_pred = pair.predict(data.X[:, [best_index]])
    pair_test = pair.predict(data.X_test[:, [best_index]])

    # 1.5. Множественная линейная регрессия
    multiple = LinearRegression().fit(X_train, y_train)
    multiple_pred = multiple.predict(data.X)
    multiple_test = multiple.predict(data.X_test)

    # 1.6. Нелинейная регрессия curve_fit
    params, converged, optimizer = _fit_curve(X_train, y_train)
    nonlinear_pred = _predict_nonlinear(data.X, params)
    nonlinear_test = _predict_nonlinear(data.X_test, params)

    results = {
        "pair": {
            "label": "1.4. Парная линейная регрессия",
            "feature": best_feature,
            "metrics": regression_metrics(data.y_test[:, 0], pair_test),
            "train_metrics": regression_metrics(y_train, pair.predict(X_train[:, [best_index]])),
            "coefficient": float(pair.coef_[0]),
            "intercept": float(pair.intercept_),
            "train_correlation": train_corr,
        },
        "multiple": {
            "label": "1.5. Множественная линейная регрессия",
            "metrics": regression_metrics(data.y_test[:, 0], multiple_test),
            "train_metrics": regression_metrics(y_train, multiple.predict(X_train)),
            "coefficients": dict(zip("abcde", map(float, multiple.coef_))),
            "intercept": float(multiple.intercept_),
        },
        "nonlinear": {
            "label": "1.6. Нелинейная регрессия (SciPy curve_fit)",
            "metrics": regression_metrics(data.y_test[:, 0], nonlinear_test),
            "train_metrics": regression_metrics(y_train, _predict_nonlinear(X_train, params)),
            "formula": "y = p0 + sum(pi*xi) + sum(pii*xi^2) + sum(pij*xi*xj)",
            "terms": POLY_NAMES,
            "params": [float(value) for value in params],
            "n_params": int(len(params)),
            "converged": bool(converged),
            "optimizer": optimizer,
        },
    }
    predictions = {
        "pair": pair_pred,
        "multiple": multiple_pred,
        "nonlinear": nonlinear_pred,
    }
    models = {"pair": pair, "multiple": multiple, "nonlinear_params": params}
    return results, predictions, models


def pair_curve(X: np.ndarray, model: LinearRegression) -> np.ndarray:
    """Линия парной регрессии для графика."""
    grid = np.linspace(float(X.min()), float(X.max()), 300)
    return grid, model.predict(grid.reshape(-1, 1))
