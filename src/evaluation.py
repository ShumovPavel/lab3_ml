# -*- coding: utf-8 -*-
"""Метрики качества регрессии и экспорт результатов."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import RESULTS_DIR


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, MSE, RMSE, коэффициент корреляции Пирсона R и R²."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        corr = 0.0
    else:
        corr = float(pearsonr(y_true, y_pred)[0])
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "mse": mse, "rmse": rmse, "r": corr, "r2": r2}


def horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float]]:
    """Метрики отдельно для каждого шага многошагового прогноза."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    rows = []
    for step in range(y_true.shape[1]):
        metrics = regression_metrics(y_true[:, step], y_pred[:, step])
        rows.append({"step": step + 1, **metrics})
    return rows


def add_method(
    storage: dict,
    key: str,
    label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extra: dict | None = None,
) -> dict:
    """Добавляет результаты метода в общий словарь."""
    metrics = regression_metrics(y_true, y_pred)
    record = {"key": key, "label": label, **metrics}
    if extra:
        record.update(extra)
    storage[key] = record
    return record


def comparison_frame(results: dict[str, dict]) -> pd.DataFrame:
    """Формирует таблицу сравнения методов."""
    rows = []
    for record in results.values():
        rows.append(
            {
                "Метод": record["label"],
                "MAE": record["mae"],
                "MSE": record["mse"],
                "RMSE": record["rmse"],
                "R": record["r"],
                "R2": record["r2"],
            }
        )
    return pd.DataFrame(rows)


def export_comparison(
    scaled: dict[str, dict],
    unscaled: dict[str, dict],
    horizon_results: dict[str, dict] | None = None,
    path: Path | None = None,
) -> Path:
    """Экспортирует results_Comparison.xlsx с тремя листами."""
    path = path or RESULTS_DIR / "results_Comparison.xlsx"
    frame_scaled = comparison_frame(scaled)
    frame_unscaled = comparison_frame(unscaled)

    combined = pd.DataFrame(
        {
            "Метод": frame_scaled["Метод"],
            "MAE без масшт.": frame_unscaled["MAE"].values,
            "MSE без масшт.": frame_unscaled["MSE"].values,
            "R без масшт.": frame_unscaled["R"].values,
            "MAE со станд.": frame_scaled["MAE"].values,
            "MSE со станд.": frame_scaled["MSE"].values,
            "R со станд.": frame_scaled["R"].values,
        }
    )
    delta = np.where(
        frame_unscaled["MAE"].values > 0,
        (frame_unscaled["MAE"].values - frame_scaled["MAE"].values)
        / frame_unscaled["MAE"].values
        * 100.0,
        np.nan,
    )
    combined["Снижение MAE, %"] = delta

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame_unscaled.to_excel(writer, sheet_name="Без масштабирования", index=False)
        frame_scaled.to_excel(writer, sheet_name="Со стандартизацией", index=False)
        combined.to_excel(writer, sheet_name="Сравнение", index=False)
        if horizon_results:
            rows = []
            for key, record in horizon_results.items():
                rows.append(
                    {
                        "Модель": record["label"],
                        "Горизонт": record.get("horizon", 7),
                        "MAE": record["mae"],
                        "MSE": record["mse"],
                        "R": record["r"],
                        "Параметров": record.get("n_params", ""),
                    }
                )
            pd.DataFrame(rows).to_excel(writer, sheet_name="Многошаговый прогноз", index=False)

    return path
