# -*- coding: utf-8 -*-
"""Пересборка отдельных рисунков по сохранённым прогнозам (без обучения моделей)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR
from src.data_loader import correlation_table, load_all
from src.visualization import (
    plot_correlation,
    plot_features_and_target,
    plot_predictions_grid,
    plot_residuals_grid,
)

FIGURE_DIR = RESULTS_DIR / "figures"


def _parts(data, full_prediction: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "train": full_prediction[data.train_index],
        "val": full_prediction[data.val_index],
        "test": full_prediction[data.test_index],
    }


def main() -> None:
    payload = json.loads((RESULTS_DIR / "lab3_results.json").read_text(encoding="utf-8"))
    saved = np.load(RESULTS_DIR / "predictions.npz")
    frame, data_by_horizon, _ = load_all()
    data1 = data_by_horizon[1]
    corr, _, _ = correlation_table(data1)

    plot_features_and_target(frame)
    plot_correlation(corr)

    base = payload["base"]
    single = payload["single_scaled"]
    ensemble = payload["ensemble"]
    neural = payload["neural"]["results"]

    methods = {
        "pair": {"label": base["pair"]["label"], **_parts(data1, saved["pair"])},
        "multiple": {"label": base["multiple"]["label"], **_parts(data1, saved["multiple"])},
        "nonlinear": {"label": base["nonlinear"]["label"], **_parts(data1, saved["nonlinear"])},
    }
    for key in ("knn", "svr", "tree"):
        methods[key] = {
            "label": single[key]["label"],
            **{part: saved[f"single_s_{key}_{part}"] for part in ("train", "val", "test")},
        }
    for key in ("rf", "bagging", "gb"):
        methods[key] = {
            "label": ensemble[key]["label"],
            **{part: saved[f"ens_s_{key}_{part}"] for part in ("train", "val", "test")},
        }
    for key in ("single", "mlp", "deep"):
        methods[f"nn_{key}"] = {
            "label": neural[f"{key}_h1_scaled"]["label"],
            **{part: saved[f"nn_s_h1_{key}_{part}"] for part in ("train", "val", "test")},
        }

    plot_predictions_grid(
        data1,
        methods,
        "fig08_predictions_all_methods.png",
        "Реальные значения и прогноз всех одношаговых моделей (Train / Validation / Test)",
    )
    plot_residuals_grid(
        data1,
        methods,
        "fig09_residuals_all_methods.png",
        "Разность y − ŷ для всех одношаговых моделей (Train / Validation / Test)",
    )
    print("Готово:", FIGURE_DIR)


if __name__ == "__main__":
    main()
