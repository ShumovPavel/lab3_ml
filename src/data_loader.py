# -*- coding: utf-8 -*-
"""Загрузка датасета варианта 4 и формирование выборок."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    BLOCK_END,
    BLOCK_START,
    DATA_FILE,
    FEATURES,
    HORIZONS,
    SEED,
    SHUFFLE_SPLIT,
    TARGET,
    TRAIN_FRACTION,
    VAL_FRACTION,
)


@dataclass
class SupervisedData:
    """Выборки для одного горизонта прогнозирования."""

    horizon: int
    X: np.ndarray
    y: np.ndarray
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_index: np.ndarray
    val_index: np.ndarray
    test_index: np.ndarray
    scaler_x: StandardScaler
    scaler_y: StandardScaler
    X_train_s: np.ndarray = field(init=False)
    X_val_s: np.ndarray = field(init=False)
    X_test_s: np.ndarray = field(init=False)
    y_train_s: np.ndarray = field(init=False)
    y_val_s: np.ndarray = field(init=False)
    y_test_s: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.X_train_s = self.scaler_x.transform(self.X_train)
        self.X_val_s = self.scaler_x.transform(self.X_val)
        self.X_test_s = self.scaler_x.transform(self.X_test)
        self.y_train_s = self.scaler_y.transform(self.y_train)
        self.y_val_s = self.scaler_y.transform(self.y_val)
        self.y_test_s = self.scaler_y.transform(self.y_test)

    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def splits(self) -> tuple[int, int, int]:
        return len(self.X_train), len(self.X_val), len(self.X_test)


def load_variant_frame() -> pd.DataFrame:
    """Читает книгу Excel и возвращает блок варианта 4."""
    raw = pd.read_excel(DATA_FILE, header=None)
    block = raw.iloc[2:, BLOCK_START:BLOCK_END].astype(float).reset_index(drop=True)
    block.columns = FEATURES + [TARGET]
    return block


def build_supervised(frame: pd.DataFrame, horizon: int) -> SupervisedData:
    """Формирует пары X(t) -> y(t+1..t+horizon) без утечки будущего."""
    values = frame[FEATURES].to_numpy(dtype=float)
    target = frame[TARGET].to_numpy(dtype=float)

    n = len(frame) - horizon
    X = values[:n]
    if horizon == 1:
        y = target[1 : n + 1].reshape(-1, 1)
    else:
        y = np.column_stack([target[k : k + n] for k in range(1, horizon + 1)])

    n_train = int(np.floor(n * TRAIN_FRACTION))
    n_val = int(np.floor(n * VAL_FRACTION))
    n_test = n - n_train - n_val

    if SHUFFLE_SPLIT:
        rng = np.random.default_rng(SEED)
        order = rng.permutation(n)
        train_idx = np.sort(order[:n_train])
        val_idx = np.sort(order[n_train : n_train + n_val])
        test_idx = np.sort(order[n_train + n_val :])
    else:
        train_idx = np.arange(0, n_train)
        val_idx = np.arange(n_train, n_train + n_val)
        test_idx = np.arange(n_train + n_val, n)

    scaler_x = StandardScaler().fit(X[train_idx])
    scaler_y = StandardScaler().fit(y[train_idx])

    return SupervisedData(
        horizon=horizon,
        X=X,
        y=y,
        X_train=X[train_idx],
        X_val=X[val_idx],
        X_test=X[test_idx],
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx],
        train_index=train_idx,
        val_index=val_idx,
        test_index=test_idx,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
    )


def correlation_table(data: SupervisedData) -> tuple[pd.DataFrame, dict[str, float], str]:
    """Матрица корреляций признаков и цели y(t+1)."""
    matrix = np.column_stack([data.X, data.y[:, 0]])
    columns = FEATURES + ["y(t+1)"]
    corr = pd.DataFrame(matrix, columns=columns).corr()

    target_corr = {feature: float(corr.loc[feature, "y(t+1)"]) for feature in FEATURES}
    best = max(target_corr, key=lambda name: abs(target_corr[name]))
    return corr, target_corr, best


def dataset_summary(frame: pd.DataFrame, data_by_horizon: dict[int, SupervisedData]) -> dict[str, Any]:
    """Сводная информация об исходных данных и выборках."""
    summary: dict[str, Any] = {
        "file": DATA_FILE.name,
        "variant": 4,
        "columns": BLOCK_START + 1,
        "n_rows": int(len(frame)),
        "n_features": len(FEATURES),
        "features": FEATURES,
        "missing": {column: int(frame[column].isna().sum()) for column in frame.columns},
        "duplicates": int(frame.duplicated().sum()),
        "statistics": {
            column: {
                "mean": float(frame[column].mean()),
                "std": float(frame[column].std()),
                "min": float(frame[column].min()),
                "max": float(frame[column].max()),
            }
            for column in frame.columns
        },
        "horizons": {},
    }
    for horizon, data in data_by_horizon.items():
        train, val, test = data.splits
        summary["horizons"][str(horizon)] = {
            "n_samples": data.n_samples,
            "train": train,
            "val": val,
            "test": test,
            "target_shape": list(data.y.shape),
        }
    return summary


def load_all() -> tuple[pd.DataFrame, dict[int, SupervisedData], dict[str, Any]]:
    """Загружает данные для всех горизонтов и сохраняет сводку."""
    frame = load_variant_frame()
    data_by_horizon = {horizon: build_supervised(frame, horizon) for horizon in HORIZONS}
    summary = dataset_summary(frame, data_by_horizon)
    return frame, data_by_horizon, summary


def set_seed(seed: int = SEED) -> None:
    """Фиксирует генераторы случайных чисел."""
    import random

    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    set_seed()
    frame, data_by_horizon, summary = load_all()
    print(frame.head())
    print(summary["horizons"])
    for horizon, data in data_by_horizon.items():
        corr, target_corr, best = correlation_table(data)
        print(f"Горизонт {horizon}: {data.splits}, лучший признак {best} {target_corr[best]:.4f}")
