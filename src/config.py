# -*- coding: utf-8 -*-
"""Конфигурация лабораторной работы № 3 (вариант 4)."""

from __future__ import annotations

from pathlib import Path

SEED = 42
VARIANT = 4
HORIZONS = (1, 7)

# Вариант 4 занимает столбцы 19–24 книги Excel (нумерация с единицы):
# a, b, c, d, e, y.
BLOCK_START = (VARIANT - 1) * 6
BLOCK_END = BLOCK_START + 6
FEATURES = ["a", "b", "c", "d", "e"]
TARGET = "y"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATA_FILE = DATA_DIR / "Lab3_Dataset.xlsx"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_DIR = ROOT_DIR / "report"

# Разбиение выборки: 70 % / 15 % / 15 %.
# Перемешивание фиксировано seed = 42; внутри каждой части сохраняется
# исходный временной порядок (нужен только для графиков).
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
SHUFFLE_SPLIT = True

# Нейронные сети
NN_EPOCHS = 220
NN_BATCH_SIZE = 64
NN_PATIENCE = 35
NN_LEARNING_RATE = 0.01
NN_WEIGHT_DECAY = 1e-5

# Оформление графиков
PLOT_DPI = 160
FIGSIZE_WIDE = (12.0, 5.4)
FIGSIZE_PANEL = (13.5, 4.2)

for _directory in (RESULTS_DIR, FIGURES_DIR, REPORT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)
