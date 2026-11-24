# -*- coding: utf-8 -*-
"""Построение и сохранение всех графиков отчёта."""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram
from sklearn.tree import DecisionTreeRegressor

from src.config import FEATURES, FIGURES_DIR, PLOT_DPI

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.unicode_minus": False,
        "figure.autolayout": False,
    }
)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#111111"]


def _save(fig, filename: str) -> Path:
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_features_and_target(frame: pd.DataFrame) -> Path:
    """Рис. 1. Все признаки и целевая переменная на едином графике."""
    fig, (ax, ax_noise) = plt.subplots(
        2,
        1,
        figsize=(13.0, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [4.0, 1.15], "hspace": 0.12},
    )
    columns = FEATURES + ["y"]
    standardized = (frame[columns] - frame[columns].mean()) / frame[columns].std()
    for column, color in zip(["a", "b", "c", "e"], [COLORS[0], COLORS[1], COLORS[2], COLORS[4]]):
        ax.plot(
            standardized.index,
            standardized[column],
            label=column,
            color=color,
            linewidth=1.5,
            zorder=2,
        )
    ax.plot(
        standardized.index,
        standardized["y"],
        label="y (цель)",
        color=COLORS[5],
        linewidth=2.1,
        linestyle="--",
        zorder=3,
    )
    ax.set_title("Признаки a–e и целевая переменная y (стандартизованные значения)")
    ax.set_ylabel("Стандартизованное значение")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)

    ax_noise.plot(
        standardized.index,
        standardized["d"],
        label="d (шумовой признак)",
        color=COLORS[3],
        linewidth=0.35,
        alpha=0.55,
    )
    ax_noise.set_ylabel("d")
    ax_noise.set_xlabel("Номер наблюдения t")
    ax_noise.legend(loc="upper right", frameon=False)
    ax_noise.set_xlim(0, len(frame) - 1)
    return _save(fig, "fig01_features_target.png")


def plot_correlation(corr: pd.DataFrame) -> Path:
    """Рис. 2. Матрица взаимных корреляций."""
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = list(corr.columns)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.values[i, j]
            ax.text(
                j,
                i,
                f"{value:.3f}".replace(".", ","),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.6 else "black",
            )
    ax.set_title("Матрица коэффициентов корреляции Пирсона")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="r")
    colorbar.ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda value, _: f"{value:.2f}".replace(".", ","))
    )
    ax.grid(False)
    return _save(fig, "fig02_correlation.png")


def plot_base_regressions(data, base_results, predictions, models) -> Path:
    """Рис. 3. Исходные данные, аппроксимирующая функция и прогноз (п. 5.2)."""
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.4))
    feature = base_results["pair"]["feature"]
    index = "abcde".index(feature)
    X_test = data.X_test[:, index]
    y_test = data.y_test[:, 0]
    order = np.argsort(X_test)

    axes[0].scatter(X_test, y_test, s=18, color="#1f77b4", alpha=0.75, label="Исходные данные (test)")
    model = models["pair"]
    grid = np.linspace(X_test.min(), X_test.max(), 300).reshape(-1, 1)
    axes[0].plot(grid[:, 0], model.predict(grid), color="#d62728", linewidth=2, label="Парная регрессия")
    axes[0].set_xlabel(f"Признак {feature}")
    axes[0].set_ylabel("y(t+1)")
    axes[0].set_title(f"Парная линейная регрессия: {feature}")
    axes[0].legend()

    for ax, key, title in [
        (axes[1], "multiple", "Множественная линейная регрессия"),
        (axes[2], "nonlinear", "Нелинейная регрессия (curve_fit)"),
    ]:
        pred = predictions[key]
        test_pred = pred[data.test_index]
        ax.plot(np.arange(len(order)), y_test[order], color="#1f77b4", linewidth=1.6, label="Реальные значения")
        ax.plot(
            np.arange(len(order)),
            test_pred[order],
            color="#d62728",
            linewidth=1.6,
            linestyle="--",
            label="Прогноз",
        )
        ax.set_xlabel("Наблюдения тестовой выборки (упорядочены по времени)")
        ax.set_ylabel("y(t+1)")
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Базовые регрессионные модели: данные и прогноз", fontsize=12)
    fig.tight_layout()
    return _save(fig, "fig03_base_regressions.png")


def _prune_tree(tree, node: int, depth: int, max_depth: int, names: list[str]):
    left = tree.children_left[node]
    right = tree.children_right[node]
    count = int(tree.n_node_samples[node])
    feature = int(tree.feature[node]) if tree.feature[node] >= 0 else -1
    threshold = float(tree.threshold[node]) if feature >= 0 else 0.0
    if depth >= max_depth or left == -1:
        if feature >= 0:
            label = f"{names[feature]} ≤ {threshold:.2f} (n={count})"
        else:
            label = f"лист, n={count}"
        return ("leaf", label, count)
    return (
        "node",
        _prune_tree(tree, left, depth + 1, max_depth, names),
        _prune_tree(tree, right, depth + 1, max_depth, names),
        count,
    )


def _count_leaves(node) -> int:
    if node[0] == "leaf":
        return 1
    return _count_leaves(node[1]) + _count_leaves(node[2])


def plot_tree_dendrogram(model: DecisionTreeRegressor, filename: str = "fig04_tree_dendrogram.png") -> Path:
    """Рис. 4. Дендрограмма верхних ярусов дерева решений (п. 5.3)."""
    tree = model.tree_
    max_depth = min(3, int(tree.max_depth))
    pruned = _prune_tree(tree, 0, 0, max_depth, FEATURES)
    n_leaves = _count_leaves(pruned)
    linkage_rows: list[list[float]] = []
    labels: list[str] = []
    counter = {"cluster": n_leaves}
    counts: dict[int, int] = {}

    def build(node, depth: int) -> int:
        if node[0] == "leaf":
            labels.append(node[1])
            leaf_id = len(labels) - 1
            counts[leaf_id] = 1
            return leaf_id
        left_id = build(node[1], depth + 1)
        right_id = build(node[2], depth + 1)
        cluster_id = counter["cluster"]
        counter["cluster"] += 1
        height = float(max_depth - depth + 1)
        count = counts[left_id] + counts[right_id]
        counts[cluster_id] = count
        linkage_rows.append([left_id, right_id, height, float(count)])
        return cluster_id

    build(pruned, 0)
    linkage = np.asarray(linkage_rows, dtype=float)

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    dendrogram(
        linkage,
        labels=labels,
        ax=ax,
        leaf_rotation=20,
        leaf_font_size=7.5,
        color_threshold=0.7 * linkage[:, 2].max(),
        above_threshold_color="#444444",
    )
    ax.set_title("Дендрограмма верхних ярусов дерева решений")
    ax.set_ylabel("Уровень ветвления")
    ax.set_xlabel("Условия разбиения / укрупнённые ветви")
    return _save(fig, filename)


def plot_learning_curves(histories: dict[str, dict], horizon: int) -> Path:
    """Рис. 5–6. Кривые обучения MSE и MAE от эпохи."""
    keys = [f"single_h{horizon}", f"mlp_h{horizon}", f"deep_h{horizon}"]
    titles = ["Однослойный персептрон", "Многослойный персептрон", "ResNet-MLP"]
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 7.2))
    for column, (key, title) in enumerate(zip(keys, titles)):
        history = histories[key]
        epochs = history["epoch"]
        axes[0, column].plot(epochs, history["train_mse"], label="Train", color="#1f77b4")
        axes[0, column].plot(epochs, history["val_mse"], label="Validation", color="#d62728")
        axes[0, column].set_title(f"{title}: MSE")
        axes[0, column].set_xlabel("Эпоха")
        axes[0, column].set_ylabel("MSE")
        axes[0, column].legend()
        axes[0, column].set_yscale("log")

        axes[1, column].plot(epochs, history["train_mae"], label="Train", color="#1f77b4")
        axes[1, column].plot(epochs, history["val_mae"], label="Validation", color="#d62728")
        axes[1, column].set_title(f"{title}: MAE")
        axes[1, column].set_xlabel("Эпоха")
        axes[1, column].set_ylabel("MAE")
        axes[1, column].legend()
        axes[1, column].set_yscale("log")
    fig.suptitle(f"Кривые обучения нейронных сетей, горизонт {horizon} шаг(ов)", fontsize=13)
    fig.tight_layout()
    return _save(fig, f"fig{'06' if horizon == 1 else '07'}_nn_learning_curves_{horizon}step.png")


def _full_series(data, predictions: dict[str, np.ndarray]):
    """Собирает прогноз в исходном временном порядке и маску разбиения."""
    actual = data.y[:, 0]
    pred = np.full(len(actual), np.nan)
    for part, indices in [
        ("train", data.train_index),
        ("val", data.val_index),
        ("test", data.test_index),
    ]:
        values = np.asarray(predictions[part]).reshape(-1)
        pred[indices] = values
    return actual, pred


def _draw_split_strip(ax, data, y_position: float) -> None:
    """Цветная полоса принадлежности наблюдений к Train / Validation / Test."""
    for indices, color in [
        (data.train_index, "#9ecae1"),
        (data.val_index, "#fdae6b"),
        (data.test_index, "#74c476"),
    ]:
        ax.scatter(
            indices,
            np.full(len(indices), y_position),
            c=color,
            s=3.5,
            marker="s",
            linewidths=0,
            clip_on=False,
            zorder=5,
        )


def _prediction_parts_from_full(data, full_prediction: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "train": full_prediction[data.train_index],
        "val": full_prediction[data.val_index],
        "test": full_prediction[data.test_index],
    }


def plot_predictions_grid(data, methods: dict, filename: str, title: str) -> Path:
    """Рис. 9. Реальные значения и прогноз для всех методов (п. 5.1)."""
    keys = list(methods)
    fig, axes = plt.subplots(4, 3, figsize=(17.0, 9.4), sharex=True, sharey=True)
    actual, _ = _full_series(data, methods[keys[0]])
    x = np.arange(len(actual))
    for ax, key in zip(axes.ravel(), keys):
        actual, pred = _full_series(data, methods[key])
        ax.plot(x, actual, color="#222222", linewidth=1.05, label="Реальные")
        ax.plot(
            x,
            pred,
            color="#d62728",
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
            label="Прогноз",
        )
        ax.set_title(methods[key]["label"], fontsize=9)
        ax.tick_params(labelsize=7)
        if ax is axes.ravel()[0]:
            ymin, ymax = ax.get_ylim()
            offset = 0.04 * (ymax - ymin)
            _draw_split_strip(ax, data, ymin - offset)
            ax.set_ylim(ymin - 2 * offset, ymax)
    handles = [
        Patch(color="#9ecae1", label="Train"),
        Patch(color="#fdae6b", label="Validation"),
        Patch(color="#74c476", label="Test"),
        plt.Line2D([], [], color="#333333", label="Реальные"),
        plt.Line2D([], [], color="#d62728", label="Прогноз"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    return _save(fig, filename)


def plot_residuals_grid(data, methods: dict, filename: str, title: str) -> Path:
    """Рис. 10. Разность y − ŷ для всех методов (п. 5.1)."""
    keys = list(methods)
    fig, axes = plt.subplots(4, 3, figsize=(17.0, 9.4), sharex=True, sharey=True)
    x = np.arange(len(data.y))
    for ax, key in zip(axes.ravel(), keys):
        actual, pred = _full_series(data, methods[key])
        ax.plot(x, actual - pred, color="#1f77b4", linewidth=0.8)
        ax.axhline(0, color="#d62728", linewidth=1.0)
        ax.set_title(methods[key]["label"], fontsize=9)
        ax.tick_params(labelsize=7)
        if ax is axes.ravel()[0]:
            ymin, ymax = ax.get_ylim()
            offset = 0.05 * (ymax - ymin)
            _draw_split_strip(ax, data, ymin - offset)
            ax.set_ylim(ymin - 2 * offset, ymax)
    handles = [
        Patch(color="#9ecae1", label="Train"),
        Patch(color="#fdae6b", label="Validation"),
        Patch(color="#74c476", label="Test"),
        plt.Line2D([], [], color="#1f77b4", label="Ошибка y − ŷ"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    return _save(fig, filename)


def plot_model_comparison(scaled: dict[str, dict], unscaled: dict[str, dict]) -> Path:
    """Рис. 10. Сравнение MAE методов со стандартизацией и без неё."""
    labels = [record["label"] for record in scaled.values()]
    mae_scaled = [record["mae"] for record in scaled.values()]
    mae_raw = [record["mae"] for record in unscaled.values()]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    ax.barh(y - 0.19, mae_raw, height=0.36, color="#9ecae1", label="Без масштабирования")
    ax.barh(y + 0.19, mae_scaled, height=0.36, color="#d62728", label="StandardScaler")
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("MAE на тестовой выборке (логарифмическая шкала)")
    ax.set_title("Сравнение методов по MAE: без масштабирования и со стандартизацией")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "fig10_model_comparison.png")


def plot_feature_importance(rf: dict[str, float], gb: dict[str, float]) -> Path:
    """Рис. 11. Важность признаков в ансамблях."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    for ax, values, title, color in [
        (axes[0], rf, "Random Forest", "#2ca02c"),
        (axes[1], gb, "Gradient Boosting", "#ff7f0e"),
    ]:
        names = list(values)
        importances = [values[name] for name in names]
        order = np.argsort(importances)
        ax.barh(np.array(names)[order], np.array(importances)[order], color=color)
        ax.set_title(f"Важность признаков: {title}")
        ax.set_xlabel("Относительная важность")
    fig.tight_layout()
    return _save(fig, "fig11_feature_importance.png")


def plot_multistep_examples(data, predictions: dict[str, dict[str, np.ndarray]]) -> Path:
    """Рис. 12. Примеры многошагового прогноза на 7 шагов вперёд."""
    archs = ["single", "mlp", "deep"]
    titles = ["Однослойный персептрон", "MLP", "ResNet-MLP"]
    row_indices = [10, 50, 100]
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 10.0))
    for column, (arch, title) in enumerate(zip(archs, titles)):
        pred = predictions[arch]["test"]
        actual = data.y_test
        for row, sample in enumerate(row_indices):
            ax = axes[row, column]
            ax.plot(range(1, 8), actual[sample], "o-", color="#333333", label="Реальные")
            ax.plot(range(1, 8), pred[sample], "s--", color="#d62728", label="Прогноз")
            ax.set_title(f"{title}, пример {row + 1}", fontsize=9)
            ax.set_xlabel("Шаг прогноза")
            ax.set_ylabel("y")
            ax.set_xticks(range(1, 8))
            if row == 0 and column == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Примеры многошагового прогноза на 7 шагов вперёд", fontsize=13)
    fig.tight_layout()
    return _save(fig, "fig12_multistep_forecast.png")


def plot_horizon_errors(nn_results: dict[str, dict]) -> Path:
    """Рис. 13. MAE по шагам горизонта для трёх сетей."""
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    colors = {"single": "#1f77b4", "mlp": "#2ca02c", "deep": "#d62728"}
    labels = {"single": "Однослойный", "mlp": "MLP", "deep": "ResNet-MLP"}
    for arch in ["single", "mlp", "deep"]:
        record = nn_results[f"{arch}_h7_scaled"]
        rows = record["horizon_metrics"]
        ax.plot([row["step"] for row in rows], [row["mae"] for row in rows], "o-", label=labels[arch], color=colors[arch])
    ax.set_xlabel("Шаг прогноза")
    ax.set_ylabel("MAE")
    ax.set_title("MAE по шагам многошагового прогноза (тестовая выборка, стандартизация)")
    ax.set_xticks(range(1, 8))
    ax.legend()
    return _save(fig, "fig13_horizon_errors.png")
