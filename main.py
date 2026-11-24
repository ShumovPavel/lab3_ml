# -*- coding: utf-8 -*-
"""Лабораторная работа № 3. Полный цикл расчётов и сборка отчёта."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.base_models import fit_base_models
from src.config import FIGURES_DIR, REPORT_DIR, RESULTS_DIR, SEED
from src.data_loader import correlation_table, load_all, set_seed
from src.ensemble_models import fit_ensemble_models
from src.evaluation import export_comparison
from src.neural_nets import fit_neural_networks
from src.single_models import fit_single_models
from src.utils import save_json
from src.visualization import (
    plot_base_regressions,
    plot_correlation,
    plot_feature_importance,
    plot_features_and_target,
    plot_horizon_errors,
    plot_learning_curves,
    plot_model_comparison,
    plot_multistep_examples,
    plot_predictions_grid,
    plot_residuals_grid,
    plot_tree_dendrogram,
)


def _record(label: str, metrics: dict, **extra) -> dict:
    return {"label": label, **metrics, **extra}


def _parts_from_full(data, prediction: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "train": prediction[data.train_index],
        "val": prediction[data.val_index],
        "test": prediction[data.test_index],
    }


def _print_metrics(title: str, records: dict[str, dict]) -> None:
    print(f"\n{title}")
    for key, record in records.items():
        m = record
        print(
            f"  {m['label']:<58s} MAE={m['mae']:.4f}  MSE={m['mse']:.4f}  R={m['r']:.4f}"
        )


def main(build_report: bool = True) -> None:
    started = time.time()
    set_seed(SEED)
    print("=" * 78)
    print("Лабораторная работа № 3. Вариант 4. Регрессия, ансамбли, нейронные сети")
    print("=" * 78)

    frame, data_by_horizon, summary = load_all()
    data1 = data_by_horizon[1]
    data7 = data_by_horizon[7]
    corr, target_corr, best_feature = correlation_table(data1)
    print("Данные:", summary["n_rows"], "наблюдений; пропусков:", sum(summary["missing"].values()))
    print("Выборки, 1 шаг:", data1.splits, "; 7 шагов:", data7.splits)
    print("Корреляции с y(t+1):", {k: round(v, 4) for k, v in target_corr.items()})
    print("Наибольшая |r|:", best_feature, round(target_corr[best_feature], 4))

    # ---------------------------------------------------------------- Рисунки 1–2
    print("\n[1/6] Предварительный анализ и корреляции...")
    fig_features = plot_features_and_target(frame)
    fig_corr = plot_correlation(corr)

    # ---------------------------------------------------------------- Базовые модели
    print("[2/6] Базовые регрессионные модели (п. 1.4–1.6)...")
    base_results, base_predictions, base_models = fit_base_models(data1)
    base_results["pair"]["label"] = (
        f"1.4. Парная линейная регрессия (признак {base_results['pair']['feature']})"
    )
    fig_base = plot_base_regressions(data1, base_results, base_predictions, base_models)
    _print_metrics("Базовые модели (тест):", {
        key: {"label": value["label"], **value["metrics"]} for key, value in base_results.items()
    })

    # ---------------------------------------------------------------- Одиночные модели
    print("[3/6] Одиночные методы: KNN, SVR, Decision Tree (п. 2.1–2.3)...")
    single_scaled, single_pred_scaled, single_models = fit_single_models(data1, scaled=True)
    single_raw, single_pred_raw, _ = fit_single_models(data1, scaled=False)
    _print_metrics("Одиночные методы, StandardScaler:", {
        key: {"label": value["label"], **value["metrics"]} for key, value in single_scaled.items()
    })
    _print_metrics("Одиночные методы, без масштабирования:", {
        key: {"label": value["label"], **value["metrics"]} for key, value in single_raw.items()
    })
    dendrogram_path = plot_tree_dendrogram(single_models["tree"])

    # ---------------------------------------------------------------- Ансамбли
    print("[4/6] Ансамблевые методы: Random Forest, Bagging, Gradient Boosting (п. 3.1–3.3)...")
    ensemble_results, ensemble_pred_scaled, ensemble_pred_raw, ensemble_models = fit_ensemble_models(data1)
    _print_metrics("Ансамбли, StandardScaler:", {
        key: {"label": value["label"], **value["metrics_scaled"]} for key, value in ensemble_results.items()
    })

    # ---------------------------------------------------------------- Нейронные сети
    print("[5/6] Нейронные сети PyTorch: 1 и 7 шагов (п. 4.1–4.3)...")
    nn_summary, nn_pred_scaled, nn_pred_raw, nn_artifacts = fit_neural_networks(data_by_horizon)
    nn_results = nn_summary["results"]
    _print_metrics("Нейронные сети, 1 шаг, StandardScaler:", {
        key: {"label": value["label"], **value["metrics"]}
        for key, value in nn_results.items() if key.endswith("h1_scaled")
    })
    _print_metrics("Нейронные сети, 7 шагов, StandardScaler:", {
        key: {"label": value["label"], **value["metrics"]}
        for key, value in nn_results.items() if key.endswith("h7_scaled")
    })

    # ---------------------------------------------------------------- Итоговые таблицы
    print("[6/6] Сводные метрики, графики и экспорт...")
    metrics_scaled: dict[str, dict] = {}
    metrics_raw: dict[str, dict] = {}
    for key in ("pair", "multiple", "nonlinear"):
        record = base_results[key]
        metrics_scaled[key] = _record(record["label"], record["metrics"])
        metrics_raw[key] = _record(record["label"], record["metrics"])
    for key, record in single_scaled.items():
        metrics_scaled[f"single_{key}"] = _record(record["label"], record["metrics"])
    for key, record in single_raw.items():
        metrics_raw[f"single_{key}"] = _record(record["label"], record["metrics"])
    for key, record in ensemble_results.items():
        metrics_scaled[f"ens_{key}"] = _record(record["label"], record["metrics_scaled"])
        metrics_raw[f"ens_{key}"] = _record(record["label"], record["metrics_raw"])
    for arch in ("single", "mlp", "deep"):
        record = nn_results[f"{arch}_h1_scaled"]
        metrics_scaled[f"nn_{arch}"] = _record(f"{record['label']} (1 шаг)", record["metrics"])
        record_raw = nn_results[f"{arch}_h1_raw"]
        metrics_raw[f"nn_{arch}"] = _record(f"{record_raw['label']} (1 шаг)", record_raw["metrics"])

    _print_metrics("ИТОГО, StandardScaler:", metrics_scaled)
    _print_metrics("ИТОГО, без масштабирования:", metrics_raw)

    multi_results = {}
    for key, record in nn_results.items():
        if key.endswith("h7_scaled"):
            multi_results[key] = {
                "label": record["label"],
                "horizon": 7,
                "n_params": record["n_params"],
                **record["metrics"],
            }
    export_path = export_comparison(metrics_scaled, metrics_raw, multi_results)

    # ---------------------------------------------------------------- Сводные методы
    methods = {
        "pair": {"label": base_results["pair"]["label"], **_parts_from_full(data1, base_predictions["pair"])},
        "multiple": {"label": base_results["multiple"]["label"], **_parts_from_full(data1, base_predictions["multiple"])},
        "nonlinear": {"label": base_results["nonlinear"]["label"], **_parts_from_full(data1, base_predictions["nonlinear"])},
        "knn": {"label": single_scaled["knn"]["label"], **single_pred_scaled["knn"]},
        "svr": {"label": single_scaled["svr"]["label"], **single_pred_scaled["svr"]},
        "tree": {"label": single_scaled["tree"]["label"], **single_pred_scaled["tree"]},
        "rf": {"label": ensemble_results["rf"]["label"], **ensemble_pred_scaled["rf"]},
        "bagging": {"label": ensemble_results["bagging"]["label"], **ensemble_pred_scaled["bagging"]},
        "gb": {"label": ensemble_results["gb"]["label"], **ensemble_pred_scaled["gb"]},
        "nn_single": {"label": nn_results["single_h1_scaled"]["label"], **nn_pred_scaled[1]["single"]},
        "nn_mlp": {"label": nn_results["mlp_h1_scaled"]["label"], **nn_pred_scaled[1]["mlp"]},
        "nn_deep": {"label": nn_results["deep_h1_scaled"]["label"], **nn_pred_scaled[1]["deep"]},
    }

    # ---------------------------------------------------------------- Сохранение результатов (до графиков)
    figure_names = {
        "features": "fig01_features_target.png",
        "correlation": "fig02_correlation.png",
        "base": "fig03_base_regressions.png",
        "dendrogram": "fig04_tree_dendrogram.png",
        "comparison": "fig10_model_comparison.png",
        "learning_1": "fig06_nn_learning_curves_1step.png",
        "learning_7": "fig07_nn_learning_curves_7step.png",
        "predictions": "fig08_predictions_all_methods.png",
        "residuals": "fig09_residuals_all_methods.png",
        "importance": "fig11_feature_importance.png",
        "multistep": "fig12_multistep_forecast.png",
        "horizon": "fig13_horizon_errors.png",
    }
    payload = {
        "dataset": summary,
        "correlation": {
            "matrix": corr.round(6).to_dict(),
            "target": target_corr,
            "best_feature": best_feature,
        },
        "splits": {
            "1": {"train": data1.splits[0], "val": data1.splits[1], "test": data1.splits[2]},
            "7": {"train": data7.splits[0], "val": data7.splits[1], "test": data7.splits[2]},
        },
        "base": base_results,
        "single_scaled": single_scaled,
        "single_raw": single_raw,
        "ensemble": ensemble_results,
        "neural": nn_summary,
        "metrics_scaled": metrics_scaled,
        "metrics_raw": metrics_raw,
        "multi_results": multi_results,
        "figures": {key: str(FIGURES_DIR / name) for key, name in figure_names.items()},
        "export": str(export_path),
    }
    json_path = save_json(RESULTS_DIR / "lab3_results.json", payload)

    np.savez_compressed(
        RESULTS_DIR / "predictions.npz",
        pair=base_predictions["pair"],
        multiple=base_predictions["multiple"],
        nonlinear=base_predictions["nonlinear"],
        **{f"single_s_{k}_{p}": v for k, parts in single_pred_scaled.items() for p, v in parts.items()},
        **{f"single_r_{k}_{p}": v for k, parts in single_pred_raw.items() for p, v in parts.items()},
        **{f"ens_s_{k}_{p}": v for k, parts in ensemble_pred_scaled.items() for p, v in parts.items()},
        **{f"ens_r_{k}_{p}": v for k, parts in ensemble_pred_raw.items() for p, v in parts.items()},
        **{f"nn_s_h{h}_{k}_{p}": v for h, archs in nn_pred_scaled.items() for k, parts in archs.items() for p, v in parts.items()},
        **{f"nn_r_h{h}_{k}_{p}": v for h, archs in nn_pred_raw.items() for k, parts in archs.items() for p, v in parts.items()},
    )
    print(f"\nРезультаты JSON: {json_path}")
    print(f"Таблица метрик: {export_path}")

    # ---------------------------------------------------------------- Графики
    def _safe_figure(name: str, function, *args, **kwargs):
        try:
            path = function(*args, **kwargs)
            print(f"  рисунок: {path}")
            return path
        except Exception as exc:  # pragma: no cover
            print(f"  ОШИБКА при построении {name}: {exc}")
            return None

    _safe_figure("features", plot_features_and_target, frame)
    _safe_figure("correlation", plot_correlation, corr)
    _safe_figure("base", plot_base_regressions, data1, base_results, base_predictions, base_models)
    _safe_figure("dendrogram", plot_tree_dendrogram, single_models["tree"])
    _safe_figure(
        "predictions", plot_predictions_grid, data1, methods,
        "fig08_predictions_all_methods.png",
        "Реальные значения и прогноз всех одношаговых моделей (Train / Validation / Test)",
    )
    _safe_figure(
        "residuals", plot_residuals_grid, data1, methods,
        "fig09_residuals_all_methods.png",
        "Разность y − ŷ для всех одношаговых моделей (Train / Validation / Test)",
    )
    _safe_figure("comparison", plot_model_comparison, metrics_scaled, metrics_raw)
    _safe_figure(
        "importance", plot_feature_importance,
        ensemble_results["rf"]["feature_importance"], ensemble_results["gb"]["feature_importance"],
    )
    _safe_figure("learning_1", plot_learning_curves, nn_artifacts["histories"], 1)
    _safe_figure("learning_7", plot_learning_curves, nn_artifacts["histories"], 7)
    _safe_figure("multistep", plot_multistep_examples, data7, nn_pred_scaled[7])
    _safe_figure("horizon", plot_horizon_errors, nn_results)

    print(f"Рисунки: {FIGURES_DIR}")

    if build_report:
        from src.report_builder import build_report

        report_path = build_report()
        print(f"Отчёт: {report_path}")

    print(f"\nГотово за {time.time() - started:.1f} с.")


if __name__ == "__main__":
    main()
