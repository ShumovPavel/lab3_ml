# -*- coding: utf-8 -*-
"""Нейронные сети на PyTorch для одношагового и многошагового прогноза."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.config import (
    NN_BATCH_SIZE,
    NN_EPOCHS,
    NN_LEARNING_RATE,
    NN_PATIENCE,
    NN_WEIGHT_DECAY,
    SEED,
)
from src.data_loader import SupervisedData
from src.evaluation import horizon_metrics, regression_metrics


torch.set_num_threads(1)


def _activation(name: str, size: int | None = None) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "prelu":
        return nn.PReLU(num_parameters=size or 1)
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Неизвестная активация: {name}")


class ScaleShift(nn.Module):
    """Обучаемое масштабирование и сдвиг выхода сети."""

    def __init__(self, out_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(out_features))
        self.shift = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x * self.scale + self.shift


class SingleLayerPerceptron(nn.Module):
    """Однослойный персептрон: Linear -> активация -> Scale/Shift."""

    def __init__(self, in_features: int, out_features: int, activation: str = "prelu"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = _activation(activation, out_features)
        self.scale_shift = ScaleShift(out_features)

    def forward(self, x):
        return self.scale_shift(self.activation(self.linear(x)))


class MLP(nn.Module):
    """Двухслойный многослойный персептрон."""

    def __init__(self, in_features: int, out_features: int, activation: str = "tanh"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            _activation(activation, 64),
            nn.Linear(64, 32),
            _activation("silu" if activation != "silu" else "tanh", 32),
            nn.Linear(32, out_features),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """Резидуальный блок с пре-нормализацией."""

    def __init__(self, dim: int, activation: str = "silu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = _activation(activation, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.fc1(self.act(self.norm1(x)))
        h = self.fc2(h)
        return self.norm2(x + h)


class DeepResNetMLP(nn.Module):
    """Глубокая сеть с residual-связями: 5 -> 64 -> ResBlock -> 32 -> out."""

    def __init__(self, in_features: int, out_features: int, activation: str = "silu"):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(in_features, 64),
            _activation(activation, 64),
        )
        self.block = ResidualBlock(64, activation)
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            _activation(activation, 32),
            nn.Linear(32, out_features),
        )

    def forward(self, x):
        return self.head(self.block(self.input(x)))


@dataclass
class TrainResult:
    model: nn.Module
    history: dict[str, list[float]]
    n_params: int
    best_epoch: int
    best_val_mse: float
    predictions: dict[str, np.ndarray]


def _build_model(arch: str, in_features: int, out_features: int, activation: str) -> nn.Module:
    if arch == "single":
        return SingleLayerPerceptron(in_features, out_features, activation)
    if arch == "mlp":
        return MLP(in_features, out_features, activation)
    if arch == "deep":
        return DeepResNetMLP(in_features, out_features, activation)
    raise ValueError(f"Неизвестная архитектура: {arch}")


def _n_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train_network(
    arch: str,
    data: SupervisedData,
    scaled: bool,
    activation: str,
    learning_rate: float = NN_LEARNING_RATE,
    optimizer_name: str = "adam",
    loss_name: str = "mse",
    epochs: int = NN_EPOCHS,
    seed: int = SEED,
) -> TrainResult:
    """Обучает сеть с ранней остановкой по валидационной MSE."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = data.X_train_s if scaled else data.X_train
    X_val = data.X_val_s if scaled else data.X_val
    X_test = data.X_test_s if scaled else data.X_test
    y_train = data.y_train_s
    y_val = data.y_val_s
    y_test = data.y_test_s

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    model = _build_model(arch, X_train.shape[1], y_train.shape[1], activation)
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=NN_WEIGHT_DECAY
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=NN_WEIGHT_DECAY
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=12, min_lr=1e-5
    )
    loss_fn = nn.MSELoss() if loss_name == "mse" else nn.HuberLoss(delta=1.0)

    history = {
        "epoch": [],
        "train_mse": [],
        "val_mse": [],
        "train_mae": [],
        "val_mae": [],
        "lr": [],
    }
    best_state = copy.deepcopy(model.state_dict())
    best_val = math.inf
    best_epoch = 0
    bad_epochs = 0
    batch_size = min(NN_BATCH_SIZE, len(X_train))

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(X_train_t))
        for start in range(0, len(X_train_t), batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = X_train_t[indices]
            batch_y = y_train_t[indices]
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_pred_scaled = model(X_train_t).numpy()
            val_pred_scaled = model(X_val_t).numpy()
        train_pred = data.scaler_y.inverse_transform(train_pred_scaled)
        val_pred = data.scaler_y.inverse_transform(val_pred_scaled)
        train_true = data.y_train
        val_true = data.y_val
        train_mse = float(np.mean((train_true - train_pred) ** 2))
        val_mse = float(np.mean((val_true - val_pred) ** 2))
        train_mae = float(np.mean(np.abs(train_true - train_pred)))
        val_mae = float(np.mean(np.abs(val_true - val_pred)))

        history["epoch"].append(epoch)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        scheduler.step(val_mse)
        if val_mse < best_val - 1e-5:
            best_val = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= NN_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_train = data.scaler_y.inverse_transform(model(X_train_t).numpy())
        pred_val = data.scaler_y.inverse_transform(model(X_val_t).numpy())
        pred_test = data.scaler_y.inverse_transform(model(X_test_t).numpy())

    return TrainResult(
        model=model,
        history=history,
        n_params=_n_params(model),
        best_epoch=best_epoch,
        best_val_mse=best_val,
        predictions={"train": pred_train, "val": pred_val, "test": pred_test},
    )


def _tune_architecture(data: SupervisedData, arch: str, activation_grid, lr_grid):
    """Подбирает активацию и learning rate по валидационной MSE (1 шаг, станд.)."""
    y_val = data.y_val
    experiments = []
    best = None
    for activation, learning_rate in [(a, lr) for a in activation_grid for lr in lr_grid]:
        result = train_network(
            arch, data, scaled=True, activation=activation,
            learning_rate=learning_rate, epochs=140,
        )
        metrics = regression_metrics(y_val, result.predictions["val"])
        record = {
            "activation": activation,
            "learning_rate": learning_rate,
            "val_mse": metrics["mse"],
            "val_mae": metrics["mae"],
            "epochs": len(result.history["epoch"]),
        }
        experiments.append(record)
        if best is None or metrics["mse"] < best[0]:
            best = (metrics["mse"], activation, learning_rate)

    return {"best_activation": best[1], "best_lr": best[2], "experiments": experiments}


def fit_neural_networks(
    data_by_horizon: dict[int, SupervisedData],
) -> tuple[dict, dict, dict, dict]:
    """Тюнинг и обучение трёх архитектур для горизонтов 1 и 7 в двух режимах."""
    data1 = data_by_horizon[1]
    arch_names = ["single", "mlp", "deep"]
    labels = {
        "single": "4.1. Однослойный линейный персептрон",
        "mlp": "4.2. Многослойный персептрон (MLP)",
        "deep": "4.3. Глубокая сеть ResNet-MLP",
    }

    tuning = {
        "single": _tune_architecture(data1, "single", ["identity", "tanh", "relu", "prelu"], [0.01, 0.03]),
        "mlp": _tune_architecture(data1, "mlp", ["tanh", "relu", "silu"], [0.005, 0.01, 0.02]),
        "deep": _tune_architecture(data1, "deep", ["tanh", "relu", "silu"], [0.005, 0.01]),
    }

    # Дополнительно проверяем функцию потерь и оптимизатор для MLP.
    optimizer_experiments = []
    best_config = tuning["mlp"]
    for optimizer_name, loss_name, lr in [
        ("adam", "mse", best_config["best_lr"]),
        ("adam", "huber", best_config["best_lr"]),
        ("sgd", "mse", 0.01),
    ]:
        result = train_network(
            "mlp", data1, scaled=True, activation=best_config["best_activation"],
            learning_rate=lr, optimizer_name=optimizer_name, loss_name=loss_name, epochs=140,
        )
        metrics = regression_metrics(data1.y_val, result.predictions["val"])
        optimizer_experiments.append(
            {
                "optimizer": optimizer_name,
                "loss": loss_name,
                "learning_rate": lr,
                "val_mse": metrics["mse"],
                "val_mae": metrics["mae"],
            }
        )

    results: dict[str, dict] = {}
    predictions_scaled: dict[int, dict[str, dict[str, np.ndarray]]] = {1: {}, 7: {}}
    predictions_raw: dict[int, dict[str, dict[str, np.ndarray]]] = {1: {}, 7: {}}
    histories: dict[str, dict] = {}
    models: dict[str, nn.Module] = {}

    for horizon, data in data_by_horizon.items():
        for arch in arch_names:
            config = tuning[arch]
            for scaled in (True, False):
                mode = "scaled" if scaled else "raw"
                result = train_network(
                    arch,
                    data,
                    scaled=scaled,
                    activation=config["best_activation"],
                    learning_rate=config["best_lr"],
                    epochs=NN_EPOCHS,
                )
                key = f"{arch}_h{horizon}_{mode}"
                record = {
                    "label": labels[arch],
                    "architecture": arch,
                    "horizon": horizon,
                    "mode": mode,
                    "activation": config["best_activation"],
                    "learning_rate": config["best_lr"],
                    "n_params": result.n_params,
                    "best_epoch": result.best_epoch,
                    "metrics": regression_metrics(data.y_test, result.predictions["test"]),
                    "val_metrics": regression_metrics(data.y_val, result.predictions["val"]),
                    "train_metrics": regression_metrics(data.y_train, result.predictions["train"]),
                }
                if horizon > 1:
                    record["horizon_metrics"] = horizon_metrics(
                        data.y_test, result.predictions["test"]
                    )
                results[key] = record
                if scaled:
                    predictions_scaled[horizon][arch] = result.predictions
                    histories[f"{arch}_h{horizon}"] = result.history
                    models[f"{arch}_h{horizon}"] = result.model
                else:
                    predictions_raw[horizon][arch] = result.predictions

    return (
        {
            "tuning": tuning,
            "optimizer_experiments": optimizer_experiments,
            "results": results,
        },
        predictions_scaled,
        predictions_raw,
        {"histories": histories, "models": models},
    )
