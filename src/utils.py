# -*- coding: utf-8 -*-
"""Вспомогательные функции."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def to_jsonable(value):
    """Рекурсивно приводит объекты NumPy к типам, понятным json."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
