# -*- coding: utf-8 -*-
"""Запуск полного цикла лабораторной работы № 3."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import main


if __name__ == "__main__":
    main(build_report=True)
