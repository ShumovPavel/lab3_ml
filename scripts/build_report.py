# -*- coding: utf-8 -*-
"""Пересборка DOCX-отчёта по готовым результатам JSON."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.report_builder import build_report


if __name__ == "__main__":
    print(build_report())
