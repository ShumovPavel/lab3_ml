# -*- coding: utf-8 -*-
"""Сборка отчёта по лабораторной работе № 3 в формате DOCX."""

from __future__ import annotations

import json
import math
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from src.config import FIGURES_DIR, REPORT_DIR, RESULTS_DIR

FONT = "Times New Roman"
MONO_FONT = "Consolas"
BODY_SIZE = 14
CAPTION_SIZE = 12
TABLE_SIZE = 9


def _num(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        if math.isnan(float(value)):
            return "—"
    except (TypeError, ValueError):
        return str(value)
    return f"{float(value):.{digits}f}".replace(".", ",")


def _signed_pct(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f}".replace(".", ",") + " %"


def _set_run_font(run, size=BODY_SIZE, bold=False, italic=False, name=FONT, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), name)


def _set_cell(cell, text: str, bold=False, size=TABLE_SIZE, align=WD_ALIGN_PARAGRAPH.CENTER, shade=None):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = align
    paragraph.paragraph_format.first_line_indent = Cm(0)
    paragraph.paragraph_format.left_indent = Cm(0)
    paragraph.paragraph_format.right_indent = Cm(0)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.0
    run = paragraph.add_run(str(text))
    _set_run_font(run, size=size, bold=bold)
    if shade:
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), shade)
        tc_pr.append(shd)


def _add_table(doc, headers, rows, widths_cm=None, font_size=TABLE_SIZE, alignments=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for index, header in enumerate(headers):
        _set_cell(table.rows[0].cells[index], header, bold=True, size=font_size, shade="D9E2F3")
    tr_pr = table.rows[0]._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)
    for row_values in rows:
        row = table.add_row()
        for index, value in enumerate(row_values):
            alignment = alignments[index] if alignments else WD_ALIGN_PARAGRAPH.CENTER
            _set_cell(row.cells[index], value, size=font_size, align=alignment)
        tr_pr = row._tr.get_or_add_trPr()
        cant_split = OxmlElement("w:cantSplit")
        tr_pr.append(cant_split)
    if widths_cm:
        for row in table.rows:
            for cell, width in zip(row.cells, widths_cm):
                cell.width = Cm(width)
    trailing = doc.add_paragraph()
    trailing.paragraph_format.first_line_indent = Cm(0)
    trailing.paragraph_format.space_after = Pt(2)
    trailing.paragraph_format.line_spacing = 1.0
    return table


def _add_caption(doc, text: str):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.first_line_indent = Cm(0)
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.line_spacing = 1.0
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    _set_run_font(run, size=CAPTION_SIZE)


def _add_figure(doc, filename: str, caption: str, width_cm: float = 16.2):
    path = FIGURES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Не найден рисунок: {path}")
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.first_line_indent = Cm(0)
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(path), width=Cm(width_cm))
    _add_caption(doc, caption)


def _add_body(doc, text: str, first_indent: float = 1.25, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=BODY_SIZE):
    paragraph = doc.add_paragraph()
    paragraph.alignment = align
    paragraph.paragraph_format.first_line_indent = Cm(first_indent)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.5
    run = paragraph.add_run(text)
    _set_run_font(run, size=size)


def _add_heading(doc, text: str, level: int = 1):
    paragraph = doc.add_paragraph(style=f"Heading {level}")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.first_line_indent = Cm(0)
    paragraph.paragraph_format.left_indent = Cm(0)
    paragraph.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    paragraph.paragraph_format.space_after = Pt(6 if level == 1 else 4)
    paragraph.paragraph_format.line_spacing = 1.0
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    _set_run_font(run, size=BODY_SIZE, bold=True, color=RGBColor(0, 0, 0))


def _add_code(doc, text: str):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.first_line_indent = Cm(0)
    paragraph.paragraph_format.left_indent = Cm(0.7)
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.line_spacing = 1.0
    run = paragraph.add_run(text)
    _set_run_font(run, size=9, name=MONO_FONT)


def _add_page_number_footer(section):
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.first_line_indent = Cm(0)
    run = paragraph.add_run()
    _set_run_font(run, size=10)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for element in (begin, instr, separate, end):
        run._r.append(element)


def _title_page(doc):
    for text, size, bold in [
        ("МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ РОССИЙСКОЙ ФЕДЕРАЦИИ", 12, False),
        ("ФЕДЕРАЛЬНОЕ ГОСУДАРСТВЕННОЕ БЮДЖЕТНОЕ ОБРАЗОВАТЕЛЬНОЕ УЧРЕЖДЕНИЕ", 12, False),
        ("ВЫСШЕГО ОБРАЗОВАНИЯ", 12, False),
        ("«ДОНСКОЙ ГОСУДАРСТВЕННЫЙ ТЕХНИЧЕСКИЙ УНИВЕРСИТЕТ»", 12, True),
    ]:
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_after = Pt(2)
        _set_run_font(paragraph.add_run(text), size=size, bold=bold)

    for _ in range(5):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.line_spacing = 1.0

    for text, size, bold, after in [
        ("ЛАБОРАТОРНАЯ РАБОТА № 3", 18, True, 10),
        ("по дисциплине «Машинное обучение»", 14, False, 8),
        (
            "на тему «Обучение с учителем. Регрессия. Ансамблевое обучение. "
            "Нейронные сети»",
            14,
            True,
            10,
        ),
        ("Вариант 4", 14, True, 24),
    ]:
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.line_spacing = 1.15
        paragraph.paragraph_format.space_after = Pt(after)
        _set_run_font(paragraph.add_run(text), size=size, bold=bold)

    for text, bold in [
        ("Выполнил:", True),
        ("студент Подшивалин Р.В.", False),
        ("группа ИПБП41", False),
        ("", False),
        ("Проверил:", True),
        ("Дольников Максим Геннадьевич", False),
    ]:
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_after = Pt(2)
        _set_run_font(paragraph.add_run(text), size=13, bold=bold)

    for _ in range(4):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.line_spacing = 1.0

    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.first_line_indent = Cm(0)
    paragraph.paragraph_format.line_spacing = 1.0
    _set_run_font(paragraph.add_run("Ростов-на-Дону\n2026 г."), size=13)


def _format_params(params: dict) -> str:
    parts = []
    for key, value in params.items():
        if isinstance(value, float):
            parts.append(f"{key}={_num(value, 4)}")
        elif value is None:
            parts.append(f"{key}=None")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _metrics_rows(metrics: dict, keys_order: list[str]):
    rows = []
    for key in keys_order:
        record = metrics[key]
        rows.append(
            [
                record["label"],
                _num(record["mae"], 4),
                _num(record["mse"], 4),
                _num(record["r"], 5),
            ]
        )
    return rows


def build_report() -> Path:
    data_path = RESULTS_DIR / "lab3_results.json"
    if not data_path.exists():
        raise FileNotFoundError("Сначала выполните python main.py — нет lab3_results.json")
    payload = json.loads(data_path.read_text(encoding="utf-8"))

    ds = payload["dataset"]
    corr = payload["correlation"]
    base = payload["base"]
    single_s = payload["single_scaled"]
    single_r = payload["single_raw"]
    ens = payload["ensemble"]
    nn = payload["neural"]
    nn_res = nn["results"]
    tuning = nn["tuning"]
    metrics_scaled = payload["metrics_scaled"]
    metrics_raw = payload["metrics_raw"]
    multi = payload["multi_results"]

    order = list(metrics_scaled.keys())
    split1 = payload["splits"]["1"]
    split7 = payload["splits"]["7"]

    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(3.0)
    section.right_margin = Cm(1.5)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.different_first_page_header_footer = True

    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(BODY_SIZE)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.first_line_indent = Cm(1.25)

    for level in (1, 2):
        style = doc.styles[f"Heading {level}"]
        style.font.name = FONT
        style.font.size = Pt(BODY_SIZE)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.first_line_indent = Cm(0)
        style.paragraph_format.space_before = Pt(12 if level == 1 else 8)
        style.paragraph_format.space_after = Pt(6 if level == 1 else 4)
        style.paragraph_format.keep_with_next = True

    _add_page_number_footer(section)
    doc.core_properties.title = "Лабораторная работа № 3. Вариант 4"
    doc.core_properties.subject = "Обучение с учителем. Регрессия. Ансамбли. Нейронные сети."
    doc.core_properties.author = "Подшивалин Р.В."

    _title_page(doc)
    doc.add_page_break()

    # ------------------------------------------------------------------ Цель и задачи
    _add_heading(doc, "1. Цель работы", 1)
    _add_body(
        doc,
        "Изучить и сравнить точность методов машинного обучения с учителем в задаче "
        "регрессии для прогнозирования будущих значений целевой переменной временного "
        "ряда (вариант 4): базовых методов статистики и оптимизации (парной, "
        "множественной и нелинейной регрессии SciPy), одиночных моделей (KNN, SVR, "
        "дерева решений), ансамблевых алгоритмов (случайного леса, бэггинга, "
        "градиентного бустинга) и нейронных сетей на PyTorch при одношаговом и "
        "многошаговом прогнозировании.",
    )

    _add_heading(doc, "2. Задачи", 1)
    tasks = [
        "Выполнить предварительную обработку датасета Lab3_Dataset.xlsx (вариант 4), "
        "сформировать выборки со сдвигом на 1 и 7 шагов вперёд, разбить данные на "
        "train/val/test и провести стандартизацию по параметрам обучающей выборки.",
        "Рассчитать матрицу взаимных корреляций и построить модели парной, "
        "множественной и нелинейной регрессии с помощью функции curve_fit() SciPy.",
        "Выполнить прогнозирование одиночными методами (KNN, SVR, Decision Tree) с "
        "подбором гиперпараметров и построить дендрограмму дерева решений.",
        "Реализовать ансамблевые методы (Random Forest, Bagging, Gradient Boosting) и "
        "определить их оптимальные гиперпараметры.",
        "Разработать и обучить в PyTorch три архитектуры (однослойный персептрон, MLP и "
        "глубокую сеть ResNet-MLP) для прогноза на 1 и 7 шагов, сравнить число "
        "параметров и кривые обучения.",
        "Провести сравнительный анализ методов по MAE, MSE и коэффициенту корреляции R "
        "на тестовой выборке с масштабированием и без него, экспортировать результаты в "
        "results_Comparison.xlsx.",
    ]
    for index, task in enumerate(tasks, 1):
        _add_body(doc, f"{index}) {task}")

    _add_heading(doc, "3. Описание структуры проекта", 1)
    _add_body(
        doc,
        "Проект расположен в каталоге lab3_ml и организован по модульному принципу. "
        "Запуск выполняется из терминала Visual Studio Code командой python main.py.",
    )
    _add_table(
        doc,
        ["Модуль / каталог", "Назначение"],
        [
            ["main.py", "Точка входа, последовательный запуск этапов 1–5 и сборка отчёта"],
            ["src/config.py", "Seed = 42, номер варианта, пути, параметры обучения"],
            ["src/data_loader.py", "Чтение книги Excel, сдвиг на 1 и 7 шагов, разбиение, StandardScaler, корреляции"],
            ["src/base_models.py", "П. 1.4–1.6: парная, множественная и нелинейная регрессия (curve_fit)"],
            ["src/single_models.py", "П. 2.1–2.3: KNN, SVR, дерево решений и подбор гиперпараметров"],
            ["src/ensemble_models.py", "П. 3.1–3.3: Random Forest, Bagging, Gradient Boosting"],
            ["src/neural_nets.py", "П. 4.1–4.3: три архитектуры PyTorch для 1 и 7 шагов"],
            ["src/visualization.py", "П. 1.1, 1.3, 4, 5.1–5.3: все рисунки отчёта"],
            ["src/evaluation.py", "П. 5.4–5.6: MAE, MSE, R, экспорт results_Comparison.xlsx"],
            ["src/report_builder.py", "Автоматическая сборка DOCX-отчёта"],
            ["data/Lab3_Dataset.xlsx", "Исходный датасет (вариант 4: a, b, c, d, e, y)"],
            ["results/", "Графики, JSON-результаты, predictions.npz, results_Comparison.xlsx"],
            ["report/", "Итоговый файл отчёта"],
            ["README.md, requirements.txt, .vscode/", "Документация, зависимости и конфигурация запуска"],
        ],
        widths_cm=[4.6, 11.8],
        font_size=8.5,
        alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT],
    )

    # ------------------------------------------------------------------ Раздел 4
    _add_heading(doc, "4. Ход выполнения работы", 1)
    _add_body(
        doc,
        "Программа запускается из терминала в каталоге проекта командой python main.py. "
        "В консоль выводятся размеры выборок, матрица корреляций, подобранные "
        "гиперпараметры, метрики всех моделей и путь к итоговым файлам. Все "
        "стохастические алгоритмы используют random_state = 42, поэтому результаты "
        "воспроизводимы.",
    )
    _add_code(doc, "python main.py")

    # 4.1
    _add_heading(doc, "4.1. Раздел 1. Предварительная обработка, корреляции и базовые модели (п. 1.1–1.6)", 2)
    _add_body(
        doc,
        f"Из книги data/Lab3_Dataset.xlsx извлечён блок варианта 4 (столбцы 19–24): "
        f"признаки a, b, c, d, e и целевая переменная y. Получено {ds['n_rows']} "
        f"наблюдений; пропусков и полных дубликатов не обнаружено. Описательные "
        "статистики признаков приведены в таблице 1.",
    )
    stats_rows = []
    for column in ["a", "b", "c", "d", "e", "y"]:
        stat = ds["statistics"][column]
        stats_rows.append(
            [
                column,
                _num(stat["mean"], 4),
                _num(stat["std"], 4),
                _num(stat["min"], 4),
                _num(stat["max"], 4),
            ]
        )
    _add_caption(doc, "Таблица 1 — Описательные статистики признаков варианта 4")
    _add_table(
        doc,
        ["Признак", "Среднее", "СКО", "Минимум", "Максимум"],
        stats_rows,
        widths_cm=[2.4, 3.4, 3.4, 3.4, 3.4],
        font_size=9,
    )

    _add_body(
        doc,
        f"Для прогноза на 1 шаг вперёд сформированы пары X(t) → y(t+1) объёмом "
        f"{split1['train'] + split1['val'] + split1['test']} наблюдений, для прогноза на "
        f"7 шагов — пары X(t) → y(t+1..t+7) объёмом "
        f"{split7['train'] + split7['val'] + split7['test']} наблюдений. Выборка разбита "
        f"в пропорции 70 % / 15 % / 15 %: для 1 шага Train = {split1['train']}, "
        f"Validation = {split1['val']}, Test = {split1['test']}; для 7 шагов — "
        f"{split7['train']} / {split7['val']} / {split7['test']}. Перемешивание выполнено "
        "с фиксированным seed = 42, внутри каждой части сохранён исходный временной "
        "порядок. StandardScaler обучен только на Train; параметры scaling применены к "
        "Validation и Test, а прогнозы нейронных сетей обратно преобразованы к шкале y.",
    )
    _add_figure(
        doc,
        "fig01_features_target.png",
        "Рисунок 1 — Динамика признаков a–e и целевой переменной y (стандартизованные значения)",
        width_cm=16.4,
    )
    _add_body(
        doc,
        "На рисунке 1 видно, что все пять признаков и целевая переменная представляют "
        "собой гладкие согласованные тренды: с ростом e значения y монотонно убывают. "
        "Признак d имеет осциллирующий шумовой характер и слабее связан с остальными "
        "переменными, тогда как a, b и e изменяются почти синхронно.",
    )
    _add_body(
        doc,
        "Матрица коэффициентов корреляции Пирсона (рисунок 2) подтверждает сильную "
        "линейную связь. Наибольшая по модулю корреляция с y(t+1) принадлежит признаку "
        f"e (r = {_num(corr['target']['e'], 4)}), далее следуют b "
        f"(r = {_num(corr['target']['b'], 4)}), a (r = {_num(corr['target']['a'], 4)}), "
        f"c (r = {_num(corr['target']['c'], 4)}) и d (r = {_num(corr['target']['d'], 4)}). "
        "Между признаками a и b коэффициент корреляции достигает "
        f"{_num(corr['matrix']['a']['b'], 4)}, между a и e — "
        f"{_num(corr['matrix']['a']['e'], 4)}, поэтому данные обладают выраженной "
        "мультиколлинеарностью.",
    )
    _add_figure(
        doc,
        "fig02_correlation.png",
        "Рисунок 2 — Матрица взаимных корреляций признаков и целевой переменной",
        width_cm=12.8,
    )

    pair = base["pair"]
    multiple = base["multiple"]
    nonlinear = base["nonlinear"]
    _add_body(
        doc,
        "Для признака с наибольшей по модулю корреляцией (e) построена модель парной "
        f"линейной регрессии: y = {_num(pair['coefficient'], 4)}·e "
        f"{'+' if pair['intercept'] >= 0 else '−'} {_num(abs(pair['intercept']), 4)}. "
        f"На тестовой выборке MAE = {_num(pair['metrics']['mae'])}, MSE = "
        f"{_num(pair['metrics']['mse'])}, R = {_num(pair['metrics']['r'], 5)}. Высокий "
        "коэффициент корреляции сочетается со значительной абсолютной ошибкой: одна "
        "прямая не описывает кривизну зависимости y от e.",
    )
    _add_body(
        doc,
        "Модель множественной линейной регрессии по всем пяти признакам использует "
        "коэффициенты "
        + ", ".join(
            f"{name} = {_num(value, 3)}" for name, value in multiple["coefficients"].items()
        )
        + f"; свободный член {_num(multiple['intercept'], 3)}. На тесте MAE = "
        f"{_num(multiple['metrics']['mae'])}, MSE = {_num(multiple['metrics']['mse'])}, "
        f"R = {_num(multiple['metrics']['r'], 5)}. Учёт всех признаков снижает MSE по "
        "сравнению с парной моделью, однако линейная форма всё ещё не соответствует "
        "нелинейному характеру данных.",
    )
    _add_body(
        doc,
        "По виду исходных данных подобрана квадратичная аппроксимирующая функция, "
        "включающая свободный член, пять линейных членов, пять квадратов и десять "
        "парных произведений (всего 21 параметр):",
    )
    _add_code(
        doc,
        "y = p0 + Σ pi·xi + Σ pii·xi² + Σ pij·xi·xj,  i,j ∈ {a,b,c,d,e}, i < j",
    )
    _add_body(
        doc,
        f"Параметры найдены функцией scipy.optimize.curve_fit (method='trf', "
        f"x_scale='jac') с начальным приближением по МНК; оптимизация сошлась "
        f"({nonlinear['optimizer']}). На тестовой выборке нелинейная модель показала "
        f"MAE = {_num(nonlinear['metrics']['mae'])}, MSE = "
        f"{_num(nonlinear['metrics']['mse'])}, R = "
        f"{_num(nonlinear['metrics']['r'], 5)}. По сравнению с множественной линейной "
        f"регрессией MSE уменьшилась в "
        f"{_num(multiple['metrics']['mse'] / nonlinear['metrics']['mse'], 1)} раза, а по "
        f"сравнению с парной — в "
        f"{_num(pair['metrics']['mse'] / nonlinear['metrics']['mse'], 1)} раза.",
    )
    _add_figure(
        doc,
        "fig03_base_regressions.png",
        "Рисунок 3 — Исходные данные, аппроксимирующая функция и прогноз базовых моделей (п. 5.2)",
        width_cm=16.4,
    )
    _add_caption(doc, "Таблица 2 — Метрики базовых регрессионных моделей на тестовой выборке")
    _add_table(
        doc,
        ["Метод", "MAE", "MSE", "R"],
        _metrics_rows(
            {"pair": {"label": pair["label"], **pair["metrics"]},
             "multiple": {"label": multiple["label"], **multiple["metrics"]},
             "nonlinear": {"label": nonlinear["label"], **nonlinear["metrics"]}},
            ["pair", "multiple", "nonlinear"],
        ),
        widths_cm=[8.2, 2.7, 2.7, 2.9],
        font_size=9,
    )

    # 4.2
    _add_heading(doc, "4.2. Раздел 2. Одиночные методы машинного обучения (п. 2.1–2.3)", 2)
    _add_body(
        doc,
        "Гиперпараметры одиночных моделей подбирались перебором по валидационной "
        "выборке: для KNN варьировались число соседей k = 1…31, вес (uniform/distance) "
        "и метрика p = 1, 2; для SVR — C, gamma и epsilon при RBF-ядре; для дерева "
        "решений — глубина, минимальное число объектов в листе и число признаков в "
        "разбиении. Критерий выбора — минимальная MSE на Validation.",
    )
    _add_caption(doc, "Таблица 3 — Оптимальные гиперпараметры и метрики одиночных моделей (стандартизация)")
    _add_table(
        doc,
        ["Метод", "Оптимальные параметры (StandardScaler)", "MAE", "MSE", "R"],
        [
            [
                "KNN",
                _format_params(single_s["knn"]["params"]),
                _num(single_s["knn"]["metrics"]["mae"]),
                _num(single_s["knn"]["metrics"]["mse"]),
                _num(single_s["knn"]["metrics"]["r"], 5),
            ],
            [
                "SVR (RBF)",
                _format_params(single_s["svr"]["params"]),
                _num(single_s["svr"]["metrics"]["mae"]),
                _num(single_s["svr"]["metrics"]["mse"]),
                _num(single_s["svr"]["metrics"]["r"], 5),
            ],
            [
                "Decision Tree",
                _format_params(single_s["tree"]["params"]),
                _num(single_s["tree"]["metrics"]["mae"]),
                _num(single_s["tree"]["metrics"]["mse"]),
                _num(single_s["tree"]["metrics"]["r"], 5),
            ],
        ],
        widths_cm=[2.6, 5.9, 2.3, 2.3, 2.5],
        font_size=8,
        alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER],
    )

    _add_body(
        doc,
        f"На стандартизованных данных SVR с RBF-ядром показала лучший результат среди "
        f"одиночных методов: MAE = {_num(single_s['svr']['metrics']['mae'])}, MSE = "
        f"{_num(single_s['svr']['metrics']['mse'])}. Дерево решений уступает ей "
        f"(MAE = {_num(single_s['tree']['metrics']['mae'])}, MSE = "
        f"{_num(single_s['tree']['metrics']['mse'])}), а KNN — "
        f"MAE = {_num(single_s['knn']['metrics']['mae'])}, MSE = "
        f"{_num(single_s['knn']['metrics']['mse'])}. Для KNN стандартизация оказалась "
        "неблагоприятной: она уравняла масштабы информативного признака e и шумового d, "
        "поэтому евклидово расстояние стало сильнее зависеть от осциллирующего d.",
    )
    _add_body(
        doc,
        f"Без масштабирования KNN, наоборот, улучшилась: MAE = "
        f"{_num(single_r['knn']['metrics']['mae'])}, MSE = "
        f"{_num(single_r['knn']['metrics']['mse'])}. Причина в том, что признак e имеет "
        "наибольшую дисперсию и одновременно наибольшую корреляцию с целью, поэтому в "
        "исходном пространстве именно он доминирует в расстоянии. SVR практически "
        "нечувствительна к масштабированию благодаря подбору gamma "
        f"(MAE без масштабирования {_num(single_r['svr']['metrics']['mae'])} против "
        f"{_num(single_s['svr']['metrics']['mae'])} со стандартизацией). Дерево решений "
        "инвариантно к монотонному масштабированию признаков, его метрики совпадают в "
        "двух режимах.",
    )
    _add_figure(
        doc,
        "fig04_tree_dendrogram.png",
        "Рисунок 4 — Дендрограмма верхних ярусов дерева решений (п. 5.3)",
        width_cm=15.4,
    )
    _add_body(
        doc,
        "Дендрограмма (рисунок 4) показывает, что корневые разбиения дерева выполняются "
        "по признаку e, затем используются c и d. Это согласуется с матрицей корреляций: "
        "e обладает наибольшей парной связью с целью и потому оказывается лучшим "
        "одиночным порогом. В Random Forest агрегированная важность выше у c, так как "
        "взаимозаменяемые коррелированные признаки делят вклад между собой.",
    )

    # 4.3
    _add_heading(doc, "4.3. Раздел 3. Ансамблевые методы машинного обучения (п. 3.1–3.3)", 2)
    _add_body(
        doc,
        "Для ансамблей также выполнялся перебор по Validation. Для Random Forest "
        "варьировались глубина деревьев, число признаков в разбиении и минимальный "
        "размер листа; для Bagging — доля признаков и параметры базового дерева; для "
        "Gradient Boosting — число деревьев, темп обучения, глубина и доля подвыборки.",
    )
    _add_caption(doc, "Таблица 4 — Параметры и метрики ансамблевых моделей (стандартизация)")
    _add_table(
        doc,
        ["Метод", "Оптимальные параметры", "MAE", "MSE", "R"],
        [
            [
                "Random Forest",
                _format_params(ens["rf"]["params"]),
                _num(ens["rf"]["metrics_scaled"]["mae"]),
                _num(ens["rf"]["metrics_scaled"]["mse"]),
                _num(ens["rf"]["metrics_scaled"]["r"], 5),
            ],
            [
                "Bagging",
                _format_params(ens["bagging"]["top_trials"][0]["params"]),
                _num(ens["bagging"]["metrics_scaled"]["mae"]),
                _num(ens["bagging"]["metrics_scaled"]["mse"]),
                _num(ens["bagging"]["metrics_scaled"]["r"], 5),
            ],
            [
                "Gradient Boosting",
                _format_params(ens["gb"]["params"]),
                _num(ens["gb"]["metrics_scaled"]["mae"]),
                _num(ens["gb"]["metrics_scaled"]["mse"]),
                _num(ens["gb"]["metrics_scaled"]["r"], 5),
            ],
        ],
        widths_cm=[3.0, 5.7, 2.3, 2.3, 2.4],
        font_size=8,
        alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER],
    )

    _add_body(
        doc,
        f"Все три ансамбля превзошли одиночное дерево решений. Лучшим оказался Bagging: "
        f"MAE = {_num(ens['bagging']['metrics_scaled']['mae'])}, MSE = "
        f"{_num(ens['bagging']['metrics_scaled']['mse'])}; Random Forest показал "
        f"MAE = {_num(ens['rf']['metrics_scaled']['mae'])}, MSE = "
        f"{_num(ens['rf']['metrics_scaled']['mse'])}, Gradient Boosting — MAE = "
        f"{_num(ens['gb']['metrics_scaled']['mae'])}, MSE = "
        f"{_num(ens['gb']['metrics_scaled']['mse'])}. По сравнению с одиночным деревом "
        f"(MSE = {_num(single_s['tree']['metrics']['mse'])}) ансамбли снизили MSE на "
        f"{_num(min((1 - ens['bagging']['metrics_scaled']['mse'] / single_s['tree']['metrics']['mse']) * 100, (1 - ens['gb']['metrics_scaled']['mse'] / single_s['tree']['metrics']['mse']) * 100), 1)}–"
        f"{_num(max((1 - ens['bagging']['metrics_scaled']['mse'] / single_s['tree']['metrics']['mse']) * 100, (1 - ens['gb']['metrics_scaled']['mse'] / single_s['tree']['metrics']['mse']) * 100), 1)} %, "
        "что объясняется усреднением дисперсии и уменьшением переобучения.",
    )
    _add_body(
        doc,
        "Для древовидных моделей стандартизация не меняет результат: разбиения "
        "определяются порядком значений признаков, а не их масштабом. Поэтому метрики "
        "Random Forest, Bagging и Gradient Boosting в таблицах без масштабирования и со "
        "стандартизацией совпадают. Важность признаков (рисунок 12) в Random Forest "
        f"распределена следующим образом: c — {_num(ens['rf']['feature_importance']['c'] * 100, 1)} %, "
        f"a — {_num(ens['rf']['feature_importance']['a'] * 100, 1)} %, "
        f"b — {_num(ens['rf']['feature_importance']['b'] * 100, 1)} %, "
        f"e — {_num(ens['rf']['feature_importance']['e'] * 100, 1)} %, "
        f"d — {_num(ens['rf']['feature_importance']['d'] * 100, 1)} %. Максимальный вклад "
        "даёт признак c; признак e, несмотря на наибольшую парную корреляцию с целью, "
        "получает меньшую агрегированную важность, поскольку сильно коррелирован с a, b "
        "и c и ансамбль распределяет вклад между взаимозаменяемыми признаками. Шумовой "
        "признак d практически не используется деревьями (0,0 %).",
    )

    # 4.4
    _add_heading(doc, "4.4. Раздел 4. Нейронные сети для прогноза на 1 и 7 шагов (п. 4.1–4.3)", 2)
    _add_body(
        doc,
        "Нейросетевые модели реализованы на PyTorch. Для каждого горизонта "
        "использовались три архитектуры: однослойный линейный персептрон с обучаемыми "
        "масштабом и сдвигом, двухслойный MLP и глубокая сеть с residual-связями. "
        "Выходной слой содержит один нейрон для прогноза на 1 шаг и семь нейронов для "
        "векторного прогноза Y(t+1..t+7). Обучение выполнялось оптимизатором Adam, "
        "функция потерь — MSE, batch_size = 64, ранняя остановка по Validation MSE "
        "(patience = 35 эпох), начальный learning rate выбран при тюнинге.",
    )
    nn_rows = []
    for arch, label in [
        ("single", "4.1. Однослойный персептрон"),
        ("mlp", "4.2. Многослойный персептрон (MLP)"),
        ("deep", "4.3. Глубокая сеть ResNet-MLP"),
    ]:
        for horizon in (1, 7):
            record = nn_res[f"{arch}_h{horizon}_scaled"]
            activation = record["activation"]
            second = "tanh" if activation == "silu" else "silu"
            if arch == "single":
                layers = f"Linear(5→out) + {activation} + Scale/Shift"
            elif arch == "mlp":
                layers = f"5 → 64 ({activation}) → 32 ({second}) → out"
            else:
                layers = f"5 → 64 ({activation}) + ResBlock(64×64) → 32 ({activation}) → out + skip"
            nn_rows.append(
                [
                    label,
                    f"{horizon} шаг" if horizon == 1 else f"{horizon} шагов",
                    layers,
                    str(record["n_params"]),
                    _num(record["metrics"]["mae"]),
                    _num(record["metrics"]["mse"]),
                    _num(record["metrics"]["r"], 5),
                    str(record["best_epoch"]),
                ]
            )

    _add_caption(doc, "Таблица 5 — Архитектуры нейронных сетей и метрики на тестовой выборке (стандартизация)")
    _add_table(
        doc,
        ["Архитектура", "Горизонт", "Схема слоёв и активаций", "Парам.", "MAE", "MSE", "R", "Эпоха"],
        nn_rows,
        widths_cm=[3.6, 1.7, 4.0, 1.5, 1.3, 1.3, 1.3, 1.6],
        font_size=7.5,
        alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER],
    )

    _add_body(
        doc,
        f"Число обучаемых параметров однослойной сети минимально — "
        f"{nn_res['single_h1_scaled']['n_params']} для 1 шага и "
        f"{nn_res['single_h7_scaled']['n_params']} для 7 шагов. MLP содержит "
        f"{nn_res['mlp_h1_scaled']['n_params']} и {nn_res['mlp_h7_scaled']['n_params']} "
        f"параметров, глубокая сеть — {nn_res['deep_h1_scaled']['n_params']} и "
        f"{nn_res['deep_h7_scaled']['n_params']} соответственно. Несмотря на рост "
        "числа параметров, глубокая сеть не дала заметного выигрыша по сравнению с MLP: "
        "зависимость целевой переменной достаточно гладкая, и избыточная ёмкость модели "
        "не требуется.",
    )

    _add_body(doc, "Результаты подбора активации и learning rate по валидационной MSE:")
    tuning_rows = []
    for arch, label in [
        ("single", "Однослойный персептрон"),
        ("mlp", "MLP"),
        ("deep", "ResNet-MLP"),
    ]:
        config = tuning[arch]
        best = min(config["experiments"], key=lambda item: item["val_mse"])
        tuning_rows.append(
            [
                label,
                config["best_activation"],
                _num(config["best_lr"], 3),
                _num(best["val_mse"], 5),
                str(best["epochs"]),
            ]
        )
    _add_caption(doc, "Таблица 6 — Выбранные гиперпараметры нейронных сетей")
    _add_table(
        doc,
        ["Архитектура", "Активация", "Learning rate", "Val MSE", "Эпох"],
        tuning_rows,
        widths_cm=[4.6, 2.6, 3.0, 3.0, 2.0],
        font_size=9,
    )

    opt_rows = [
        [
            item["optimizer"].upper(),
            item["loss"].upper(),
            _num(item["learning_rate"], 3),
            _num(item["val_mse"], 5),
            _num(item["val_mae"], 5),
        ]
        for item in nn["optimizer_experiments"]
    ]
    _add_caption(doc, "Таблица 7 — Сравнение оптимизаторов и функций потерь для MLP")
    _add_table(
        doc,
        ["Оптимизатор", "Функция потерь", "Learning rate", "Val MSE", "Val MAE"],
        opt_rows,
        widths_cm=[3.2, 3.4, 3.2, 3.2, 3.2],
        font_size=9,
    )

    _add_body(
        doc,
        "По результатам таблиц 6 и 7 для MLP выбраны Adam и MSE: Adam обеспечивает "
        "адаптивный шаг для разнородных по масштабу признаков и сходится быстрее SGD, а "
        "MSE напрямую соответствует критерию сравнения моделей. Huber-потеря даёт близкий "
        "результат, но не превосходит MSE на данной задаче.",
    )

    _add_figure(
        doc,
        "fig06_nn_learning_curves_1step.png",
        "Рисунок 5 — Кривые обучения MSE и MAE для прогноза на 1 шаг вперёд",
        width_cm=16.4,
    )
    _add_body(
        doc,
        "Кривые обучения (рисунок 5) показывают устойчивую сходимость: Train и "
        "Validation MSE монотонно убывают, признаков переобучения не наблюдается. "
        "Однослойная сеть выходит на плато со значительно большей ошибкой, чем MLP и "
        "ResNet-MLP, что подтверждает нелинейность зависимости.",
    )
    _add_figure(
        doc,
        "fig07_nn_learning_curves_7step.png",
        "Рисунок 6 — Кривые обучения MSE и MAE для прогноза на 7 шагов вперёд",
        width_cm=16.4,
    )
    _add_figure(
        doc,
        "fig12_multistep_forecast.png",
        "Рисунок 7 — Примеры многошагового прогноза на 7 шагов вперёд (тестовая выборка)",
        width_cm=16.4,
    )
    _add_figure(
        doc,
        "fig13_horizon_errors.png",
        "Рисунок 8 — MAE по шагам горизонта для трёх нейросетевых архитектур",
        width_cm=13.5,
    )

    horizon_rows = []
    for step in range(1, 8):
        row = [str(step)]
        for arch in ("single", "mlp", "deep"):
            record = next(
                item for item in nn_res[f"{arch}_h7_scaled"]["horizon_metrics"] if item["step"] == step
            )
            row.append(_num(record["mae"]))
        horizon_rows.append(row)
    _add_caption(doc, "Таблица 8 — MAE по шагам многошагового прогноза (стандартизация)")
    _add_table(
        doc,
        ["Шаг", "Однослойный", "MLP", "ResNet-MLP"],
        horizon_rows,
        widths_cm=[2.0, 4.6, 4.6, 4.6],
        font_size=9,
    )

    _add_body(
        doc,
        f"Для 7-шагового прогноза MLP показал MAE = "
        f"{_num(nn_res['mlp_h7_scaled']['metrics']['mae'])}, MSE = "
        f"{_num(nn_res['mlp_h7_scaled']['metrics']['mse'])}, ResNet-MLP — MAE = "
        f"{_num(nn_res['deep_h7_scaled']['metrics']['mae'])}, MSE = "
        f"{_num(nn_res['deep_h7_scaled']['metrics']['mse'])}. Ошибка плавно растёт с "
        "увеличением шага, что естественно для векторного прогноза: дальние значения "
        "содержат большую неопределённость. Однослойная сеть не способна описать "
        "нелинейную зависимость и даёт ошибку на порядок выше.",
    )

    # 4.5
    _add_heading(doc, "4.5. Раздел 5. Визуализация прогнозов и остатков (п. 5.1–5.3)", 2)
    _add_body(
        doc,
        "Для всех двенадцати одношаговых моделей построены графики реальных и "
        "спрогнозированных значений на объединённой последовательности "
        "[Train | Validation | Test] с вертикальными разделителями (рисунок 9), а также "
        "графики разности y − ŷ (рисунок 10). Индивидуальные графики методов сохранены в "
        "results/figures/.",
    )
    _add_figure(
        doc,
        "fig08_predictions_all_methods.png",
        "Рисунок 9 — Реальные значения и прогноз всех одношаговых моделей (Train / Validation / Test)",
        width_cm=16.4,
    )
    _add_figure(
        doc,
        "fig09_residuals_all_methods.png",
        "Рисунок 10 — Разность y − ŷ для всех одношаговых моделей (Train / Validation / Test)",
        width_cm=16.4,
    )
    _add_body(
        doc,
        "У парной и множественной линейной регрессии остатки имеют выраженную "
        "систематическую составляющую: модель завышает прогноз на одних участках и "
        "занижает на других. У SVR, ансамблей и MLP остатки группируются около нуля, а "
        "их размах на порядок меньше. Наибольшие всплески ошибок приходятся на края "
        "диапазона, где нелинейность зависимости проявляется сильнее.",
    )
    _add_figure(
        doc,
        "fig10_model_comparison.png",
        "Рисунок 11 — Сравнение MAE методов без масштабирования и со стандартизацией",
        width_cm=16.4,
    )
    _add_figure(
        doc,
        "fig11_feature_importance.png",
        "Рисунок 12 — Важность признаков в Random Forest и Gradient Boosting",
        width_cm=14.4,
    )

    # ------------------------------------------------------------------ Результаты
    _add_heading(doc, "5. Результаты", 1)
    _add_body(
        doc,
        "Итоговая таблица метрик экспортирована в файл results/results_Comparison.xlsx "
        "(листы «Без масштабирования», «Со стандартизацией», «Сравнение», "
        "«Многошаговый прогноз»). Ниже приведены результаты на тестовой выборке для двух "
        "режимов обучения (п. 5.4–5.6).",
    )

    headers_raw = ["Метод", "MAE", "MSE", "R"]
    headers_scaled = ["Метод", "MAE", "MSE", "R", "Снижение MAE"]
    rows_raw = []
    rows_scaled = []
    for key in order:
        record_s = metrics_scaled[key]
        record_r = metrics_raw[key]
        delta = (record_r["mae"] - record_s["mae"]) / record_r["mae"] * 100 if record_r["mae"] else float("nan")
        rows_raw.append(
            [record_r["label"], _num(record_r["mae"]), _num(record_r["mse"]), _num(record_r["r"], 5)]
        )
        rows_scaled.append([record_s["label"], _num(record_s["mae"]), _num(record_s["mse"]), _num(record_s["r"], 5), _signed_pct(delta)])
    _add_caption(doc, "Таблица 9 — Результаты на тестовой выборке без масштабирования признаков")
    _add_table(doc, headers_raw, rows_raw, widths_cm=[9.2, 2.4, 2.4, 2.4], font_size=7.5)
    _add_caption(doc, "Таблица 10 — Результаты на тестовой выборке со стандартизацией StandardScaler (по Train)")
    _add_table(doc, headers_scaled, rows_scaled, widths_cm=[7.2, 2.3, 2.3, 2.3, 2.4], font_size=7.5)

    best_key = min(metrics_scaled, key=lambda key: metrics_scaled[key]["mse"])
    best = metrics_scaled[best_key]
    worst_key = max(metrics_scaled, key=lambda key: metrics_scaled[key]["mse"])
    worst = metrics_scaled[worst_key]
    best_raw_key = min(metrics_raw, key=lambda key: metrics_raw[key]["mse"])
    best_raw = metrics_raw[best_raw_key]

    _add_body(
        doc,
        f"При стандартизации наименьшую MSE показал метод «{best['label']}»: MAE = "
        f"{_num(best['mae'])}, MSE = {_num(best['mse'])}, R = {_num(best['r'], 5)}. "
        f"Близкие результаты у Bagging (MSE = "
        f"{_num(ens['bagging']['metrics_scaled']['mse'])}) и Random Forest (MSE = "
        f"{_num(ens['rf']['metrics_scaled']['mse'])}). Без масштабирования лучшим оказался "
        f"«{best_raw['label']}» (MAE = {_num(best_raw['mae'])}, MSE = "
        f"{_num(best_raw['mse'])}). Наихудший результат в обоих режимах у парной линейной "
        f"регрессии: MAE = {_num(pair['metrics']['mae'])}, MSE = "
        f"{_num(pair['metrics']['mse'])}; множественная линейная регрессия улучшает MSE "
        f"до {_num(multiple['metrics']['mse'])}.",
    )

    delta_knn = (metrics_raw["single_knn"]["mae"] - metrics_scaled["single_knn"]["mae"]) / metrics_raw["single_knn"]["mae"] * 100
    delta_mlp = (metrics_raw["nn_mlp"]["mae"] - metrics_scaled["nn_mlp"]["mae"]) / metrics_raw["nn_mlp"]["mae"] * 100
    delta_single = (metrics_raw["nn_single"]["mae"] - metrics_scaled["nn_single"]["mae"]) / metrics_raw["nn_single"]["mae"] * 100
    _add_body(
        doc,
        f"Влияние стандартизации неоднозначно. Для нейронных сетей она полезна: MAE MLP "
        f"снизилась на {_num(abs(delta_mlp), 1)} %, однослойного персептрона — на "
        f"{_num(abs(delta_single), 1)} % (в режиме без масштабирования), поскольку "
        "градиентный метод работает с признаками одного порядка. Для KNN стандартизация "
        f"ухудшила MAE на {_num(abs(delta_knn), 1)} %: информативный признак e перестал "
        "доминировать в расстоянии, и возросло влияние шумового d. SVR и древовидные "
        "модели практически инвариантны к масштабированию.",
    )
    _add_body(
        doc,
        "Все рассмотренные модели, кроме парной и множественной линейной регрессии, "
        "обеспечили R > 0,999, поэтому коэффициент корреляции в данной задаче слабо "
        "различает методы. Более информативны MAE и MSE: они показывают, что переход от "
        "линейных моделей к нелинейным и ансамблевым снижает среднеквадратическую ошибку "
        f"с {_num(pair['metrics']['mse'])} до {_num(best['mse'])}, то есть более чем в "
        f"{_num(pair['metrics']['mse'] / best['mse'], 0)} раз.",
    )
    _add_body(
        doc,
        "Наилучшие результаты объясняются характером данных: целевая переменная является "
        "гладкой нелинейной функцией признаков, а признаки сильно коррелированы. "
        "Нелинейная регрессия curve_fit и ансамбли деревьев учитывают кривизну и "
        "взаимодействия, тогда как одна гиперплоскость не может описать зависимость. "
        "Однослойный персептрон с линейным выходом также ограничен линейной формой, "
        "поэтому уступает MLP и ансамблям. Наихудшие результаты у парной регрессии: "
        "единственный признак e хотя и имеет максимальную корреляцию с целью, не "
        "содержит всей информации об остальных переменных.",
    )

    # ------------------------------------------------------------------ Выводы
    _add_heading(doc, "6. Выводы", 1)
    _add_body(
        doc,
        "В ходе выполнения лабораторной работы решена задача прогнозирования временного "
        "ряда (вариант 4) широким набором методов машинного обучения: от парной и "
        "множественной линейной регрессии до ансамблей деревьев и глубоких нейронных "
        "сетей. Приобретены навыки формирования выборок со сдвигом на 1 и 7 шагов, "
        "корректного разбиения и масштабирования данных без утечки информации из "
        "тестовой выборки, подбора гиперпараметров по Validation и интерпретации MAE, "
        "MSE и R.",
    )
    _add_body(
        doc,
        "Сравнение показало, что нелинейные и ансамблевые методы значительно превосходят "
        "линейные: MSE снижается с "
        f"{_num(pair['metrics']['mse'])} (парная регрессия) до "
        f"{_num(best['mse'])} (лучшая модель). Нейросетевые модели MLP и ResNet-MLP "
        "обеспечивают точность, близкую к лучшим ансамблям, как для одношагового, так и "
        "для многошагового прогноза, а однослойный персептрон ожидаемо ограничен линейной "
        "формой зависимости. Стандартизация ускоряет и стабилизирует обучение нейронных "
        "сетей, но может ухудшать метрические методы (KNN) при наличии шумовых "
        "признаков. Цель работы достигнута: все пункты задания выполнены, результаты "
        "сведены в results_Comparison.xlsx.",
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "Лабораторная_работа_3_Вариант_4.docx"
    doc.save(report_path)
    return report_path


if __name__ == "__main__":
    print(build_report())
