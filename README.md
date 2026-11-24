# Лабораторная работа № 3. Вариант 4

Проект на Python для Visual Studio Code. Выполняется полный цикл: загрузка
данных варианта 4 из `Lab3_Dataset.xlsx`, формирование выборок со сдвигом на
1 и 7 шагов, разбиение Train / Validation / Test, стандартизация по параметрам
Train, корреляционный анализ, парная / множественная / нелинейная регрессия,
одиночные модели (KNN, SVR, Decision Tree), ансамбли (Random Forest, Bagging,
Gradient Boosting), нейронные сети PyTorch (однослойный персептрон, MLP,
ResNet-MLP), сравнение метрик и автоматическая сборка отчёта DOCX.

#Быстрый запуск

powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py


После выполнения:

- рисунки — в results/figures/;
- числовые результаты — results/lab3_results.json;
- сохранённые прогнозы — results/predictions.npz;
- итоговая таблица — results/results_Comparison.xlsx;
- отчёт — report/Лабораторная_работа_3_Вариант_4.docx;
- версия отчёта для печати — report/Лабораторная_работа_3_Вариант_4.pdf.

Отдельные этапы:

powershell
python scripts/run_all.py         # полный цикл и сборка отчёта
python scripts/build_report.py    # только пересборка DOCX по готовому JSON
python scripts/rebuild_figures.py # пересборка рисунков по сохранённым прогнозам


# Структура

lab3_ml/
├── main.py                 точка входа
├── data/
│   └── Lab3_Dataset.xlsx   исходная книга Excel (вариант 4)
├── src/
│   ├── config.py           seed, вариант, пути, параметры
│   ├── data_loader.py      чтение блока варианта 4, сдвиги, разбиение
│   ├── base_models.py      п. 1.4–1.6
│   ├── single_models.py    п. 2.1–2.3
│   ├── ensemble_models.py  п. 3.1–3.3
│   ├── neural_nets.py      п. 4.1–4.3 (PyTorch)
│   ├── visualization.py    все рисунки
│   ├── evaluation.py       MAE, MSE, R, экспорт Excel
│   ├── report_builder.py   сборка DOCX
│   └── utils.py            сериализация JSON
├── scripts/
│   ├── run_all.py
│   ├── build_report.py
│   └── rebuild_figures.py
├── results/                создаётся автоматически
├── .vscode/                конфигурация запуска
├── requirements.txt
└── README.md


# Данные варианта 4

- 1000 наблюдений, признаки a, b, c, d, e, целевая переменная y;
- пропусков и дубликатов нет;
- для 1 шага формируется 999 пар X(t) → y(t+1), для 7 шагов — 993 пары
  X(t) → y(t+1..t+7);
- разбиение 70 % / 15 % / 15 % с random_state = 42;
- `StandardScaler` обучается только на Train.

# Основные результаты

- Наибольшая корреляция с целью у признака e (r ≈ −0,987).
- Парная линейная регрессия: MAE ≈ 1,374, MSE ≈ 2,864.
- Множественная линейная регрессия: MAE ≈ 0,728, MSE ≈ 0,778.
- Нелинейная регрессия `curve_fit` (квадратичный базис, 21 параметр):
  MAE ≈ 0,142, MSE ≈ 0,032.
- Лучшие результаты на тесте: SVR RBF (MSE ≈ 0,0054) и ансамбли
  Bagging / Random Forest (MSE ≈ 0,0061–0,0062).
- Нейронные сети MLP и ResNet-MLP: MSE ≈ 0,0096–0,0151 для 1 и 7 шагов.
- Однослойный персептрон ограничен линейной формой и уступает MLP.

Все стохастические алгоритмы используют random_state = 42 / torch.manual_seed(42),
поэтому результаты воспроизводимы.
