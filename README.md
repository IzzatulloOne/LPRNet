# ANPR System

Система распознавания автомобильных номеров — scalable, multi-region ready.

## Архитектура

```
[Камера / изображение]
        ↓
[YOLOv10 — детекция bbox]
        ↓
[Deskew + CLAHE + Resize]
        ↓
[LPRNet — распознавание символов]
        ↓
[PostProcessor — регион + валидация + коррекция]
        ↓
[FastAPI → JSON]
```

## Быстрый старт

```bash
cp .env.example .env
# Прописать путь к весам YOLOv10 в .env
pip install -r requirements.txt
python scripts/run_server.py
```

API будет доступно на http://localhost:8000/docs

## Тесты

```bash
pytest tests/ -v
```

## Добавить новый регион (например, KZ)

1. Создать `regions/kz/rules.py` с классом `KZRegionHandler(BaseRegionHandler)`
2. В `regions/registry.py` добавить в `_autoregister()`:
   ```python
   from .kz.rules import KZRegionHandler
   RegionRegistry.register(KZRegionHandler())
   ```
3. Ядро системы не меняется.

## Структура проекта

```
anpr_system/
├── api/            FastAPI приложение
├── core/
│   ├── detection/      YOLOv10 детектор
│   ├── preprocessing/  Deskew + CLAHE
│   ├── recognition/    LPRNet / Mock
│   ├── postprocessing/ Региональная валидация
│   └── pipeline.py     Главный pipeline
├── regions/
│   ├── base.py         Абстрактный handler
│   ├── registry.py     Реестр регионов
│   └── uzb/            UZB правила
├── configs/        Settings + Logging
├── models/         Веса моделей (не в git)
├── data/           Датасет (не в git)
├── logs/           Нераспознанные + метрики
├── scripts/        CLI утилиты
└── tests/          Unit + Integration тесты
```
