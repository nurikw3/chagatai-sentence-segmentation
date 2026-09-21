# Chagatai Sentence Segmentation

Репозиторий строит единый датасет для Sentence Boundary Detection. Истинная разметка хранится на уровне слов: `0` — слово не завершает предложение, `1` — EOS. Представление Stanza `0/1/2` создаётся отдельно и не смешивается с каноническими данными.

## Структура

```text
src/unified_dataset/
  cleaning.py
  sources.py
  augmentation.py
  labeling.py
  validation.py
  adapters/stanza.py
scripts/
  build_unified_dataset.py
data/UNIFIED/
  raw/       # локальные источники, не коммитятся
  builds/    # воспроизводимые сборки, не коммитятся
legacy/
  old_pipeline/
  generated_data/
```

Старые `data/CHAGATAI`, `data/UZS`, отдельные augmentation/labeling-скрипты, модели и CharLM-файлы перенесены в `legacy/` и больше не являются активными входами.

## Сборка датасета

Установить зависимости:

```bash
uv sync
```

Chagatai-only:

```bash
uv run python scripts/build_unified_dataset.py --export-stanza
```

Сбалансированный Chagatai + South Uzbek + Uyghur:

```bash
uv run python scripts/build_unified_dataset.py \
  --include-uzs \
  --include-uyghur \
  --max-uzs-sentences 3035 \
  --max-uyghur-sentences 3035 \
  --export-stanza \
  --output-dir data/UNIFIED/builds/chagatai_uzs_uyghur_balanced
```

Chagatai делится в исходном порядке 70/10/20 до аугментации. UZS и Uyghur используются только в train. Random и partial augmentation также разрешены только в train; dev/test состоят только из последовательного Chagatai.

## Выходные файлы

Каждая сборка содержит:

- `source_sentences.csv` — очищенные исходные предложения с provenance и split;
- `train.csv`, `dev.csv`, `test.csv` — канонические последовательности;
- `stats.csv` — статистика по языку и методу;
- `manifest.json` — параметры, хэши входов и результаты проверок;
- `stanza/` — опциональная проекция в `.txt` и `.toklabels`.

`validation.py` проверяет отсутствие leakage, полное покрытие источников, train-only auxiliary languages, точное восстановление токенов из source spans и корректность EOS.

## Обучение и тесты

Основное обучение выполняется в `stanza.ipynb`. Ноутбук принимает только schema 2.0 unified build и создаёт Stanza-файлы из канонической word-level разметки.

```bash
uv run pytest -q
uv run python -m compileall -q src scripts tests
```

Исторические результаты сохранены в `results.txt`; новые запуски должны указывать manifest или хэши использованного датасета.
