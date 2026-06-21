<p align="center">
  <a href="https://commons.wikimedia.org/wiki/File:Paper_Scroll_3.svg">
    <img src="https://upload.wikimedia.org/wikipedia/commons/2/2c/Paper_Scroll_3.svg" alt="Paper scroll" width="96">
  </a>
</p>

<h1 align="center">Chagatai Sentence Segmentation</h1>

Пайплайн обучает custom Stanza tokenizer для сегментации чагатайского текста
без пунктуации. Границы предложений берутся из строк `Original` файла
`Dataset_OCR.ods`.

## Запуск

Установить зависимости:

```bash
uv sync
```

Пересобрать train/dev/test и Stanza labels из ODS:

```bash
uv run python augmentation.py
uv run python labeling.py
```

Быстрое тестовое обучение:

```bash
uv run python train_stanza_tokenizer.py \
  --steps 20 \
  --eval-steps 5 \
  --report-steps 5 \
  --device mps
```

Основное обучение:

```bash
uv run python train_stanza_tokenizer.py \
  --steps 2000 \
  --eval-steps 100 \
  --report-steps 25 \
  --early-stop-steps 500 \
  --device mps
```

На машине без Apple MPS замените `--device mps` на `--device cpu`.

Проверить модель на строке из test split:

```bash
uv run python check_stanza_tokenizer.py --row 0
```

Проверить на своем тексте:

```bash
uv run python check_stanza_tokenizer.py \
  --text "چاغاتای متنی"
```

## Результаты

- CSV с token labels: `*_final.csv`
- Stanza labels: `stanza_chg/tokenizer/`
- Обученная модель: `stanza_chg/models/chg_sic_tokenizer.pt`
