# Unified word-level SBD dataset

This directory contains reproducible, model-agnostic sentence-boundary data.
The canonical label is attached to each word:

- `0` — the word is not the end of a sentence;
- `1` — the word is the final word of a sentence (`EOS`).

No Stanza character labels or transformer subword labels are stored here.
Those representations must be derived later from the canonical word labels.

## Inputs

Raw inputs are kept under `data/UNIFIED/raw/` and are ignored by Git. The
current local build expects:

- `chagatai_pages_1_35.xlsx`;
- `chagatai_pages_35_181.xlsx`;
- `uyghur_corpus.csv`;
- the Hugging Face snapshot of `tahrirchi/lutfiy` downloaded with:

  ```bash
  hf download tahrirchi/lutfiy \
    --repo-type dataset \
    --local-dir data/UNIFIED/raw/lutfiy_hf
  ```

## Build variants

The base command creates a Chagatai-only dataset. South Uzbek and Uyghur are
independent, train-only additions:

```bash
uv run python build_unified_dataset.py
uv run python build_unified_dataset.py --include-uzs
uv run python build_unified_dataset.py --include-uyghur
uv run python build_unified_dataset.py --include-uzs --include-uyghur
```

By default the Lutfiy `books` domain is used because it contains sentence-like
running text. Use `--uzs-sources books,web,dictionary` to include every domain.
Use `--max-uzs-sentences N` and `--max-uyghur-sentences N` for deterministic
caps; `0` means no cap.

Chagatai is split in source order before augmentation: 70% train, 10% dev,
20% test. Auxiliary languages are always train-only. Dev and test contain only
sequentially concatenated Chagatai source sentences.

## Output schema

Each build directory contains:

- `source_sentences.csv` — one normalized source sentence per row, with stable
  provenance and split assignment;
- `train.csv`, `dev.csv`, `test.csv` — canonical sequences;
- `stats.csv` — counts by split, language, and augmentation method;
- `manifest.json` — parameters, input hashes, Hugging Face revision, and
  validation results.

Sequence columns:

- `sequence_id`, `split`, `language`, `method`;
- `source_sentence_ids` — JSON list referencing `source_sentences.csv`;
- `fragment_spans` — `[start, end)` token spans retained from each source;
- `boundary_at_end` — whether the end of each retained fragment is a true EOS;
- `text`, `tokens`, `labels` — cleaned text and JSON word/label arrays;
- `num_tokens`, `num_source_sentences`, `num_boundaries`.

Train uses three methods: sequential concatenation of 2–4 adjacent sentences,
random concatenation of 2–5 distinct sentences, and partial boundary merge.
For partial merge the retained tail of sentence 1 ends in `1`, while the final
token of the retained head of sentence 2 remains `0`.
