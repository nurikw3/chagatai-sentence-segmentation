# Repository Guidelines

## Project Structure & Module Organization

This repository trains and evaluates a custom Stanza tokenizer for Chagatai sentence segmentation. Root-level scripts are the main workflow entry points: `train_stanza_tokenizer.py`, `evaluate_stanza_tokenizer.py`, `check_stanza_tokenizer.py`, and `build_charlm_corpus.py` (builds raw-text corpora for Stanza's pretrained charlm). Active Stanza tokenizer inputs and trained models live in `stanza_chg/tokenizer/` and `stanza_chg/models/`. `stanza.ipynb` (repo root) is the primary training/experimentation entry point, run on the Jupyter server with 2x NVIDIA T4 GPUs already configured; root-level `.py` scripts (`train_stanza_tokenizer.py`, etc.) are for quick local/CPU smoke tests only.

Corpus-specific preprocessing is under `data/CHAGATAI/` and `data/UZS/`, including `augmentation.py`, `labeling.py`, generated `*_final.csv` files, and source spreadsheet assets. Keep large or generated data clearly separated from hand-edited scripts.

Note the datasets are not symmetric: `data/CHAGATAI/` has `train_final.csv`, `dev_final.csv`, **and** `test_final.csv`; `data/UZS/` only has `train_final.csv` and `dev_final.csv` (no UZS test split). This is why cross-lingual evaluation runs (train on UZS, test on Chagatai) use `data/CHAGATAI/test_final.csv` as the eval set — see `results.txt`.

On Kaggle, these local folder names don't carry over — the datasets are mounted under different names:
- `data/UZS/` → `/kaggle/input/datasets/nurikw3/lutify-2`
- `data/CHAGATAI/` → `/kaggle/input/datasets/nurikw3/chagatai-test`
- `charlm_corpus/` (output of `build_charlm_corpus.py`) → `/kaggle/input/datasets/nurikw3/charlm-corpus`

Keep this mapping in mind when reading paths in `stanza.ipynb` or in `results.txt` logs — `chagatai-test` on Kaggle is the full Chagatai corpus (not just the test split; the name is just the dataset slug).

## Build, Test, and Development Commands

- `uv sync`: install dependencies from `pyproject.toml` and `uv.lock`.
- `uv run python train_stanza_tokenizer.py --steps 20 --eval-steps 5 --report-steps 5 --device cpu`: run a quick smoke training pass. Use `--device mps` on Apple Silicon when available.
- `uv run python evaluate_stanza_tokenizer.py --device cpu`: evaluate the trained tokenizer on the test split.
- `uv run python check_stanza_tokenizer.py --text "چاغاتای متنی"`: inspect sentence predictions for custom text.
- `(cd data/CHAGATAI && uv run python augmentation.py && uv run python labeling.py)`: regenerate Chagatai CSVs and labels from `Dataset_OCR.ods`. Note that root training reads `stanza_chg/tokenizer/`; keep regenerated tokenizer files synchronized with that location before training.
- `(cd data/UZS && uv run python augmentation.py && uv run python labeling.py)`: regenerate South Uzbek CSVs and labels using the same augmentation pipeline; UZS is currently the larger of the two corpora and the main augmentation volume comes from here.
- `uv run python build_charlm_corpus.py`: build raw-text corpora for Stanza charlm pretraining. Writes `charlm_corpus/chagatai_charlm.txt` (Chagatai **train+dev only** — test split is excluded via a hard assert to prevent leakage) and `charlm_corpus/uzs_charlm.txt` (full UZS corpus, no test split exists there). Both files are cleaned with the same `clean_sentence()` used for tokenizer training data, so charlm sees the same character distribution the tokenizer will.

## Coding Style & Naming Conventions

Use Python 3.11+ and standard PEP 8 formatting: 4-space indentation, snake_case functions and variables, and UPPER_CASE constants. Prefer `pathlib.Path` for paths, type hints for public helpers, and small functions with explicit arguments. Keep scripts runnable with a `main()` guard. Preserve UTF-8 handling for Arabic-script text and CSV output.

## Testing Guidelines

There is no formal test suite yet. Use the quick training command, `evaluate_stanza_tokenizer.py`, and targeted `check_stanza_tokenizer.py` runs as regression checks. When adding tests, place them in `tests/`, name files `test_*.py`, and prefer small fixtures over committing additional large generated datasets.

Append every evaluation run (in-domain and cross-lingual) to `results.txt` at the repo root — one dated section per run with the setup (train/eval data used), the optimizer config, and the full token/sentence boundary metrics. Do not overwrite previous entries; this file is the running log of experiment results.

## Commit & Pull Request Guidelines

Git history currently uses short conventional-style messages such as `docs: upd`. Continue with concise prefixes like `docs:`, `data:`, `fix:`, or `train:` followed by an imperative summary.

Pull requests should describe the changed workflow or dataset, list commands run, and include before/after evaluation metrics when model behavior changes. Link related issues when available and call out any regenerated CSV, tokenizer label, or model artifact.

## Security & Configuration Tips

Do not commit local virtual environments, private datasets, or machine-specific paths. Before replacing `.pt` model files or generated labels, verify the source data and record the exact command used to produce them.

## About Project and Data
This project tackles Chagatai **Sentence Boundary Detection (SBD)**: splitting a large stream of Chagatai words (no punctuation) into sentences. Modeling is done exclusively through **Stanza** (no BERT/token-classification path, despite what `Report.docx` implies for an earlier approach).

Two datasets are used, both in Arabic script:
- **Chagatai (`data/CHAGATAI/`)** — the primary, original-language dataset, but very small in volume.
- **UZS (`data/UZS/`)** — South Uzbek, the closest related language, used to compensate for the limited size of the Chagatai data; it is currently the larger of the two corpora. Exact integration strategy (joint training vs. pretrain/fine-tune vs. fallback) is not finalized yet — treat as an open design decision, not a fixed pipeline.

Both datasets are available locally under `data/` and on the Jupyter server.

**Data augmentation**: three methods are applied to both datasets via `augmentation.py` / `labeling.py` in each corpus folder:
- *Sequential Concatenation* — consecutive sentences joined in original order (natural boundaries).
- *Random Shuffling* — sentences from random, unrelated parts of the corpus combined together (train only).
- *Partial Merge* — the start of one sentence spliced with the end of another, creating unnatural boundaries to help the model learn internal transitions (train only).

**Jupyter server**: all training and experimentation should happen in `stanza.ipynb` (repo root), which already has server access configured with **2x NVIDIA T4 GPUs**. Local CLI scripts (`train_stanza_tokenizer.py`, etc.) are for quick smoke tests on CPU/local machine; substantive training runs belong in the notebook.

## Data Cleaning
`clean_sentence()` in each corpus's `augmentation.py` strips punctuation, hyphens/dashes (`-–—`), and Arabic diacritics (harakat/tashkeel) before tokens are built — none of these should appear in the `text`/`tokens`/`labels`/`stanza_labels` columns of `*_final.csv`. `original_text` intentionally keeps the raw, uncleaned source for debugging/traceability — dashes/diacritics showing up there are expected, not a bug.

This was fixed after finding real leakage: hyphens (`-`) appeared in ~22% of UZS sentences (dialogue dashes) and were not being stripped; Arabic diacritics sometimes floated as standalone OCR-artifact tokens; and one UZS "sentence" (`"- ..."`) was pure punctuation and collapsed to zero tokens, breaking boundary-count validation. `labeling.py`'s `validate()` now asserts no token in any `*_final.csv` is letter-and-digit-free — rerun `labeling.py` after any `augmentation.py` change to catch regressions.

## charlm (contextualized character LM)
`build_charlm_corpus.py` (repo root) builds the raw-text corpora needed to pretrain Stanza's optional `--charlm` embeddings for the tokenizer. charlm is self-supervised (no boundary labels), but the Chagatai **test split must still never appear in it** — the script re-derives the exact same train/dev/test split as `data/CHAGATAI/augmentation.py` (same seed/ratios) and hard-asserts no test sentence leaks into the output. UZS has no test split, so its full corpus is used as-is. Both outputs are passed through the same `clean_sentence()` as the tokenizer training data (see Data Cleaning above), so charlm's character distribution matches what the tokenizer actually sees.

Output: `charlm_corpus/chagatai_charlm.txt` (train+dev only, ~695 sentences) and `charlm_corpus/uzs_charlm.txt` (full corpus, ~19000 sentences). On Kaggle this is uploaded as its own dataset at `/kaggle/input/datasets/nurikw3/charlm-corpus`. Train with `stanza.utils.training.run_charlm` (forward + backward per language), then pass `--charlm` / `--charlm_shorthand` to the tokenizer training command.