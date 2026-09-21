# Repository Guidelines

## Project structure

This repository builds a canonical, model-agnostic word-level dataset for Chagatai Sentence Boundary Detection and projects it to Stanza character labels through an adapter.

Active code is split by responsibility under `src/unified_dataset/`:

- `cleaning.py` — Unicode normalization and deterministic noise filters;
- `sources.py` — source readers, provenance, deduplication, and the 70/10/20 Chagatai split;
- `augmentation.py` — sequential, random, and partial-merge sequence construction;
- `labeling.py` — canonical word-level O/EOS records;
- `validation.py` — leakage, provenance, coverage, and label validation;
- `adapters/stanza.py` — canonical word labels to Stanza 0/1/2 character labels.

`scripts/build_unified_dataset.py` is the main CLI. Raw inputs and generated builds live under `data/UNIFIED/raw/` and `data/UNIFIED/builds/` and are intentionally Git-ignored. `stanza.ipynb` is the primary Kaggle/Jupyter training entry point.

Historical scripts and notebooks are under `legacy/old_pipeline/`. Historical generated CSVs, tokenizer inputs, models, CharLM files, and experiment artifacts are under `legacy/generated_data/`. They are not active training inputs.

## Kaggle notebook workflow

When a workflow must run on Kaggle, put Kaggle-only helper code directly in `stanza.ipynb`; Kaggle will not automatically have a newly created local helper script. The notebook discovers a schema 2.0 unified build and contains its Stanza projection inline.

After every programmatic notebook edit, parse every code cell with `ast.parse`. Keep each source line as a valid JSON string with escaped newlines.

## Build and test commands

- `uv sync`: install dependencies from `pyproject.toml` and `uv.lock`.
- `uv run pytest -q`: run the unit test suite.
- `uv run python -m compileall -q src scripts tests`: run a syntax smoke check.
- `uv run python scripts/build_unified_dataset.py --export-stanza`: build Chagatai-only canonical and Stanza data.
- `uv run python scripts/build_unified_dataset.py --include-uzs --include-uyghur --max-uzs-sentences 3035 --max-uyghur-sentences 3035 --export-stanza --output-dir data/UNIFIED/builds/chagatai_uzs_uyghur_balanced`: build the balanced multilingual variant.

## Data contract

The canonical label is attached to each word: `0` means not EOS and `1` means EOS. Chagatai source sentences are deduplicated and split in source order 70/10/20 before augmentation. South Uzbek and Uyghur are train-only. Dev and test contain only sequentially concatenated Chagatai.

Train augmentation has three methods:

- sequential concatenation of 2–4 adjacent source sentences;
- random concatenation of 2–5 distinct train sentences;
- partial merge of the tail of one sentence and the head of another, with EOS only at the end of the first fragment.

`clean_sentence()` in `src/unified_dataset/cleaning.py` applies NFKC, removes punctuation, symbols, controls, and combining marks, and normalizes whitespace. Source filtering rejects URL/bibliography remnants and residual Latin, Cyrillic, or CJK letters. `raw_text` intentionally preserves the original source for traceability.

Every generated sequence must reconstruct exactly from its `source_sentence_ids` and `[start, end)` `fragment_spans`. Sequential rows must cover each selected source exactly once. Do not weaken provenance, leakage, exact coverage, or reconstruction checks to make a build pass.

## Coding style

Use Python 3.11+, four-space indentation, snake_case names, UPPER_CASE constants, `pathlib.Path`, type hints for public helpers, and small functions with explicit arguments. Keep scripts runnable with a `main()` guard and preserve UTF-8 Arabic-script text.

## Tests and experiment results

Tests live in `tests/` and use small fixtures. Every pipeline change must run the unit suite and an end-to-end balanced build.

Append every model evaluation to `results.txt` at the repository root. Use one dated section per run and include the exact dataset manifest or hashes, optimizer configuration, and complete token/sentence-boundary metrics. Do not overwrite previous entries.

Substantive training belongs in `stanza.ipynb`, which targets the configured 2x NVIDIA T4 Jupyter/Kaggle environment. The notebook must not inspect test text or labels until final evaluation.

## Commits and security

Use concise conventional-style commit prefixes such as `docs:`, `data:`, `fix:`, or `train:`. Pull requests should list commands run and before/after metrics when model behavior changes.

Do not commit private corpora, local virtual environments, machine-specific paths, or newly generated model artifacts without explicit approval. Verify source hashes and record the exact build command before publishing a dataset or model.
