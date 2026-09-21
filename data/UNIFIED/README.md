# Unified word-level SBD dataset

This directory stores local raw inputs and reproducible schema 2.0 builds. Both `raw/` and `builds/` are Git-ignored.

## Inputs

The builder expects:

- `raw/chagatai_pages_1_35.xlsx`;
- `raw/chagatai_pages_35_181.xlsx`;
- `raw/lutfiy_hf/data/train-00000-of-00001.parquet`;
- `raw/uyghur_corpus.csv`.

Download the Lutfiy snapshot with:

```bash
hf download tahrirchi/lutfiy \
  --repo-type dataset \
  --local-dir data/UNIFIED/raw/lutfiy_hf
```

## Canonical contract

Word labels are `0` for non-EOS and `1` for EOS. Chagatai is deduplicated and split in source order 70/10/20 before augmentation. UZS and Uyghur are train-only. Dev and test contain only sequential Chagatai.

Cleaning applies NFKC, removes punctuation, symbols, controls, and combining marks, and rejects residual Latin, Cyrillic, CJK, URL, and bibliography noise. Every sequence retains source IDs and fragment spans.

## Build

```bash
uv run python scripts/build_unified_dataset.py --export-stanza

uv run python scripts/build_unified_dataset.py \
  --include-uzs --include-uyghur \
  --max-uzs-sentences 3035 \
  --max-uyghur-sentences 3035 \
  --export-stanza \
  --output-dir data/UNIFIED/builds/chagatai_uzs_uyghur_balanced
```

The full uncapped multilingual build is intentionally not the default because it is dominated by Uyghur. Use explicit caps for controlled language balance.

## Output

- `source_sentences.csv`: normalized sources, provenance, split, and token counts;
- `train.csv`, `dev.csv`, `test.csv`: canonical sequences and word labels;
- `stats.csv`: counts by split, language, and method;
- `manifest.json`: schema, parameters, source hashes, cleaning statistics, and validation checks;
- `stanza/`: optional character-level projection produced by the adapter.

The validator checks source isolation, exact sequence overlap, auxiliary train-only policy, reconstruction from source spans, sequential coverage exactly once, and partial-merge EOS semantics.
