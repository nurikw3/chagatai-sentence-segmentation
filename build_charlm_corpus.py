"""Build raw-text corpora for Stanza charlm (contextualized char-LM) pretraining.

charlm is self-supervised (no boundary labels needed), but it still must not
see any text from the Chagatai test split — otherwise the tokenizer trained
on top of it gets an indirect data leak from eval data.

Output:
  charlm_corpus/chagatai_charlm.txt  -> Chagatai train+dev sentences only
  charlm_corpus/uzs_charlm.txt       -> full UZS corpus (no test split exists)

Usage:
  uv run python build_charlm_corpus.py
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "charlm_corpus"

CHAGATAI_DIR = BASE_DIR / "data" / "CHAGATAI"
UZS_DIR = BASE_DIR / "data" / "UZS"


def _load_module(name: str, folder: Path):
    """Import augmentation.py from a specific corpus folder without module
    name clashes (both folders have a module literally called augmentation)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, folder / "augmentation.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_chagatai_charlm_corpus() -> list[str]:
    chg = _load_module("chagatai_augmentation", CHAGATAI_DIR)

    sentences = chg.read_ods_original_sentences()
    split_index = int(len(sentences) * chg.TRAIN_RATIO)
    train_source = sentences[:split_index]
    test_source = sentences[split_index:]

    dev_size = max(1, int(len(train_source) * chg.DEV_RATIO_FROM_TRAIN))
    dev_source = train_source[-dev_size:]
    train_source = train_source[:-dev_size]

    corpus = train_source + dev_source

    # Hard safety check: make sure nothing from the held-out test split leaks in.
    test_set = set(test_source)
    leaked = [s for s in corpus if s in test_set]
    assert not leaked, f"Data leak: {len(leaked)} test sentence(s) ended up in charlm corpus"

    # Apply the same cleaning (strip punctuation, hyphens, diacritics) used
    # for tokenizer training data, so charlm sees the same character
    # distribution the tokenizer will actually see at train/inference time.
    cleaned = [chg.clean_sentence(s) for s in corpus]
    cleaned = [s for s in cleaned if s]  # drop any that became empty
    return cleaned


def build_uzs_charlm_corpus() -> list[str]:
    uzs = _load_module("uzs_augmentation", UZS_DIR)

    sentences = uzs.read_xlsx_sentences()
    sentences = sentences[: uzs.TOTAL_SENTENCES]
    # No held-out test split exists for UZS, so the whole corpus is fair game.
    cleaned = [uzs.clean_sentence(s) for s in sentences]
    cleaned = [s for s in cleaned if s]
    return cleaned


def write_corpus(sentences: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sentences) + "\n", encoding="utf-8")


def main() -> None:
    chagatai_sentences = build_chagatai_charlm_corpus()
    uzs_sentences = build_uzs_charlm_corpus()

    write_corpus(chagatai_sentences, OUT_DIR / "chagatai_charlm.txt")
    write_corpus(uzs_sentences, OUT_DIR / "uzs_charlm.txt")

    print("DONE: charlm raw-text corpora written")
    print(f"Chagatai (train+dev only, test excluded): {len(chagatai_sentences)} sentences")
    print(f"UZS (full corpus, no test split exists):   {len(uzs_sentences)} sentences")
    print(f"Output dir: {OUT_DIR}")
    print()
    print("Next step (Stanza charlm training), e.g.:")
    print("  python3 -m stanza.utils.training.run_charlm chg --forward --txt_file "
          f"{OUT_DIR / 'chagatai_charlm.txt'}")
    print("  python3 -m stanza.utils.training.run_charlm chg --backward --txt_file "
          f"{OUT_DIR / 'chagatai_charlm.txt'}")
    print("(repeat for uzs_charlm.txt if training a separate/combined charlm)")


if __name__ == "__main__":
    main()