from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import stanza


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "stanza_chg" / "models" / "chg_sic_tokenizer.pt"
DEFAULT_TEST_CSV = BASE_DIR / "test_final.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a trained Chagatai Stanza tokenizer/SBD model."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--text", default=None)
    return parser.parse_args()


def load_text_from_row(path: Path, row_index: int) -> tuple[str, list[str]]:
    df = pd.read_csv(path, dtype=str)
    if row_index < 0 or row_index >= len(df):
        raise SystemExit(
            f"Row {row_index} is out of range for {path.name}. "
            f"Valid rows: 0-{len(df) - 1}."
        )
    row = df.iloc[row_index]
    gold = [sentence.strip() for sentence in json.loads(row["sentence_texts"])]
    return row["text"], gold


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(
            f"Model not found: {args.model}\n"
            "Train it first:\n"
            "  uv run python Chagatai_BERT_Project/train_stanza_tokenizer.py"
        )

    if args.text is None:
        text, gold = load_text_from_row(args.test_csv, args.row)
    else:
        text = args.text
        gold = []

    nlp = stanza.Pipeline(
        lang="chg",
        processors="tokenize",
        tokenize_model_path=str(args.model),
        allow_unknown_language=True,
        use_gpu=False,
        logging_level="WARNING",
    )
    doc = nlp(text)
    predicted = [sentence.text for sentence in doc.sentences]

    print("INPUT:")
    print(text)
    print("\nPREDICTED:")
    for index, sentence in enumerate(predicted, start=1):
        print(f"{index}. {sentence}")

    if gold:
        print("\nGOLD:")
        for index, sentence in enumerate(gold, start=1):
            print(f"{index}. {sentence}")
        print(f"\nCOUNTS: predicted={len(predicted)} gold={len(gold)}")


if __name__ == "__main__":
    main()
