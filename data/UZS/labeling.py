from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from augmentation import (
    STANZA_DIR,
    build_fragment_sample,
    build_sample,
    main as rebuild_from_xlsx,
    write_stanza_split,
)


BASE_DIR = Path(__file__).resolve().parent
REQUIRED_COLUMNS = {
    "sentence_texts",
    "num_boundaries",
    "tokens",
    "labels",
    "stanza_labels",
}


def relabel_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    if "sentence_texts" not in df.columns:
        raise ValueError(
            f"{path.name} has no sentence_texts column. "
            "Run augmentation.py to rebuild from lutfiy_uzs_uzn.xlsx."
        )

    rows = []
    for _, row in df.iterrows():
        sentences = json.loads(row["sentence_texts"])
        if "fragments" in row and pd.notna(row["fragments"]):
            fragments = json.loads(row["fragments"])
            boundary_at_end = json.loads(row["boundary_at_end"])
            rebuilt = build_fragment_sample(
                source_sentences=sentences,
                fragments=fragments,
                boundary_at_end=boundary_at_end,
                method=row.get("method", "relabelled"),
            )
        else:
            rebuilt = build_sample(sentences, row.get("method", "relabelled"))
        rebuilt["method"] = row.get("method", rebuilt["method"])
        rows.append(rebuilt)

    rebuilt_df = pd.DataFrame(rows)
    rebuilt_df.to_csv(path, index=False, encoding="utf-8-sig")
    return rebuilt_df


def has_no_letters(token: str) -> bool:
    return not any(ch.isalpha() or ch.isdigit() for ch in token)


def validate(df: pd.DataFrame, name: str) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    garbage_examples: list[str] = []
    for index, row in df.iterrows():
        tokens = str(row["tokens"]).split()
        labels = [int(value) for value in str(row["labels"]).split()]
        if len(tokens) != len(labels):
            raise ValueError(f"{name} row {index}: token/label count mismatch")
        if sum(labels) != int(row["num_boundaries"]):
            raise ValueError(f"{name} row {index}: labels do not match num_boundaries")
        if len(str(row["text"])) != len(str(row["stanza_labels"])):
            raise ValueError(f"{name} row {index}: Stanza char labels mismatch")
        if str(row["stanza_labels"]).count("2") != int(row["num_boundaries"]):
            raise ValueError(f"{name} row {index}: wrong Stanza boundary count")

        for token in tokens:
            if has_no_letters(token):
                garbage_examples.append(token)

    if garbage_examples:
        sample = garbage_examples[:10]
        raise ValueError(
            f"{name}: {len(garbage_examples)} garbage token(s) with no letters "
            f"leaked into training data (e.g. {sample}). "
            "Extend PUNCTUATION_RE in augmentation.py to strip these characters."
        )


def main() -> None:
    final_paths = [
        BASE_DIR / "train_final.csv",
        BASE_DIR / "dev_final.csv",
    ]

    if not all(path.exists() for path in final_paths):
        rebuild_from_xlsx()

    rebuilt = {}
    for path in final_paths:
        df = relabel_file(path)
        validate(df, path.name)
        rebuilt[path.stem.replace("_final", "")] = df

    for split_name, df in rebuilt.items():
        write_stanza_split(df, split_name)

    (STANZA_DIR / "mwt.json").write_text("[]\n", encoding="utf-8")

    print("DONE: labels validated and Stanza files refreshed")
    for name, df in rebuilt.items():
        print(f"{name}: {len(df)} samples")
    print(f"Stanza files: {STANZA_DIR}")


if __name__ == "__main__":
    main()
