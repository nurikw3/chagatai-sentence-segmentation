from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def stanza_text_and_labels(tokens: list[str], word_labels: list[int]) -> tuple[str, str]:
    """Project canonical word labels to Stanza character labels 0/1/2."""
    if not tokens or len(tokens) != len(word_labels):
        raise ValueError("Expected equally sized non-empty token and word-label lists")
    if any(label not in (0, 1) for label in word_labels):
        raise ValueError("Canonical word labels must be 0 or 1")

    chars: list[str] = []
    labels: list[str] = []
    for token_index, (token, word_label) in enumerate(zip(tokens, word_labels)):
        if not token:
            raise ValueError("Empty tokens cannot be projected to Stanza")
        if token_index:
            chars.append(" ")
            labels.append("0")
        for char_index, char in enumerate(token):
            chars.append(char)
            if char_index == len(token) - 1:
                labels.append("2" if word_label else "1")
            else:
                labels.append("0")

    text = "".join(chars)
    label_text = "".join(labels)
    if len(text) != len(label_text):
        raise AssertionError("Stanza text and label lengths differ")
    return text, label_text


def export_stanza_split(csv_path: Path, output_dir: Path, split: str) -> int:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    texts: list[str] = []
    labels: list[str] = []
    for row_number, row in enumerate(df.itertuples(index=False), start=2):
        tokens = json.loads(row.tokens)
        word_labels = json.loads(row.labels)
        text, label_text = stanza_text_and_labels(tokens, word_labels)
        if text != row.text:
            raise ValueError(f"{csv_path.name} row {row_number}: canonical text mismatch")
        texts.append(text)
        labels.append(label_text)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{split}.txt").write_text(
        "\n\n".join(texts) + "\n\n", encoding="utf-8"
    )
    (output_dir / f"{split}.toklabels").write_text(
        "\n\n".join(labels) + "\n\n", encoding="utf-8"
    )
    return len(texts)


def export_stanza_dataset(build_dir: Path, output_dir: Path) -> dict[str, int]:
    counts = {
        split: export_stanza_split(build_dir / f"{split}.csv", output_dir, split)
        for split in ("train", "dev", "test")
    }
    (output_dir / "mwt.json").write_text("[]\n", encoding="utf-8")
    return counts
