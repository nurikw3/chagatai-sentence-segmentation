"""Prepare train-only corpus for CharLM pretraining with strict leakage prevention.

Ensures zero sentences from validation or test splits are included.
"""
from __future__ import annotations

import json
from pathlib import Path
from datasets import load_dataset

DATASET_REPO = "chagatai-project/chagatai-sbd"
CONFIG_NAME = "chagatai_uzs_uyghur_balanced"


def prepare_charlm_corpus(
    output_dir: Path | str = "data/charlm",
    train_ratio: float = 0.95,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET_REPO} ({CONFIG_NAME})...")
    ds = load_dataset(DATASET_REPO, CONFIG_NAME)

    train_data = ds["train"]
    val_data = ds["validation"] if "validation" in ds else ds["dev"]
    test_data = ds["test"]

    train_texts = [row["text"] for row in train_data if row["text"].strip()]
    val_texts = set(row["text"] for row in val_data if row["text"].strip())
    test_texts = set(row["text"] for row in test_data if row["text"].strip())

    # Hard safety check: assert zero overlap with val and test
    train_set = set(train_texts)
    val_overlap = train_set.intersection(val_texts)
    test_overlap = train_set.intersection(test_texts)

    assert not val_overlap, f"Data leak detected: {len(val_overlap)} val sentences found in train!"
    assert not test_overlap, f"Data leak detected: {len(test_overlap)} test sentences found in train!"

    # Split train_texts into 95% train and 5% internal dev (NO val or test sentences)
    split_idx = int(len(train_texts) * train_ratio)
    charlm_train = train_texts[:split_idx]
    charlm_dev = train_texts[split_idx:]

    train_file = output_path / "charlm_train.txt"
    dev_file = output_path / "charlm_dev.txt"

    train_file.write_text("\n".join(charlm_train) + "\n", encoding="utf-8")
    dev_file.write_text("\n".join(charlm_dev) + "\n", encoding="utf-8")

    print("CharLM corpus successfully prepared with ZERO leakage:")
    print(f"  Train sentences: {len(charlm_train)} -> {train_file}")
    print(f"  Dev sentences:   {len(charlm_dev)} -> {dev_file}")
    print(f"  Val overlap: 0, Test overlap: 0")

    return train_file, dev_file


if __name__ == "__main__":
    prepare_charlm_corpus()
