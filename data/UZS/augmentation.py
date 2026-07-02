from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
XLSX_PATH = BASE_DIR / "lutfiy_uzs_uzn.xlsx"
OUTPUT_DIR = BASE_DIR
STANZA_DIR = BASE_DIR / "stanza_uzs" / "tokenizer"

RANDOM_SEED = 42
TOTAL_SENTENCES = 19000
TRAIN_SENTENCES = 16000
DEV_SENTENCES = 3000
RANDOM_TRAIN_SAMPLES = 1900
PARTIAL_MERGE_SAMPLES = 1900

PUNCTUATION_RE = re.compile(r"""[.۔!?؟,:;؛،()\[\]{}"“”‘’«»*ـ_/|\-–—№]""")
# Arabic diacritics (harakat/tashkeel) are irrelevant for sentence boundary
# detection, and OCR noise sometimes leaves them floating as standalone
# "tokens" separated by whitespace from their base letter. Strip them
# entirely rather than trying to special-case the floating ones.
DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")
SPACE_RE = re.compile(r"\s+")


def normalize_spaces(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def clean_sentence(text: str) -> str:
    text = text.replace("\u200c", " ").replace("\u200d", " ")
    text = DIACRITICS_RE.sub("", text)
    text = PUNCTUATION_RE.sub(" ", text)
    return normalize_spaces(text)


def read_xlsx_sentences(path: Path = XLSX_PATH) -> list[str]:
    df = pd.read_excel(path)
    df = df[df["source"] == "books"].copy()
    sentences = df["tgt_sent"].dropna().astype(str).str.strip().tolist()
    sentences = [normalize_spaces(s) for s in sentences if s.strip()]
    # Drop "sentences" that are pure punctuation/formatting artifacts (e.g. "- ...")
    # and would collapse to zero tokens after clean_sentence, which breaks
    # boundary-count validation downstream.
    return [s for s in sentences if clean_sentence(s)]


def token_labels_from_fragments(
    fragments: Iterable[str], boundary_at_end: Iterable[bool]
) -> tuple[list[str], list[int]]:
    tokens: list[str] = []
    labels: list[int] = []

    for fragment, is_boundary in zip(fragments, boundary_at_end):
        fragment_tokens = clean_sentence(fragment).split()
        if not fragment_tokens:
            continue

        for index, token in enumerate(fragment_tokens):
            tokens.append(token)
            labels.append(
                1 if is_boundary and index == len(fragment_tokens) - 1 else 0
            )

    return tokens, labels


def token_labels_from_sentences(sentences: Iterable[str]) -> tuple[list[str], list[int]]:
    sentence_list = list(sentences)
    return token_labels_from_fragments(sentence_list, [True] * len(sentence_list))


def stanza_text_and_labels(tokens: list[str], token_labels: list[int]) -> tuple[str, str]:
    chars: list[str] = []
    labels: list[str] = []

    for token_index, (token, token_label) in enumerate(zip(tokens, token_labels)):
        if token_index:
            chars.append(" ")
            labels.append("0")

        for char_index, char in enumerate(token):
            chars.append(char)
            if char_index == len(token) - 1:
                labels.append("2" if token_label else "1")
            else:
                labels.append("0")

    text = "".join(chars)
    label_text = "".join(labels)
    if len(text) != len(label_text):
        raise ValueError("Stanza text and label lengths do not match")

    return text, label_text


def build_fragment_sample(
    source_sentences: list[str],
    fragments: list[str],
    boundary_at_end: list[bool],
    method: str,
) -> dict[str, object]:
    tokens, labels = token_labels_from_fragments(fragments, boundary_at_end)
    text, stanza_labels = stanza_text_and_labels(tokens, labels)

    return {
        "original_text": " ".join(source_sentences),
        "text": text,
        "method": method,
        "num_sentences": len(source_sentences),
        "num_boundaries": sum(boundary_at_end),
        "sentence_texts": json.dumps(source_sentences, ensure_ascii=False),
        "fragments": json.dumps(fragments, ensure_ascii=False),
        "boundary_at_end": json.dumps(boundary_at_end),
        "tokens": " ".join(tokens),
        "labels": " ".join(str(label) for label in labels),
        "stanza_labels": stanza_labels,
    }


def build_sample(sentences: list[str], method: str) -> dict[str, object]:
    return build_fragment_sample(
        source_sentences=sentences,
        fragments=sentences,
        boundary_at_end=[True] * len(sentences),
        method=method,
    )


def sequential_concatenation(
    sentences: list[str],
    rng: random.Random,
    min_size: int = 2,
    max_size: int = 4,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    index = 0

    while index + min_size <= len(sentences):
        group_size = rng.randint(min_size, max_size)
        if index + group_size > len(sentences):
            group_size = len(sentences) - index
        if group_size < min_size:
            break

        group = sentences[index : index + group_size]
        samples.append(build_sample(group, "sequential"))
        index += group_size

    return samples


def random_shuffling(
    sentences: list[str],
    rng: random.Random,
    num_samples: int = RANDOM_TRAIN_SAMPLES,
    min_size: int = 2,
    max_size: int = 5,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []

    for _ in range(num_samples):
        group_size = rng.randint(min_size, max_size)
        group = rng.sample(sentences, group_size)
        samples.append(build_sample(group, "random_shuffling"))

    return samples


def partial_merge(
    sentences: list[str],
    rng: random.Random,
    num_samples: int = PARTIAL_MERGE_SAMPLES,
) -> list[dict[str, object]]:
    eligible = [
        sentence for sentence in sentences if len(clean_sentence(sentence).split()) >= 2
    ]
    if len(eligible) < 2:
        return []

    samples: list[dict[str, object]] = []
    for _ in range(num_samples):
        first, second = rng.sample(eligible, 2)
        first_tokens = clean_sentence(first).split()
        second_tokens = clean_sentence(second).split()

        first_cut = rng.randint(1, len(first_tokens) - 1)
        second_cut = rng.randint(1, len(second_tokens) - 1)
        first_tail = " ".join(first_tokens[first_cut:])
        second_head = " ".join(second_tokens[:second_cut])

        samples.append(
            build_fragment_sample(
                source_sentences=[first, second],
                fragments=[first_tail, second_head],
                boundary_at_end=[True, False],
                method="partial_merge",
            )
        )

    return samples


def write_stanza_split(df: pd.DataFrame, split_name: str) -> None:
    STANZA_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = STANZA_DIR / f"{split_name}.txt"
    label_path = STANZA_DIR / f"{split_name}.toklabels"

    texts = df["text"].astype(str).tolist()
    labels = df["stanza_labels"].astype(str).tolist()

    for text, label in zip(texts, labels):
        if len(text) != len(label):
            raise ValueError(f"Bad Stanza labels in {split_name}: {text[:80]}")

    txt_path.write_text("\n\n".join(texts) + "\n\n", encoding="utf-8")
    label_path.write_text("\n\n".join(labels) + "\n\n", encoding="utf-8")


def write_outputs(train_df: pd.DataFrame, dev_df: pd.DataFrame) -> None:
    train_seq_df = train_df[train_df["method"] == "sequential"].copy()
    train_seq_df.to_csv(OUTPUT_DIR / "train_sequential.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(OUTPUT_DIR / "train_final.csv", index=False, encoding="utf-8-sig")
    dev_df.to_csv(OUTPUT_DIR / "dev_final.csv", index=False, encoding="utf-8-sig")

    write_stanza_split(train_df, "train")
    write_stanza_split(dev_df, "dev")
    (STANZA_DIR / "mwt.json").write_text("[]\n", encoding="utf-8")


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    sentences = read_xlsx_sentences()
    sentences = sentences[:TOTAL_SENTENCES]

    train_source = sentences[:TRAIN_SENTENCES]
    dev_source = sentences[TRAIN_SENTENCES:]

    train_samples = sequential_concatenation(train_source, rng)
    train_samples.extend(random_shuffling(train_source, rng))
    train_samples.extend(partial_merge(train_source, rng))
    dev_samples = sequential_concatenation(dev_source, rng)

    train_df = pd.DataFrame(train_samples)
    dev_df = pd.DataFrame(dev_samples)
    write_outputs(train_df, dev_df)

    print("DONE: sentence boundary datasets created")
    print(f"Source sentences: {len(sentences)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Dev samples: {len(dev_df)}")
    print(f"Stanza files: {STANZA_DIR}")
    print(train_df["method"].value_counts().to_string())


if __name__ == "__main__":
    main()
