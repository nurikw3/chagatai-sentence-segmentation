from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "UNIFIED" / "raw"
BUILDS_DIR = PROJECT_ROOT / "data" / "UNIFIED" / "builds"

DEFAULT_CHAGATAI_1_35 = RAW_DIR / "chagatai_pages_1_35.xlsx"
DEFAULT_CHAGATAI_35_181 = RAW_DIR / "chagatai_pages_35_181.xlsx"
DEFAULT_UZS_PARQUET = (
    RAW_DIR / "lutfiy_hf" / "data" / "train-00000-of-00001.parquet"
)
DEFAULT_UYGHUR_CSV = RAW_DIR / "uyghur_corpus.csv"

TERMINAL_PUNCTUATION_RE = re.compile(r"[.!?\u061f\u06d4\u2026]+")
PARENTHETICAL_FOOTNOTE_RE = re.compile(
    r"[\(\[\{]\s*[0-9\u0660-\u0669\u06f0-\u06f9]+\s*[\)\]\}]"
)
SPACE_RE = re.compile(r"\s+")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
URL_REMNANT_RE = re.compile(
    r"(?:^|\s)(?:https?|www|com|org|net|html?|php)(?:\s|$)", re.IGNORECASE
)
MIN_AUXILIARY_ARABIC_SCRIPT_RATIO = 0.75


@dataclass(frozen=True)
class SourceSentence:
    source_sentence_id: str
    language: str
    source_order: int
    source_dataset: str
    source_locator: str
    raw_text: str
    cleaned_text: str
    tokens: tuple[str, ...]
    split: str = "train"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_spaces(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def clean_sentence(text: str) -> str:
    """Return punctuation-free word text while preserving letters and digits."""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u200c", " ").replace("\u200d", " ")
    text = PARENTHETICAL_FOOTNOTE_RE.sub(" ", text)

    cleaned: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category[0] == "M":
            continue
        if category[0] in {"P", "S"} or category == "Cf":
            cleaned.append(" ")
        else:
            cleaned.append(char)
    return normalize_spaces("".join(cleaned))


def has_word_character(token: str) -> bool:
    return any(char.isalpha() or char.isdigit() for char in token)


def is_arabic_script_character(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x0600 <= codepoint <= 0x06FF
        or 0x0750 <= codepoint <= 0x077F
        or 0x08A0 <= codepoint <= 0x08FF
        or 0xFB50 <= codepoint <= 0xFDFF
        or 0xFE70 <= codepoint <= 0xFEFF
    )


def arabic_script_ratio(text: str) -> float:
    arabic = 0
    other_script_letters = 0
    for char in text:
        name = unicodedata.name(char, "")
        if is_arabic_script_character(char):
            arabic += 1
        elif any(
            script in name
            for script in ("LATIN", "CYRILLIC", "CJK", "IDEOGRAPH")
        ):
            other_script_letters += 1
    total = arabic + other_script_letters
    return arabic / total if total else 0.0


def auxiliary_noise_reasons(cleaned_text: str) -> tuple[str, ...]:
    reasons: list[str] = []
    if arabic_script_ratio(cleaned_text) < MIN_AUXILIARY_ARABIC_SCRIPT_RATIO:
        reasons.append("low_arabic_script_ratio")
    if URL_REMNANT_RE.search(cleaned_text):
        reasons.append("url_or_domain_remnant")
    return tuple(reasons)


def deduplicate_cleaned_sources(
    sources: Sequence[SourceSentence],
) -> tuple[list[SourceSentence], int]:
    seen: set[str] = set()
    kept: list[SourceSentence] = []
    dropped = 0
    for source in sources:
        if source.cleaned_text in seen:
            dropped += 1
            continue
        seen.add(source.cleaned_text)
        kept.append(source)
    return kept, dropped


def make_source_sentence(
    *,
    source_sentence_id: str,
    language: str,
    source_order: int,
    source_dataset: str,
    source_locator: str,
    raw_text: str,
) -> SourceSentence | None:
    raw_text = normalize_spaces(raw_text)
    cleaned_text = clean_sentence(raw_text)
    tokens = tuple(cleaned_text.split())
    if not tokens or not all(has_word_character(token) for token in tokens):
        return None
    return SourceSentence(
        source_sentence_id=source_sentence_id,
        language=language,
        source_order=source_order,
        source_dataset=source_dataset,
        source_locator=source_locator,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        tokens=tokens,
    )


def read_chagatai_sources(
    early_path: Path,
    late_path: Path,
) -> tuple[list[SourceSentence], dict[str, object]]:
    early_df = pd.read_excel(early_path)
    required_early = {"Original", "Pages"}
    if not required_early.issubset(early_df.columns):
        raise ValueError(
            f"{early_path} must contain columns {sorted(required_early)}"
        )
    early_df = early_df.copy()
    early_df["_page"] = early_df["Pages"].ffill()

    late_df = pd.read_excel(late_path)
    required_late = {"Num", "Original", "sentence_id"}
    if not required_late.issubset(late_df.columns):
        raise ValueError(
            f"{late_path} must contain columns {sorted(required_late)}"
        )

    # Page 35 is present in both workbooks with slightly different OCR. Keep the
    # newer, sentence-aligned copy from the 35-181 workbook exactly once.
    early_rows = early_df[
        early_df["Original"].notna() & (early_df["_page"].fillna(0) < 35)
    ]

    # Three page pairs in the 35-181 workbook are exact duplicated exports.
    # Detect page-level duplicates from their ordered Original rows instead of
    # hard-coding page numbers.
    seen_page_fingerprints: dict[tuple[str, ...], int] = {}
    kept_late_indexes: list[int] = []
    duplicate_pages: dict[int, int] = {}
    for page_number, page_df in late_df.groupby("Num", sort=False):
        fingerprint = tuple(
            normalize_spaces(value)
            for value in page_df["Original"].fillna("").astype(str).tolist()
        )
        page_number = int(page_number)
        if fingerprint in seen_page_fingerprints:
            duplicate_pages[page_number] = seen_page_fingerprints[fingerprint]
            continue
        seen_page_fingerprints[fingerprint] = page_number
        kept_late_indexes.extend(page_df.index.tolist())
    late_rows = late_df.loc[kept_late_indexes]

    sources: list[SourceSentence] = []
    dropped_empty = 0

    for row_index, row in early_rows.iterrows():
        page = int(row["_page"]) if pd.notna(row["_page"]) else "unknown"
        source = make_source_sentence(
            source_sentence_id=f"chg_{len(sources) + 1:06d}",
            language="chg",
            source_order=len(sources),
            source_dataset="chagatai_pages_1_35",
            source_locator=f"excel_row={row_index + 2};page={page}",
            raw_text=str(row["Original"]),
        )
        if source is None:
            dropped_empty += 1
        else:
            sources.append(source)

    for row_index, row in late_rows.iterrows():
        source = make_source_sentence(
            source_sentence_id=f"chg_{len(sources) + 1:06d}",
            language="chg",
            source_order=len(sources),
            source_dataset="chagatai_pages_35_181",
            source_locator=(
                f"excel_row={row_index + 2};page={int(row['Num'])};"
                f"sentence_id={int(row['sentence_id'])}"
            ),
            raw_text=str(row["Original"]),
        )
        if source is None:
            dropped_empty += 1
        else:
            sources.append(source)

    sources, dropped_duplicates = deduplicate_cleaned_sources(sources)
    metadata = {
        "early_rows_included": len(early_rows),
        "late_rows_included_before_cleaning": len(late_rows),
        "duplicate_late_pages_dropped": duplicate_pages,
        "duplicate_cleaned_sentences_dropped": dropped_duplicates,
        "empty_rows_dropped": dropped_empty,
        "missing_early_page_markers": sorted(
            set(range(1, 35))
            - {int(value) for value in early_df["Pages"].dropna().tolist() if value < 35}
        ),
    }
    return sources, metadata


def split_chagatai_sources(
    sources: Sequence[SourceSentence],
    train_ratio: float = 0.70,
    dev_ratio: float = 0.10,
) -> list[SourceSentence]:
    if not sources:
        raise ValueError("No Chagatai source sentences were found")
    if train_ratio <= 0 or dev_ratio <= 0 or train_ratio + dev_ratio >= 1:
        raise ValueError("Expected positive train/dev ratios with a non-empty test split")

    train_end = math.floor(len(sources) * train_ratio)
    dev_end = train_end + math.floor(len(sources) * dev_ratio)
    result: list[SourceSentence] = []
    for index, source in enumerate(sources):
        if index < train_end:
            split = "train"
        elif index < dev_end:
            split = "dev"
        else:
            split = "test"
        result.append(replace(source, split=split))
    return result


def read_uzs_sources(
    parquet_path: Path,
    allowed_sources: set[str],
) -> tuple[list[SourceSentence], dict[str, object]]:
    try:
        df = pd.read_parquet(parquet_path)
    except ImportError as error:
        raise RuntimeError(
            "Reading Lutfiy requires pyarrow. Run `uv sync` after updating dependencies."
        ) from error

    required = {"tgt_sent", "tgt_lang", "source"}
    if not required.issubset(df.columns):
        raise ValueError(f"{parquet_path} must contain columns {sorted(required)}")
    bad_languages = sorted(set(df["tgt_lang"].dropna()) - {"uzs_Arab"})
    if bad_languages:
        raise ValueError(f"Unexpected Lutfiy target languages: {bad_languages}")

    if allowed_sources:
        selected_df = df[df["source"].isin(allowed_sources)]
    else:
        selected_df = df

    sources: list[SourceSentence] = []
    seen_cleaned: set[str] = set()
    dropped_empty = 0
    dropped_duplicates = 0
    dropped_noise = 0
    noise_reason_counts: Counter[str] = Counter()
    for row_index, row in selected_df.iterrows():
        if pd.isna(row["tgt_sent"]):
            dropped_empty += 1
            continue
        source = make_source_sentence(
            source_sentence_id=f"uzs_hf_{row_index + 1:06d}",
            language="uzs",
            source_order=len(sources),
            source_dataset="tahrirchi/lutfiy",
            source_locator=f"parquet_row={row_index};domain={row['source']}",
            raw_text=str(row["tgt_sent"]),
        )
        if source is None:
            dropped_empty += 1
            continue
        noise_reasons = auxiliary_noise_reasons(source.cleaned_text)
        if noise_reasons:
            dropped_noise += 1
            noise_reason_counts.update(noise_reasons)
            continue
        if source.cleaned_text in seen_cleaned:
            dropped_duplicates += 1
            continue
        seen_cleaned.add(source.cleaned_text)
        sources.append(source)

    metadata = {
        "rows_in_repository": len(df),
        "allowed_sources": sorted(allowed_sources),
        "rows_after_source_filter": len(selected_df),
        "empty_rows_dropped": dropped_empty,
        "noise_rows_dropped": dropped_noise,
        "noise_reason_counts": dict(sorted(noise_reason_counts.items())),
        "duplicate_cleaned_sentences_dropped": dropped_duplicates,
    }
    return sources, metadata


def strip_markdown_line(line: str) -> str | None:
    line = line.strip()
    if not line:
        return None
    if re.match(r"^(#{1,6}\s|---+$|\*{3,}$)", line):
        return None
    if line.startswith("**") and ":" in line[:80]:
        return None
    line = MARKDOWN_LINK_RE.sub(r"\1", line)
    line = line.replace("**", " ").replace("__", " ").replace("`", " ")
    return normalize_spaces(line)


def split_uyghur_article(text: str) -> Iterable[str]:
    for raw_line in str(text).splitlines():
        line = strip_markdown_line(raw_line)
        if not line:
            continue
        for part in TERMINAL_PUNCTUATION_RE.split(line):
            part = normalize_spaces(part)
            if part:
                yield part


def read_uyghur_sources(
    csv_path: Path,
) -> tuple[list[SourceSentence], dict[str, object]]:
    sources: list[SourceSentence] = []
    seen_cleaned: set[str] = set()
    dropped_empty = 0
    dropped_duplicates = 0
    dropped_noise = 0
    noise_reason_counts: Counter[str] = Counter()
    article_count = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "text" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain a text column")
        for article_index, row in enumerate(reader, start=1):
            article_count += 1
            for segment_index, raw_sentence in enumerate(
                split_uyghur_article(row["text"]), start=1
            ):
                source = make_source_sentence(
                    source_sentence_id=(
                        f"uig_article_{article_index:05d}_segment_{segment_index:04d}"
                    ),
                    language="uig",
                    source_order=len(sources),
                    source_dataset="uyghur_corpus",
                    source_locator=(
                        f"csv_row={article_index + 1};segment={segment_index}"
                    ),
                    raw_text=raw_sentence,
                )
                if source is None:
                    dropped_empty += 1
                    continue
                noise_reasons = auxiliary_noise_reasons(source.cleaned_text)
                if noise_reasons:
                    dropped_noise += 1
                    noise_reason_counts.update(noise_reasons)
                    continue
                if source.cleaned_text in seen_cleaned:
                    dropped_duplicates += 1
                    continue
                seen_cleaned.add(source.cleaned_text)
                sources.append(source)

    metadata = {
        "articles": article_count,
        "empty_segments_dropped": dropped_empty,
        "noise_rows_dropped": dropped_noise,
        "noise_reason_counts": dict(sorted(noise_reason_counts.items())),
        "duplicate_cleaned_sentences_dropped": dropped_duplicates,
        "sentence_boundaries_inferred_from": "terminal punctuation and line breaks",
    }
    return sources, metadata


def limit_sources(
    sources: Sequence[SourceSentence],
    max_sentences: int,
    seed: int,
    language: str,
) -> list[SourceSentence]:
    if max_sentences <= 0 or len(sources) <= max_sentences:
        return list(sources)
    rng = random.Random(f"{seed}:{language}:source_limit")
    indexes = sorted(rng.sample(range(len(sources)), max_sentences))
    return [sources[index] for index in indexes]


def choose_sequential_group_sizes(
    sentence_count: int,
    rng: random.Random,
    min_size: int = 2,
    max_size: int = 4,
) -> list[int]:
    if sentence_count < min_size:
        return []
    sizes: list[int] = []
    remaining = sentence_count
    while remaining:
        candidates = [
            size
            for size in range(min_size, max_size + 1)
            if size <= remaining and remaining - size != 1
        ]
        if not candidates:
            raise AssertionError(f"Cannot partition {sentence_count} into 2-4 groups")
        size = rng.choice(candidates)
        sizes.append(size)
        remaining -= size
    return sizes


def build_sequence_record(
    *,
    split: str,
    language: str,
    method: str,
    sources: Sequence[SourceSentence],
    fragments: Sequence[Sequence[str]],
    spans: Sequence[tuple[int, int]],
    boundary_at_end: Sequence[bool],
) -> dict[str, object]:
    if not (len(sources) == len(fragments) == len(spans) == len(boundary_at_end)):
        raise ValueError("Source, fragment, span, and boundary lengths must match")

    tokens: list[str] = []
    labels: list[int] = []
    for fragment, is_boundary in zip(fragments, boundary_at_end):
        fragment = list(fragment)
        if not fragment:
            raise ValueError("An augmented fragment cannot be empty")
        tokens.extend(fragment)
        labels.extend([0] * len(fragment))
        if is_boundary:
            labels[-1] = 1

    return {
        "sequence_id": "",
        "split": split,
        "language": language,
        "method": method,
        "source_sentence_ids": json.dumps(
            [source.source_sentence_id for source in sources], ensure_ascii=False
        ),
        "fragment_spans": json.dumps(spans, ensure_ascii=False),
        "boundary_at_end": json.dumps(boundary_at_end),
        "text": " ".join(tokens),
        "tokens": json.dumps(tokens, ensure_ascii=False),
        "labels": json.dumps(labels),
        "num_tokens": len(tokens),
        "num_source_sentences": len(sources),
        "num_boundaries": sum(boundary_at_end),
    }


def full_sentence_record(
    sources: Sequence[SourceSentence], split: str, method: str
) -> dict[str, object]:
    return build_sequence_record(
        split=split,
        language=sources[0].language,
        method=method,
        sources=sources,
        fragments=[source.tokens for source in sources],
        spans=[(0, len(source.tokens)) for source in sources],
        boundary_at_end=[True] * len(sources),
    )


def sequential_concatenation(
    sources: Sequence[SourceSentence],
    split: str,
    seed: int,
) -> list[dict[str, object]]:
    if not sources:
        return []
    # The grouping schedule depends on split and source count, not language.
    # Equal-sized language pools therefore yield exactly balanced sequence counts.
    rng = random.Random(f"{seed}:{split}:sequential")
    sizes = choose_sequential_group_sizes(len(sources), rng)
    records: list[dict[str, object]] = []
    offset = 0
    for size in sizes:
        group = sources[offset : offset + size]
        records.append(full_sentence_record(group, split, "sequential"))
        offset += size
    if offset != len(sources):
        raise AssertionError("Sequential concatenation did not cover every source")
    return records


def random_concatenation(
    sources: Sequence[SourceSentence],
    sample_count: int,
    seed: int,
) -> list[dict[str, object]]:
    if len(sources) < 2 or sample_count <= 0:
        return []
    language = sources[0].language
    rng = random.Random(f"{seed}:{language}:train:random")
    records: list[dict[str, object]] = []
    for _ in range(sample_count):
        group_size = rng.randint(2, min(5, len(sources)))
        group = rng.sample(list(sources), group_size)
        records.append(full_sentence_record(group, "train", "random"))
    return records


def partial_boundary_merge(
    sources: Sequence[SourceSentence],
    sample_count: int,
    seed: int,
) -> list[dict[str, object]]:
    eligible = [source for source in sources if len(source.tokens) >= 2]
    if len(eligible) < 2 or sample_count <= 0:
        return []
    language = sources[0].language
    rng = random.Random(f"{seed}:{language}:train:partial")
    records: list[dict[str, object]] = []
    for _ in range(sample_count):
        first, second = rng.sample(eligible, 2)
        first_cut = rng.randint(1, len(first.tokens) - 1)
        second_cut = rng.randint(1, len(second.tokens) - 1)
        records.append(
            build_sequence_record(
                split="train",
                language=language,
                method="partial",
                sources=[first, second],
                fragments=[first.tokens[first_cut:], second.tokens[:second_cut]],
                spans=[
                    (first_cut, len(first.tokens)),
                    (0, second_cut),
                ],
                boundary_at_end=[True, False],
            )
        )
    return records


def build_train_language(
    sources: Sequence[SourceSentence],
    seed: int,
    augmentation_multiplier: float,
) -> list[dict[str, object]]:
    sequential = sequential_concatenation(sources, "train", seed)
    augmented_count = round(len(sequential) * augmentation_multiplier)
    records = list(sequential)
    records.extend(random_concatenation(sources, augmented_count, seed))
    records.extend(partial_boundary_merge(sources, augmented_count, seed))
    return records


def assign_sequence_ids(
    records: Sequence[dict[str, object]], split: str
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        copied = dict(record)
        copied["sequence_id"] = f"{split}_{index:07d}"
        result.append(copied)
    return result


def validate_dataset(
    sources: Sequence[SourceSentence],
    split_records: dict[str, Sequence[dict[str, object]]],
) -> dict[str, object]:
    source_by_id = {source.source_sentence_id: source for source in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("Duplicate source_sentence_id values")

    seen_cleaned: set[tuple[str, str]] = set()
    for source in sources:
        cleaned_key = (source.language, source.cleaned_text)
        if cleaned_key in seen_cleaned:
            raise ValueError(
                f"Duplicate cleaned source sentence: {source.source_sentence_id}"
            )
        seen_cleaned.add(cleaned_key)
        if source.language != "chg" and auxiliary_noise_reasons(
            source.cleaned_text
        ):
            raise ValueError(
                f"Auxiliary corpus noise leaked: {source.source_sentence_id}"
            )

    chagatai_ids = {
        split: {
            source.source_sentence_id
            for source in sources
            if source.language == "chg" and source.split == split
        }
        for split in ("train", "dev", "test")
    }
    if chagatai_ids["train"] & chagatai_ids["dev"]:
        raise ValueError("Chagatai train/dev source leakage")
    if chagatai_ids["train"] & chagatai_ids["test"]:
        raise ValueError("Chagatai train/test source leakage")
    if chagatai_ids["dev"] & chagatai_ids["test"]:
        raise ValueError("Chagatai dev/test source leakage")

    seen_sequence_ids: set[str] = set()
    sequential_coverage: dict[str, set[str]] = {
        "train": set(),
        "dev": set(),
        "test": set(),
    }
    for split, records in split_records.items():
        for row_number, record in enumerate(records, start=1):
            sequence_id = str(record["sequence_id"])
            if sequence_id in seen_sequence_ids:
                raise ValueError(f"Duplicate sequence id: {sequence_id}")
            seen_sequence_ids.add(sequence_id)
            if record["split"] != split:
                raise ValueError(f"{sequence_id}: split column mismatch")

            tokens = json.loads(str(record["tokens"]))
            labels = json.loads(str(record["labels"]))
            source_ids = json.loads(str(record["source_sentence_ids"]))
            spans = json.loads(str(record["fragment_spans"]))
            boundary_flags = json.loads(str(record["boundary_at_end"]))
            if not tokens or len(tokens) != len(labels):
                raise ValueError(f"{sequence_id}: token/label mismatch")
            if any(label not in (0, 1) for label in labels):
                raise ValueError(f"{sequence_id}: labels must be 0/1")
            if any(not has_word_character(token) for token in tokens):
                raise ValueError(f"{sequence_id}: punctuation-only token leaked")
            if sum(labels) != int(record["num_boundaries"]):
                raise ValueError(f"{sequence_id}: wrong boundary count")
            if sum(boundary_flags) != int(record["num_boundaries"]):
                raise ValueError(f"{sequence_id}: wrong boundary flags")
            expected_labels: list[int] = []
            for span, is_boundary in zip(spans, boundary_flags):
                fragment_length = int(span[1]) - int(span[0])
                if fragment_length <= 0:
                    raise ValueError(f"{sequence_id}: empty fragment span")
                expected_labels.extend([0] * fragment_length)
                if is_boundary:
                    expected_labels[-1] = 1
            if labels != expected_labels:
                raise ValueError(f"{sequence_id}: EOS is not at the fragment boundary")
            if any(source_id not in source_by_id for source_id in source_ids):
                raise ValueError(f"{sequence_id}: unknown source id")
            if any(source_by_id[source_id].language != record["language"] for source_id in source_ids):
                raise ValueError(f"{sequence_id}: mixed languages")
            for source_id in source_ids:
                source = source_by_id[source_id]
                if source.language == "chg" and source.split != split:
                    raise ValueError(f"{sequence_id}: Chagatai source split leakage")
                if source.language != "chg" and split != "train":
                    raise ValueError(f"{sequence_id}: auxiliary source outside train")
            if split != "train" and (
                record["language"] != "chg" or record["method"] != "sequential"
            ):
                raise ValueError(f"{sequence_id}: dev/test must be sequential Chagatai")
            if record["method"] in {"random", "partial"} and split != "train":
                raise ValueError(f"{sequence_id}: augmentation outside train")
            if record["method"] == "partial" and (
                labels[-1] != 0 or sum(labels) != 1
            ):
                raise ValueError(f"{sequence_id}: invalid partial-merge labels")
            if record["method"] == "sequential":
                sequential_coverage[split].update(source_ids)

    for split in ("dev", "test"):
        if sequential_coverage[split] != chagatai_ids[split]:
            raise ValueError(f"{split}: sequential rows do not cover all Chagatai sources")
    if not chagatai_ids["train"].issubset(sequential_coverage["train"]):
        raise ValueError("train: missing Chagatai sources in sequential rows")

    return {
        "status": "passed",
        "chagatai_source_id_intersection": {
            "train_dev": 0,
            "train_test": 0,
            "dev_test": 0,
        },
        "dev_test_are_chagatai_sequential_only": True,
        "auxiliary_languages_are_train_only": True,
        "source_cleaned_text_is_unique_per_language": True,
        "auxiliary_noise_filtered": True,
        "token_label_lengths_match": True,
        "partial_merge_final_token_is_not_eos": True,
    }


def source_rows(sources: Sequence[SourceSentence]) -> list[dict[str, object]]:
    return [
        {
            "source_sentence_id": source.source_sentence_id,
            "split": source.split,
            "language": source.language,
            "source_order": source.source_order,
            "source_dataset": source.source_dataset,
            "source_locator": source.source_locator,
            "raw_text": source.raw_text,
            "cleaned_text": source.cleaned_text,
            "num_tokens": len(source.tokens),
        }
        for source in sources
    ]


def stats_rows(
    split_records: dict[str, Sequence[dict[str, object]]]
) -> list[dict[str, object]]:
    stats: list[dict[str, object]] = []
    for split, records in split_records.items():
        grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
        for record in records:
            grouped.setdefault(
                (str(record["language"]), str(record["method"])), []
            ).append(record)
        for (language, method), group in sorted(grouped.items()):
            stats.append(
                {
                    "split": split,
                    "language": language,
                    "method": method,
                    "sequences": len(group),
                    "tokens": sum(int(row["num_tokens"]) for row in group),
                    "boundaries": sum(int(row["num_boundaries"]) for row in group),
                    "source_references": sum(
                        int(row["num_source_sentences"]) for row in group
                    ),
                }
            )
    return stats


def read_hf_revision(lutfiy_dir: Path) -> str | None:
    metadata_dir = lutfiy_dir / ".cache" / "huggingface" / "download"
    readme_metadata = metadata_dir / "README.md.metadata"
    if not readme_metadata.exists():
        return None
    lines = readme_metadata.read_text(encoding="utf-8").splitlines()
    return lines[0] if lines else None


def write_outputs(
    output_dir: Path,
    sources: Sequence[SourceSentence],
    split_records: dict[str, Sequence[dict[str, object]]],
    manifest: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows(sources)).to_csv(
        output_dir / "source_sentences.csv", index=False, encoding="utf-8"
    )
    for split in ("train", "dev", "test"):
        pd.DataFrame(split_records[split]).to_csv(
            output_dir / f"{split}.csv", index=False, encoding="utf-8"
        )
    pd.DataFrame(stats_rows(split_records)).to_csv(
        output_dir / "stats.csv", index=False, encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_variant_name(include_uzs: bool, include_uyghur: bool) -> str:
    parts = ["chagatai"]
    if include_uzs:
        parts.append("uzs")
    if include_uyghur:
        parts.append("uyghur")
    return "_".join(parts) if len(parts) > 1 else "chagatai_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the model-agnostic word-level SBD dataset."
    )
    parser.add_argument("--include-uzs", action="store_true")
    parser.add_argument("--include-uyghur", action="store_true")
    parser.add_argument(
        "--max-uzs-sentences",
        type=int,
        default=0,
        help="Deterministic cap after cleaning; 0 keeps all selected sentences.",
    )
    parser.add_argument(
        "--max-uyghur-sentences",
        type=int,
        default=0,
        help="Deterministic cap after sentence segmentation; 0 keeps all.",
    )
    parser.add_argument(
        "--uzs-sources",
        default="books",
        help="Comma-separated Lutfiy domains. Empty means every domain.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augmentation-multiplier",
        type=float,
        default=1.0,
        help="Random and partial rows per sequential row in each train language.",
    )
    parser.add_argument("--chagatai-1-35", type=Path, default=DEFAULT_CHAGATAI_1_35)
    parser.add_argument(
        "--chagatai-35-181", type=Path, default=DEFAULT_CHAGATAI_35_181
    )
    parser.add_argument("--uzs-parquet", type=Path, default=DEFAULT_UZS_PARQUET)
    parser.add_argument("--uyghur-csv", type=Path, default=DEFAULT_UYGHUR_CSV)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_uzs_sentences < 0 or args.max_uyghur_sentences < 0:
        raise ValueError("Sentence caps must be non-negative")
    if args.augmentation_multiplier < 0:
        raise ValueError("Augmentation multiplier must be non-negative")

    chagatai, chagatai_metadata = read_chagatai_sources(
        args.chagatai_1_35, args.chagatai_35_181
    )
    chagatai = split_chagatai_sources(chagatai)
    all_sources: list[SourceSentence] = list(chagatai)
    source_metadata: dict[str, object] = {"chagatai": chagatai_metadata}

    train_languages: dict[str, list[SourceSentence]] = {
        "chg": [source for source in chagatai if source.split == "train"]
    }

    if args.include_uzs:
        allowed_sources = {
            value.strip() for value in args.uzs_sources.split(",") if value.strip()
        }
        uzs, uzs_metadata = read_uzs_sources(args.uzs_parquet, allowed_sources)
        uzs = limit_sources(uzs, args.max_uzs_sentences, args.seed, "uzs")
        train_languages["uzs"] = uzs
        all_sources.extend(uzs)
        source_metadata["uzs"] = {
            **uzs_metadata,
            "sentences_after_limit": len(uzs),
        }

    if args.include_uyghur:
        uyghur, uyghur_metadata = read_uyghur_sources(args.uyghur_csv)
        uyghur = limit_sources(
            uyghur, args.max_uyghur_sentences, args.seed, "uig"
        )
        train_languages["uig"] = uyghur
        all_sources.extend(uyghur)
        source_metadata["uyghur"] = {
            **uyghur_metadata,
            "sentences_after_limit": len(uyghur),
        }

    train_records: list[dict[str, object]] = []
    for language in ("chg", "uzs", "uig"):
        if language in train_languages:
            train_records.extend(
                build_train_language(
                    train_languages[language],
                    args.seed,
                    args.augmentation_multiplier,
                )
            )

    dev_sources = [
        source for source in chagatai if source.language == "chg" and source.split == "dev"
    ]
    test_sources = [
        source for source in chagatai if source.language == "chg" and source.split == "test"
    ]
    split_records = {
        "train": assign_sequence_ids(train_records, "train"),
        "dev": assign_sequence_ids(
            sequential_concatenation(dev_sources, "dev", args.seed), "dev"
        ),
        "test": assign_sequence_ids(
            sequential_concatenation(test_sources, "test", args.seed), "test"
        ),
    }
    checks = validate_dataset(all_sources, split_records)

    default_variant_name = build_variant_name(args.include_uzs, args.include_uyghur)
    output_dir = args.output_dir or BUILDS_DIR / default_variant_name
    variant_name = output_dir.name
    input_paths = [args.chagatai_1_35, args.chagatai_35_181]
    if args.include_uzs:
        input_paths.append(args.uzs_parquet)
    if args.include_uyghur:
        input_paths.append(args.uyghur_csv)

    source_counts = Counter(
        (source.split, source.language) for source in all_sources
    )
    manifest = {
        "schema_version": "1.1",
        "variant": variant_name,
        "seed": args.seed,
        "labels": {"0": "O/not sentence end", "1": "EOS/sentence end"},
        "include_uzs": args.include_uzs,
        "include_uyghur": args.include_uyghur,
        "uzs_sources": sorted(
            value.strip() for value in args.uzs_sources.split(",") if value.strip()
        ),
        "max_uzs_sentences": args.max_uzs_sentences,
        "max_uyghur_sentences": args.max_uyghur_sentences,
        "augmentation_multiplier": args.augmentation_multiplier,
        "chagatai_split": {"train": 0.70, "dev": 0.10, "test": 0.20},
        "source_counts": {
            f"{split}:{language}": count
            for (split, language), count in sorted(source_counts.items())
        },
        "sequence_counts": {
            split: len(records) for split, records in split_records.items()
        },
        "input_files": {
            str(path.resolve()): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in input_paths
        },
        "huggingface": {
            "dataset": "tahrirchi/lutfiy" if args.include_uzs else None,
            "revision": (
                read_hf_revision(args.uzs_parquet.parents[1])
                if args.include_uzs
                else None
            ),
        },
        "source_processing": source_metadata,
        "checks": checks,
    }
    write_outputs(output_dir, all_sources, split_records, manifest)

    print(f"DONE: {variant_name}")
    print(f"Output: {output_dir}")
    print(f"Source sentences: {len(all_sources):,}")
    print(
        "Sequences: "
        + ", ".join(
            f"{split}={len(records):,}" for split, records in split_records.items()
        )
    )
    print("Checks: passed")


if __name__ == "__main__":
    main()
