from __future__ import annotations

import csv
import hashlib
import math
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from .cleaning import clean_sentence, has_word_character, normalize_spaces, source_noise_reasons


TERMINAL_PUNCTUATION_RE = re.compile(r"[.!?\u061f\u06d4\u2026]+")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")


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


def read_chagatai_sources(
    early_path: Path,
    late_path: Path,
) -> tuple[list[SourceSentence], dict[str, object]]:
    early_df = pd.read_excel(early_path)
    required_early = {"Original", "Pages"}
    if not required_early.issubset(early_df.columns):
        raise ValueError(f"{early_path} must contain columns {sorted(required_early)}")
    early_df = early_df.copy()
    early_df["_page"] = early_df["Pages"].ffill()

    late_df = pd.read_excel(late_path)
    required_late = {"Num", "Original", "sentence_id"}
    if not required_late.issubset(late_df.columns):
        raise ValueError(f"{late_path} must contain columns {sorted(required_late)}")

    early_rows = early_df[
        early_df["Original"].notna() & (early_df["_page"].fillna(0) < 35)
    ]

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
    dropped_noise = 0
    noise_reason_counts: Counter[str] = Counter()

    def append_source(source: SourceSentence | None) -> None:
        nonlocal dropped_empty, dropped_noise
        if source is None:
            dropped_empty += 1
            return
        reasons = source_noise_reasons(source.cleaned_text)
        if reasons:
            dropped_noise += 1
            noise_reason_counts.update(reasons)
            return
        sources.append(source)

    for row_index, row in early_rows.iterrows():
        page = int(row["_page"]) if pd.notna(row["_page"]) else "unknown"
        append_source(
            make_source_sentence(
                source_sentence_id=f"chg_{len(sources) + 1:06d}",
                language="chg",
                source_order=len(sources),
                source_dataset="chagatai_pages_1_35",
                source_locator=f"excel_row={row_index + 2};page={page}",
                raw_text=str(row["Original"]),
            )
        )

    for row_index, row in late_rows.iterrows():
        append_source(
            make_source_sentence(
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
        )

    sources, dropped_duplicates = deduplicate_cleaned_sources(sources)
    sources = [replace(source, source_order=index) for index, source in enumerate(sources)]
    metadata = {
        "early_rows_included": len(early_rows),
        "late_rows_included_before_cleaning": len(late_rows),
        "duplicate_late_pages_dropped": duplicate_pages,
        "duplicate_cleaned_sentences_dropped": dropped_duplicates,
        "empty_rows_dropped": dropped_empty,
        "noise_rows_dropped": dropped_noise,
        "noise_reason_counts": dict(sorted(noise_reason_counts.items())),
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
        split = "train" if index < train_end else "dev" if index < dev_end else "test"
        result.append(replace(source, split=split))
    return result


def read_uzs_sources(
    parquet_path: Path,
    allowed_sources: set[str],
) -> tuple[list[SourceSentence], dict[str, object]]:
    try:
        df = pd.read_parquet(parquet_path)
    except ImportError as error:
        raise RuntimeError("Reading Lutfiy requires pyarrow. Run `uv sync`.") from error

    required = {"tgt_sent", "tgt_lang", "source"}
    if not required.issubset(df.columns):
        raise ValueError(f"{parquet_path} must contain columns {sorted(required)}")
    bad_languages = sorted(set(df["tgt_lang"].dropna()) - {"uzs_Arab"})
    if bad_languages:
        raise ValueError(f"Unexpected Lutfiy target languages: {bad_languages}")

    selected_df = df[df["source"].isin(allowed_sources)] if allowed_sources else df
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
        reasons = source_noise_reasons(source.cleaned_text)
        if reasons:
            dropped_noise += 1
            noise_reason_counts.update(reasons)
            continue
        if source.cleaned_text in seen_cleaned:
            dropped_duplicates += 1
            continue
        seen_cleaned.add(source.cleaned_text)
        sources.append(source)

    sources = [replace(source, source_order=index) for index, source in enumerate(sources)]
    return sources, {
        "rows_in_repository": len(df),
        "allowed_sources": sorted(allowed_sources),
        "rows_after_source_filter": len(selected_df),
        "empty_rows_dropped": dropped_empty,
        "noise_rows_dropped": dropped_noise,
        "noise_reason_counts": dict(sorted(noise_reason_counts.items())),
        "duplicate_cleaned_sentences_dropped": dropped_duplicates,
    }


def strip_markdown_line(line: str) -> str | None:
    line = line.strip()
    if not line or re.match(r"^(#{1,6}\s|---+$|\*{3,}$)", line):
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


def read_uyghur_sources(csv_path: Path) -> tuple[list[SourceSentence], dict[str, object]]:
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
            segments = split_uyghur_article(row["text"])
            for segment_index, raw_sentence in enumerate(segments, start=1):
                source = make_source_sentence(
                    source_sentence_id=(
                        f"uig_article_{article_index:05d}_"
                        f"segment_{segment_index:04d}"
                    ),
                    language="uig",
                    source_order=len(sources),
                    source_dataset="uyghur_corpus",
                    source_locator=f"csv_row={article_index + 1};segment={segment_index}",
                    raw_text=raw_sentence,
                )
                if source is None:
                    dropped_empty += 1
                    continue
                reasons = source_noise_reasons(source.cleaned_text)
                if reasons:
                    dropped_noise += 1
                    noise_reason_counts.update(reasons)
                    continue
                if source.cleaned_text in seen_cleaned:
                    dropped_duplicates += 1
                    continue
                seen_cleaned.add(source.cleaned_text)
                sources.append(source)

    sources = [replace(source, source_order=index) for index, source in enumerate(sources)]
    return sources, {
        "articles": article_count,
        "empty_segments_dropped": dropped_empty,
        "noise_rows_dropped": dropped_noise,
        "noise_reason_counts": dict(sorted(noise_reason_counts.items())),
        "duplicate_cleaned_sentences_dropped": dropped_duplicates,
        "sentence_boundaries_inferred_from": "terminal punctuation and line breaks",
    }


def limit_sources(
    sources: Sequence[SourceSentence],
    max_sentences: int,
    seed: int,
    language: str,
) -> list[SourceSentence]:
    if max_sentences <= 0 or len(sources) <= max_sentences:
        selected = list(sources)
    else:
        rng = random.Random(f"{seed}:{language}:source_limit")
        indexes = sorted(rng.sample(range(len(sources)), max_sentences))
        selected = [sources[index] for index in indexes]
    return [replace(source, source_order=index) for index, source in enumerate(selected)]
