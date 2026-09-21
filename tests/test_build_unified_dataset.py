from __future__ import annotations

import json
import random

from build_unified_dataset import (
    SourceSentence,
    choose_sequential_group_sizes,
    clean_sentence,
    partial_boundary_merge,
)


def source(source_id: str, text: str) -> SourceSentence:
    cleaned = clean_sentence(text)
    return SourceSentence(
        source_sentence_id=source_id,
        language="chg",
        source_order=int(source_id.removeprefix("s")),
        source_dataset="fixture",
        source_locator=source_id,
        raw_text=text,
        cleaned_text=cleaned,
        tokens=tuple(cleaned.split()),
    )


def test_clean_sentence_removes_punctuation_diacritics_and_footnotes() -> None:
    assert clean_sentence("سَلام، دنیا! (١)") == "سلام دنیا"


def test_sequential_sizes_cover_all_sentences_without_singletons() -> None:
    for count in range(2, 100):
        sizes = choose_sequential_group_sizes(count, random.Random(count))
        assert sum(sizes) == count
        assert all(2 <= size <= 4 for size in sizes)


def test_partial_merge_has_only_internal_eos() -> None:
    sources = [
        source("s1", "الف ب ت ث"),
        source("s2", "ج ح خ د"),
    ]
    record = partial_boundary_merge(sources, sample_count=1, seed=42)[0]
    labels = json.loads(record["labels"])

    assert sum(labels) == 1
    assert labels[-1] == 0
    assert record["boundary_at_end"] == "[true, false]"
