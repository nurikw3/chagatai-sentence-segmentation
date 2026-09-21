from __future__ import annotations

import json
import random

import pytest

from unified_dataset.adapters.stanza import stanza_text_and_labels
from unified_dataset.augmentation import (
    choose_sequential_group_sizes,
    partial_boundary_merge,
    sequential_concatenation,
)
from unified_dataset.cleaning import clean_sentence, source_noise_reasons
from unified_dataset.labeling import assign_sequence_ids
from unified_dataset.sources import (
    SourceSentence,
    deduplicate_cleaned_sources,
    split_chagatai_sources,
)
from unified_dataset.validation import validate_dataset


def source(source_id: str, text: str, *, split: str = "train") -> SourceSentence:
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
        split=split,
    )


def test_clean_sentence_removes_punctuation_diacritics_and_footnotes() -> None:
    assert clean_sentence("سَلام، دنیا! (١)") == "سلام دنیا"


def test_noise_filter_rejects_residual_scripts_and_urls() -> None:
    assert source_noise_reasons("ئۇيغۇرچە تېكىست") == ()
    assert "latin_script" in source_noise_reasons("زامانىمىز Martin Heidegger")
    assert "cyrillic_script" in source_noise_reasons("طلبی متن Олтинчи")
    assert "url_or_bibliography_remnant" in source_noise_reasons(
        "ئۇيغۇرچە تور ئادرېسى http example"
    )


def test_sequential_sizes_cover_all_sentences_without_singletons() -> None:
    for count in range(2, 100):
        sizes = choose_sequential_group_sizes(count, random.Random(count))
        assert sum(sizes) == count
        assert all(2 <= size <= 4 for size in sizes)


def test_sequential_concatenation_covers_every_source_once() -> None:
    sources = [source(f"s{i}", f"الف ب {i}") for i in range(1, 70)]
    records = sequential_concatenation(sources, "train", seed=42)
    ids = [
        source_id
        for record in records
        for source_id in json.loads(record["source_sentence_ids"])
    ]
    assert ids == [item.source_sentence_id for item in sources]


def test_partial_merge_has_only_internal_eos() -> None:
    sources = [source("s1", "الف ب ت ث"), source("s2", "ج ح خ د")]
    record = partial_boundary_merge(sources, sample_count=1, seed=42)[0]
    labels = json.loads(record["labels"])
    assert sum(labels) == 1
    assert labels[-1] == 0
    assert record["boundary_at_end"] == "[true, false]"


def test_exact_cleaned_duplicates_keep_first_source() -> None:
    first = source("s1", "الف ب")
    duplicate = source("s2", "الف ب")
    distinct = source("s3", "ج د")
    kept, dropped = deduplicate_cleaned_sources([first, duplicate, distinct])
    assert kept == [first, distinct]
    assert dropped == 1


def test_split_is_70_10_20_and_preserves_order() -> None:
    sources = [source(f"s{i}", f"الف {i}") for i in range(1, 101)]
    split = split_chagatai_sources(sources)
    assert [item.split for item in split].count("train") == 70
    assert [item.split for item in split].count("dev") == 10
    assert [item.split for item in split].count("test") == 20
    assert [item.source_sentence_id for item in split] == [
        item.source_sentence_id for item in sources
    ]


def test_stanza_projection_uses_character_labels_0_1_2() -> None:
    text, labels = stanza_text_and_labels(["اب", "جد"], [0, 1])
    assert text == "اب جد"
    assert labels == "01002"


def test_validation_reconstructs_tokens_from_source_spans() -> None:
    train_sources = [source(f"s{i}", f"الف ب {i}") for i in range(1, 5)]
    records = assign_sequence_ids(
        sequential_concatenation(train_sources, "train", seed=42), "train"
    )
    checks = validate_dataset(
        train_sources,
        {"train": records, "dev": [], "test": []},
    )
    assert checks["tokens_reconstruct_from_source_spans"] is True

    broken = {key: list(value) for key, value in {"train": records, "dev": [], "test": []}.items()}
    broken["train"] = [dict(record) for record in broken["train"]]
    broken["train"][0]["text"] = "битый текст"
    with pytest.raises(ValueError, match="text does not match tokens"):
        validate_dataset(train_sources, broken)
