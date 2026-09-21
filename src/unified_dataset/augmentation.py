from __future__ import annotations

import random
from collections.abc import Sequence

from .labeling import build_sequence_record, full_sentence_record
from .sources import SourceSentence


def choose_sequential_group_sizes(
    sentence_count: int,
    rng: random.Random,
    min_size: int = 2,
    max_size: int = 4,
) -> list[int]:
    """Partition all sources without silently dropping a final singleton."""
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


def sequential_concatenation(
    sources: Sequence[SourceSentence],
    split: str,
    seed: int,
) -> list[dict[str, object]]:
    if not sources:
        return []
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
    forbidden_texts: set[str] | None = None,
) -> list[dict[str, object]]:
    if len(sources) < 2 or sample_count <= 0:
        return []
    language = sources[0].language
    rng = random.Random(f"{seed}:{language}:train:random")
    records: list[dict[str, object]] = []
    seen_texts = set(forbidden_texts or ())
    attempts = 0
    max_attempts = max(sample_count * 50, sample_count + 1_000)
    while len(records) < sample_count and attempts < max_attempts:
        attempts += 1
        group_size = rng.randint(2, min(5, len(sources)))
        group = rng.sample(list(sources), group_size)
        orders = [source.source_order for source in group]
        if orders == sorted(orders):
            continue
        if any(
            abs(first - second) == 1
            for index, first in enumerate(orders)
            for second in orders[index + 1 :]
        ):
            continue
        record = full_sentence_record(group, "train", "random")
        text = str(record["text"])
        if text in seen_texts:
            continue
        seen_texts.add(text)
        records.append(record)
    if len(records) != sample_count:
        raise ValueError(
            f"Could not generate {sample_count} unique random sequences for {language}"
        )
    return records


def partial_boundary_merge(
    sources: Sequence[SourceSentence],
    sample_count: int,
    seed: int,
    forbidden_texts: set[str] | None = None,
) -> list[dict[str, object]]:
    eligible = [source for source in sources if len(source.tokens) >= 2]
    if len(eligible) < 2 or sample_count <= 0:
        return []
    language = sources[0].language
    rng = random.Random(f"{seed}:{language}:train:partial")
    records: list[dict[str, object]] = []
    seen_texts = set(forbidden_texts or ())
    attempts = 0
    max_attempts = max(sample_count * 50, sample_count + 1_000)
    while len(records) < sample_count and attempts < max_attempts:
        attempts += 1
        first, second = rng.sample(eligible, 2)
        first_cut = rng.randint(1, len(first.tokens) - 1)
        second_cut = rng.randint(1, len(second.tokens) - 1)
        record = build_sequence_record(
            split="train",
            language=language,
            method="partial",
            sources=[first, second],
            fragments=[first.tokens[first_cut:], second.tokens[:second_cut]],
            spans=[(first_cut, len(first.tokens)), (0, second_cut)],
            boundary_at_end=[True, False],
        )
        text = str(record["text"])
        if text in seen_texts:
            continue
        seen_texts.add(text)
        records.append(record)
    if len(records) != sample_count:
        raise ValueError(
            f"Could not generate {sample_count} unique partial sequences for {language}"
        )
    return records


def build_train_language(
    sources: Sequence[SourceSentence],
    seed: int,
    augmentation_multiplier: float,
    forbidden_texts: set[str] | None = None,
) -> list[dict[str, object]]:
    sequential = sequential_concatenation(sources, "train", seed)
    seen_texts = set(forbidden_texts or ())
    sequential_texts = [str(record["text"]) for record in sequential]
    if len(sequential_texts) != len(set(sequential_texts)):
        raise ValueError(f"Duplicate sequential sequences for {sources[0].language}")
    overlap = seen_texts.intersection(sequential_texts)
    if overlap:
        raise ValueError(
            f"Sequential sequence overlaps an earlier language for {sources[0].language}"
        )
    seen_texts.update(sequential_texts)

    augmented_count = round(len(sequential) * augmentation_multiplier)
    records = list(sequential)
    random_records = random_concatenation(
        sources,
        augmented_count,
        seed,
        forbidden_texts=seen_texts,
    )
    records.extend(random_records)
    seen_texts.update(str(record["text"]) for record in random_records)
    records.extend(
        partial_boundary_merge(
            sources,
            augmented_count,
            seed,
            forbidden_texts=seen_texts,
        )
    )
    return records
