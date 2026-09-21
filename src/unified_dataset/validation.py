from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence

from .cleaning import has_word_character, source_noise_reasons
from .sources import SourceSentence


def validate_dataset(
    sources: Sequence[SourceSentence],
    split_records: dict[str, Sequence[dict[str, object]]],
) -> dict[str, object]:
    source_by_id = {source.source_sentence_id: source for source in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("Duplicate source_sentence_id values")

    seen_cleaned: set[tuple[str, str]] = set()
    for source in sources:
        key = (source.language, source.cleaned_text)
        if key in seen_cleaned:
            raise ValueError(f"Duplicate cleaned source sentence: {source.source_sentence_id}")
        seen_cleaned.add(key)
        reasons = source_noise_reasons(source.cleaned_text)
        if reasons:
            raise ValueError(
                f"Source noise leaked: {source.source_sentence_id}: {', '.join(reasons)}"
            )
        if tuple(source.cleaned_text.split()) != source.tokens:
            raise ValueError(f"{source.source_sentence_id}: cleaned text/token mismatch")

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
    sequential_reference_counts: dict[tuple[str, str], Counter[str]] = {}
    sequence_texts: dict[str, set[str]] = {split: set() for split in split_records}

    for split, records in split_records.items():
        for record in records:
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
            if str(record["text"]) != " ".join(tokens):
                raise ValueError(f"{sequence_id}: text does not match tokens")
            if int(record["num_tokens"]) != len(tokens):
                raise ValueError(f"{sequence_id}: wrong num_tokens")
            if any(label not in (0, 1) for label in labels):
                raise ValueError(f"{sequence_id}: labels must be 0/1")
            if any(not has_word_character(token) for token in tokens):
                raise ValueError(f"{sequence_id}: punctuation-only token leaked")
            if not (len(source_ids) == len(spans) == len(boundary_flags)):
                raise ValueError(f"{sequence_id}: provenance lengths do not match")
            if int(record["num_source_sentences"]) != len(source_ids):
                raise ValueError(f"{sequence_id}: wrong num_source_sentences")
            if any(source_id not in source_by_id for source_id in source_ids):
                raise ValueError(f"{sequence_id}: unknown source id")

            expected_tokens: list[str] = []
            expected_labels: list[int] = []
            referenced_sources: list[SourceSentence] = []
            for source_id, span, is_boundary in zip(source_ids, spans, boundary_flags):
                source = source_by_id[source_id]
                referenced_sources.append(source)
                start, end = int(span[0]), int(span[1])
                if start < 0 or end <= start or end > len(source.tokens):
                    raise ValueError(f"{sequence_id}: invalid fragment span for {source_id}")
                fragment = list(source.tokens[start:end])
                expected_tokens.extend(fragment)
                expected_labels.extend([0] * len(fragment))
                if is_boundary:
                    expected_labels[-1] = 1

            if tokens != expected_tokens:
                raise ValueError(f"{sequence_id}: tokens do not match source spans")
            if labels != expected_labels:
                raise ValueError(f"{sequence_id}: labels do not match fragment boundaries")
            if sum(labels) != int(record["num_boundaries"]):
                raise ValueError(f"{sequence_id}: wrong boundary count")
            if sum(boundary_flags) != int(record["num_boundaries"]):
                raise ValueError(f"{sequence_id}: wrong boundary flags")
            if any(source.language != record["language"] for source in referenced_sources):
                raise ValueError(f"{sequence_id}: mixed languages")

            for source in referenced_sources:
                if source.language == "chg" and source.split != split:
                    raise ValueError(f"{sequence_id}: Chagatai source split leakage")
                if source.language != "chg" and split != "train":
                    raise ValueError(f"{sequence_id}: auxiliary source outside train")

            method = str(record["method"])
            if method not in {"sequential", "random", "partial"}:
                raise ValueError(f"{sequence_id}: unknown augmentation method {method}")
            if split != "train" and (
                record["language"] != "chg" or method != "sequential"
            ):
                raise ValueError(f"{sequence_id}: dev/test must be sequential Chagatai")
            if method in {"random", "partial"} and split != "train":
                raise ValueError(f"{sequence_id}: augmentation outside train")
            if method == "partial" and (
                len(source_ids) != 2
                or boundary_flags != [True, False]
                or labels[-1] != 0
                or sum(labels) != 1
            ):
                raise ValueError(f"{sequence_id}: invalid partial-merge labels")
            if method == "sequential":
                if not all(boundary_flags):
                    raise ValueError(f"{sequence_id}: sequential fragment without EOS")
                if any(
                    list(span) != [0, len(source.tokens)]
                    for source, span in zip(referenced_sources, spans)
                ):
                    raise ValueError(f"{sequence_id}: sequential record uses partial span")
                orders = [source.source_order for source in referenced_sources]
                if orders != sorted(orders) or len(orders) != len(set(orders)):
                    raise ValueError(f"{sequence_id}: sequential source order changed")
                key = (split, str(record["language"]))
                sequential_reference_counts.setdefault(key, Counter()).update(source_ids)

            sequence_text = str(record["text"])
            if sequence_text in sequence_texts[split]:
                raise ValueError(f"{sequence_id}: duplicate sequence text within {split}")
            sequence_texts[split].add(sequence_text)

    for first, second in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = sequence_texts[first] & sequence_texts[second]
        if overlap:
            raise ValueError(f"Exact sequence text leakage between {first} and {second}")

    source_groups: dict[tuple[str, str], set[str]] = {}
    for source in sources:
        source_groups.setdefault((source.split, source.language), set()).add(
            source.source_sentence_id
        )
    for key, expected_ids in source_groups.items():
        counts = sequential_reference_counts.get(key, Counter())
        if set(counts) != expected_ids:
            raise ValueError(f"{key}: sequential rows do not cover every source")
        repeated = [source_id for source_id, count in counts.items() if count != 1]
        if repeated:
            raise ValueError(f"{key}: sources repeated in sequential rows: {repeated[:5]}")

    return {
        "status": "passed",
        "chagatai_source_id_intersection": {
            "train_dev": 0,
            "train_test": 0,
            "dev_test": 0,
        },
        "exact_sequence_text_intersection": {
            "train_dev": 0,
            "train_test": 0,
            "dev_test": 0,
        },
        "dev_test_are_chagatai_sequential_only": True,
        "auxiliary_languages_are_train_only": True,
        "source_cleaned_text_is_unique_per_language": True,
        "source_noise_filtered": True,
        "token_label_lengths_match": True,
        "tokens_reconstruct_from_source_spans": True,
        "sequential_sources_covered_exactly_once": True,
        "partial_merge_final_token_is_not_eos": True,
    }
