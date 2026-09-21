from __future__ import annotations

import json
from collections.abc import Sequence

from .sources import SourceSentence


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
    """Create the canonical word-level O/EOS record for one sequence."""
    if not sources:
        raise ValueError("A sequence must reference at least one source sentence")
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
    sources: Sequence[SourceSentence],
    split: str,
    method: str,
) -> dict[str, object]:
    if not sources:
        raise ValueError("A full-sentence record cannot be empty")
    return build_sequence_record(
        split=split,
        language=sources[0].language,
        method=method,
        sources=sources,
        fragments=[source.tokens for source in sources],
        spans=[(0, len(source.tokens)) for source in sources],
        boundary_at_end=[True] * len(sources),
    )


def assign_sequence_ids(
    records: Sequence[dict[str, object]],
    split: str,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        copied = dict(record)
        copied["sequence_id"] = f"{split}_{index:07d}"
        result.append(copied)
    return result


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
    split_records: dict[str, Sequence[dict[str, object]]],
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
