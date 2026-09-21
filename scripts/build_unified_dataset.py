from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_dataset.adapters.stanza import export_stanza_dataset  # noqa: E402
from unified_dataset.augmentation import (  # noqa: E402
    build_train_language,
    sequential_concatenation,
)
from unified_dataset.cleaning import (  # noqa: E402
    FORBIDDEN_AUXILIARY_SCRIPTS,
    MIN_ARABIC_LETTER_RATIO,
)
from unified_dataset.labeling import (  # noqa: E402
    assign_sequence_ids,
    source_rows,
    stats_rows,
)
from unified_dataset.sources import (  # noqa: E402
    SourceSentence,
    limit_sources,
    read_chagatai_sources,
    read_uyghur_sources,
    read_uzs_sources,
    sha256_file,
    split_chagatai_sources,
)
from unified_dataset.validation import validate_dataset  # noqa: E402


RAW_DIR = PROJECT_ROOT / "data" / "UNIFIED" / "raw"
BUILDS_DIR = PROJECT_ROOT / "data" / "UNIFIED" / "builds"
DEFAULT_CHAGATAI_1_35 = RAW_DIR / "chagatai_pages_1_35.xlsx"
DEFAULT_CHAGATAI_35_181 = RAW_DIR / "chagatai_pages_35_181.xlsx"
DEFAULT_UZS_PARQUET = RAW_DIR / "lutfiy_hf" / "data" / "train-00000-of-00001.parquet"
DEFAULT_UYGHUR_CSV = RAW_DIR / "uyghur_corpus.csv"


def read_hf_revision(lutfiy_dir: Path) -> str | None:
    metadata = lutfiy_dir / ".cache" / "huggingface" / "download" / "README.md.metadata"
    if not metadata.exists():
        return None
    lines = metadata.read_text(encoding="utf-8").splitlines()
    return lines[0] if lines else None


def write_outputs(
    output_dir: Path,
    sources: list[SourceSentence],
    split_records: dict[str, list[dict[str, object]]],
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
        description="Build the canonical word-level SBD dataset."
    )
    parser.add_argument("--include-uzs", action="store_true")
    parser.add_argument("--include-uyghur", action="store_true")
    parser.add_argument("--max-uzs-sentences", type=int, default=0)
    parser.add_argument("--max-uyghur-sentences", type=int, default=0)
    parser.add_argument(
        "--uzs-sources",
        default="books",
        help="Comma-separated Lutfiy domains; empty means every domain.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augmentation-multiplier",
        type=float,
        default=1.0,
        help="Random and partial rows per sequential row in each train language.",
    )
    parser.add_argument("--chagatai-1-35", type=Path, default=DEFAULT_CHAGATAI_1_35)
    parser.add_argument("--chagatai-35-181", type=Path, default=DEFAULT_CHAGATAI_35_181)
    parser.add_argument("--uzs-parquet", type=Path, default=DEFAULT_UZS_PARQUET)
    parser.add_argument("--uyghur-csv", type=Path, default=DEFAULT_UYGHUR_CSV)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--export-stanza",
        action="store_true",
        help="Also write Stanza character-level files under OUTPUT_DIR/stanza.",
    )
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
        uzs, metadata = read_uzs_sources(args.uzs_parquet, allowed_sources)
        uzs = limit_sources(uzs, args.max_uzs_sentences, args.seed, "uzs")
        train_languages["uzs"] = uzs
        all_sources.extend(uzs)
        source_metadata["uzs"] = {**metadata, "sentences_after_limit": len(uzs)}

    if args.include_uyghur:
        uyghur, metadata = read_uyghur_sources(args.uyghur_csv)
        uyghur = limit_sources(uyghur, args.max_uyghur_sentences, args.seed, "uig")
        train_languages["uig"] = uyghur
        all_sources.extend(uyghur)
        source_metadata["uyghur"] = {
            **metadata,
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

    dev_sources = [source for source in chagatai if source.split == "dev"]
    test_sources = [source for source in chagatai if source.split == "test"]
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

    output_dir = args.output_dir or BUILDS_DIR / build_variant_name(
        args.include_uzs, args.include_uyghur
    )
    input_paths = [args.chagatai_1_35, args.chagatai_35_181]
    if args.include_uzs:
        input_paths.append(args.uzs_parquet)
    if args.include_uyghur:
        input_paths.append(args.uyghur_csv)

    source_counts = Counter((source.split, source.language) for source in all_sources)
    manifest = {
        "schema_version": "2.0",
        "variant": output_dir.name,
        "seed": args.seed,
        "labels": {"0": "O/not sentence end", "1": "EOS/sentence end"},
        "cleaning": {
            "unicode_normalization": "NFKC",
            "min_arabic_letter_ratio": MIN_ARABIC_LETTER_RATIO,
            "forbidden_auxiliary_scripts": list(FORBIDDEN_AUXILIARY_SCRIPTS),
            "punctuation_symbols_marks_removed": True,
        },
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
            str(path.resolve()): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
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

    stanza_counts = None
    if args.export_stanza:
        stanza_counts = export_stanza_dataset(output_dir, output_dir / "stanza")

    print(f"DONE: {output_dir.name}")
    print(f"Output: {output_dir}")
    print(f"Source sentences: {len(all_sources):,}")
    print(
        "Sequences: "
        + ", ".join(
            f"{split}={len(records):,}" for split, records in split_records.items()
        )
    )
    if stanza_counts is not None:
        print(f"Stanza export: {stanza_counts}")
    print("Checks: passed")


if __name__ == "__main__":
    main()
