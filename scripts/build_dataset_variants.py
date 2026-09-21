from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_dataset.sources import (  # noqa: E402
    read_chagatai_sources,
    split_chagatai_sources,
)


RAW_DIR = PROJECT_ROOT / "data" / "UNIFIED" / "raw"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "UNIFIED" / "builds"
DEFAULT_CHAGATAI_1_35 = RAW_DIR / "chagatai_pages_1_35.xlsx"
DEFAULT_CHAGATAI_35_181 = RAW_DIR / "chagatai_pages_35_181.xlsx"
DEFAULT_UZS_PARQUET = RAW_DIR / "lutfiy_hf" / "data" / "train-00000-of-00001.parquet"
DEFAULT_UYGHUR_CSV = RAW_DIR / "uyghur_corpus.csv"
BUILDER = PROJECT_ROOT / "scripts" / "build_unified_dataset.py"


@dataclass(frozen=True)
class Variant:
    name: str
    include_uzs: bool = False
    include_uyghur: bool = False
    balance_auxiliary: bool = False


VARIANTS = (
    Variant("chagatai_only"),
    Variant("chagatai_uyghur", include_uyghur=True, balance_auxiliary=True),
    Variant("chagatai_uzs", include_uzs=True, balance_auxiliary=True),
    Variant(
        "chagatai_uzs_uyghur_full",
        include_uzs=True,
        include_uyghur=True,
    ),
    Variant(
        "chagatai_uzs_uyghur_balanced",
        include_uzs=True,
        include_uyghur=True,
        balance_auxiliary=True,
    ),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chagatai_train_count(early_path: Path, late_path: Path) -> int:
    sources, _ = read_chagatai_sources(early_path, late_path)
    split_sources = split_chagatai_sources(sources)
    return sum(source.split == "train" for source in split_sources)


def build_command(
    variant: Variant,
    *,
    output_dir: Path,
    chagatai_train_sentences: int,
    seed: int,
    augmentation_multiplier: float,
    chagatai_1_35: Path,
    chagatai_35_181: Path,
    uzs_parquet: Path,
    uyghur_csv: Path,
    uzs_sources: str,
    export_stanza: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(BUILDER),
        "--seed",
        str(seed),
        "--augmentation-multiplier",
        str(augmentation_multiplier),
        "--chagatai-1-35",
        str(chagatai_1_35),
        "--chagatai-35-181",
        str(chagatai_35_181),
        "--uzs-parquet",
        str(uzs_parquet),
        "--uyghur-csv",
        str(uyghur_csv),
        "--uzs-sources",
        uzs_sources,
        "--output-dir",
        str(output_dir),
    ]
    if variant.include_uzs:
        command.append("--include-uzs")
        if variant.balance_auxiliary:
            command.extend(
                ["--max-uzs-sentences", str(chagatai_train_sentences)]
            )
    if variant.include_uyghur:
        command.append("--include-uyghur")
        if variant.balance_auxiliary:
            command.extend(
                ["--max-uyghur-sentences", str(chagatai_train_sentences)]
            )
    if export_stanza:
        command.append("--export-stanza")
    return command


def read_manifest(variant_dir: Path) -> dict[str, object]:
    path = variant_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing build manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_equal_frames(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    description: str,
) -> None:
    if list(expected.columns) != list(actual.columns) or not expected.equals(actual):
        raise ValueError(f"Variant audit failed: {description} differs")


def audit_variants(
    output_root: Path,
    variants: list[Variant],
    chagatai_train_sentences: int,
) -> dict[str, object]:
    required_files = {
        "source_sentences.csv",
        "train.csv",
        "dev.csv",
        "test.csv",
        "stats.csv",
        "manifest.json",
    }
    baseline_sources: pd.DataFrame | None = None
    baseline_train: pd.DataFrame | None = None
    baseline_dev_hash: str | None = None
    baseline_test_hash: str | None = None
    summaries: dict[str, object] = {}

    for variant in variants:
        variant_dir = output_root / variant.name
        missing = sorted(
            name for name in required_files if not (variant_dir / name).exists()
        )
        if missing:
            raise ValueError(f"{variant.name}: missing files: {missing}")

        manifest = read_manifest(variant_dir)
        checks = manifest.get("checks", {})
        if not isinstance(checks, dict) or checks.get("status") != "passed":
            raise ValueError(f"{variant.name}: internal validation did not pass")

        source_counts = manifest.get("source_counts", {})
        if not isinstance(source_counts, dict):
            raise ValueError(f"{variant.name}: invalid source_counts")
        if source_counts.get("train:chg") != chagatai_train_sentences:
            raise ValueError(f"{variant.name}: Chagatai train count changed")

        expected_languages = {"chg"}
        if variant.include_uzs:
            expected_languages.add("uzs")
        if variant.include_uyghur:
            expected_languages.add("uig")
        actual_languages = {
            key.split(":", 1)[1]
            for key in source_counts
            if key.startswith("train:")
        }
        if actual_languages != expected_languages:
            raise ValueError(
                f"{variant.name}: expected train languages {sorted(expected_languages)}, "
                f"got {sorted(actual_languages)}"
            )

        if variant.balance_auxiliary:
            for language in expected_languages - {"chg"}:
                if source_counts.get(f"train:{language}") != chagatai_train_sentences:
                    raise ValueError(
                        f"{variant.name}: {language} is not sentence-balanced to Chagatai"
                    )

        sources = pd.read_csv(variant_dir / "source_sentences.csv")
        chagatai_sources = sources[sources["language"] == "chg"].reset_index(drop=True)
        train = pd.read_csv(variant_dir / "train.csv")
        chagatai_train = train[train["language"] == "chg"].reset_index(drop=True)
        if baseline_sources is None:
            baseline_sources = chagatai_sources
            baseline_train = chagatai_train
            baseline_dev_hash = file_sha256(variant_dir / "dev.csv")
            baseline_test_hash = file_sha256(variant_dir / "test.csv")
        else:
            assert_equal_frames(
                baseline_sources,
                chagatai_sources,
                f"{variant.name} Chagatai sources",
            )
            assert_equal_frames(
                baseline_train,
                chagatai_train,
                f"{variant.name} Chagatai train sequences",
            )
            if file_sha256(variant_dir / "dev.csv") != baseline_dev_hash:
                raise ValueError(f"{variant.name}: dev.csv changed across variants")
            if file_sha256(variant_dir / "test.csv") != baseline_test_hash:
                raise ValueError(f"{variant.name}: test.csv changed across variants")

        if variant.balance_auxiliary:
            stats = pd.read_csv(variant_dir / "stats.csv")
            train_stats = stats[stats["split"] == "train"]
            for method in ("sequential", "random", "partial"):
                method_stats = train_stats[train_stats["method"] == method]
                sequences = {
                    str(row.language): int(row.sequences)
                    for row in method_stats.itertuples(index=False)
                }
                expected = sequences.get("chg")
                if expected is None or any(
                    sequences.get(language) != expected
                    for language in expected_languages
                ):
                    raise ValueError(
                        f"{variant.name}: {method} sequence counts are not balanced"
                    )

        summaries[variant.name] = {
            "source_counts": source_counts,
            "sequence_counts": manifest.get("sequence_counts", {}),
            "sentence_balanced_to_chagatai": variant.balance_auxiliary,
            "manifest_sha256": file_sha256(variant_dir / "manifest.json"),
        }

    return {
        "schema_version": "2.0",
        "status": "passed",
        "variant_count": len(variants),
        "chagatai_train_sentences": chagatai_train_sentences,
        "checks": {
            "every_variant_passed_internal_validation": True,
            "chagatai_sources_identical_across_variants": True,
            "chagatai_train_sequences_identical_across_variants": True,
            "dev_identical_across_variants": True,
            "test_identical_across_variants": True,
            "balanced_variants_match_chagatai_sentence_and_sequence_counts": True,
        },
        "variants": summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and cross-audit the five canonical dataset variants."
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANT_BY_NAME),
        default=list(VARIANT_BY_NAME),
        help="Subset to build; by default all five variants are generated.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augmentation-multiplier", type=float, default=1.0)
    parser.add_argument("--uzs-sources", default="books")
    parser.add_argument("--chagatai-1-35", type=Path, default=DEFAULT_CHAGATAI_1_35)
    parser.add_argument("--chagatai-35-181", type=Path, default=DEFAULT_CHAGATAI_35_181)
    parser.add_argument("--uzs-parquet", type=Path, default=DEFAULT_UZS_PARQUET)
    parser.add_argument("--uyghur-csv", type=Path, default=DEFAULT_UYGHUR_CSV)
    parser.add_argument(
        "--export-stanza",
        action="store_true",
        help="Also create the derived Stanza character-level representation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.augmentation_multiplier < 0:
        raise ValueError("Augmentation multiplier must be non-negative")

    selected = [VARIANT_BY_NAME[name] for name in args.variants]
    train_count = chagatai_train_count(
        args.chagatai_1_35,
        args.chagatai_35_181,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"Chagatai train sentences used as the balance target: {train_count:,}")

    for index, variant in enumerate(selected, start=1):
        print(f"\n[{index}/{len(selected)}] Building {variant.name}", flush=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{variant.name}.",
            dir=args.output_root,
        ) as temporary_root:
            staging_dir = Path(temporary_root) / variant.name
            command = build_command(
                variant,
                output_dir=staging_dir,
                chagatai_train_sentences=train_count,
                seed=args.seed,
                augmentation_multiplier=args.augmentation_multiplier,
                chagatai_1_35=args.chagatai_1_35,
                chagatai_35_181=args.chagatai_35_181,
                uzs_parquet=args.uzs_parquet,
                uyghur_csv=args.uyghur_csv,
                uzs_sources=args.uzs_sources,
                export_stanza=args.export_stanza,
            )
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)

            final_dir = args.output_root / variant.name
            if final_dir.exists():
                shutil.rmtree(final_dir)
            staging_dir.replace(final_dir)

    report = audit_variants(args.output_root, selected, train_count)
    report_path = args.output_root / "variants_manifest.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nAll {len(selected)} variants passed: {report_path}")


if __name__ == "__main__":
    main()
