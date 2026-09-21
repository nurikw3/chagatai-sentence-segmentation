from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILDS_ROOT = PROJECT_ROOT / "data" / "UNIFIED" / "builds"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "UNIFIED" / "hf_export"
VARIANT_NAMES = (
    "chagatai_only",
    "chagatai_uyghur",
    "chagatai_uzs",
    "chagatai_uzs_uyghur_full",
    "chagatai_uzs_uyghur_balanced",
)
JSON_COLUMNS = (
    "source_sentence_ids",
    "fragment_spans",
    "boundary_at_end",
    "tokens",
    "labels",
)
SPLIT_NAMES = {"train": "train"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_csv_to_parquet_shards(
    csv_path: Path,
    output_dir: Path,
    split_name: str,
    rows_per_shard: int,
    *,
    parse_json_columns: bool,
) -> tuple[int, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_count = 0
    output_paths: list[Path] = []
    for shard_index, frame in enumerate(
        pd.read_csv(csv_path, chunksize=rows_per_shard, keep_default_na=False)
    ):
        if parse_json_columns:
            for column in JSON_COLUMNS:
                frame[column] = frame[column].map(json.loads)
        output_path = output_dir / f"{split_name}-{shard_index:05d}.parquet"
        frame.to_parquet(output_path, index=False, compression="zstd")
        row_count += len(frame)
        output_paths.append(output_path)
    if not output_paths:
        raise ValueError(f"No rows found in {csv_path}")
    return row_count, output_paths


def load_manifests(builds_root: Path) -> dict[str, dict[str, object]]:
    manifests: dict[str, dict[str, object]] = {}
    for name in VARIANT_NAMES:
        path = builds_root / name / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing build manifest: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("checks", {}).get("status") != "passed":
            raise ValueError(f"{name}: build validation did not pass")
        manifests[name] = manifest
    return manifests


def render_dataset_card(manifests: dict[str, dict[str, object]]) -> str:
    yaml_lines = [
        "---",
        "pretty_name: Chagatai Sentence Boundary Detection",
        "language:",
        "- chg",
        "- uz",
        "- ug",
        "task_categories:",
        "- token-classification",
        "tags:",
        "- sentence-boundary-detection",
        "- chagatai",
        "- word-level-labeling",
        "configs:",
    ]
    for name in VARIANT_NAMES:
        yaml_lines.extend(
            [
                f"- config_name: {name}",
                *( ["  default: true"] if name == "chagatai_only" else [] ),
                "  data_files:",
                "  - split: train",
                f'    path: "data/{name}/train-*.parquet"',
                "  - split: validation",
                '    path: "data/common/validation-*.parquet"',
                "  - split: test",
                '    path: "data/common/test-*.parquet"',
            ]
        )
    yaml_lines.append("---")

    table_rows = []
    for name in VARIANT_NAMES:
        manifest = manifests[name]
        source_counts = manifest["source_counts"]
        sequence_counts = manifest["sequence_counts"]
        languages = [f"Chagatai {source_counts['train:chg']:,}"]
        if "train:uzs" in source_counts:
            languages.append(f"UZS {source_counts['train:uzs']:,}")
        if "train:uig" in source_counts:
            languages.append(f"Uyghur {source_counts['train:uig']:,}")
        table_rows.append(
            "| `{name}` | {sources} | {train:,} | {dev:,} | {test:,} |".format(
                name=name,
                sources=", ".join(languages),
                train=sequence_counts["train"],
                dev=sequence_counts["dev"],
                test=sequence_counts["test"],
            )
        )

    body = """
# Chagatai Sentence Boundary Detection

Canonical word-level Sentence Boundary Detection data for Chagatai. South
Uzbek (`uzs`) and Uyghur (`uig`) are optional train-only auxiliary languages.
Every configuration uses the same Chagatai train source split. Validation and
test are physically shared files referenced by all five configurations.

## Load with datasets

```python
from datasets import load_dataset

dataset = load_dataset("chagatai-project/chagatai-sbd", "chagatai_only")
balanced = load_dataset(
    "chagatai-project/chagatai-sbd",
    "chagatai_uzs_uyghur_balanced",
)

train = balanced["train"]
validation = balanced["validation"]
test = balanced["test"]
```

The full multilingual configuration is large. It can be streamed without a
complete local download:

```python
full_train = load_dataset(
    "chagatai-project/chagatai-sbd",
    "chagatai_uzs_uyghur_full",
    split="train",
    streaming=True,
)
```

## Configurations

| Configuration | Train source sentences | Train sequences | Validation | Test |
|---|---:|---:|---:|---:|
{table_rows}

The pairwise and balanced configurations cap every auxiliary language at
3,035 source sentences, matching Chagatai train. This is sentence- and
sequence-level balance, not token-level balance.

> **Warning**: The `chagatai_uzs_uyghur_full` configuration is heavily skewed
> towards Uyghur (238,344 Uyghur source sentences vs. 3,035 Chagatai). For
> cross-lingual and balanced multilingual training, we strongly recommend using
> `chagatai_uzs_uyghur_balanced`.

Validation contains 433 Chagatai source sentences grouped into 145 sequential
examples. Test contains 868 Chagatai source sentences grouped into 295
sequential examples.

## Labels and columns

`labels` is aligned with `tokens`:

- `0`: the word does not end a sentence;
- `1`: the word is the final word of a real source sentence.

Each row also includes the cleaned `text`, language, augmentation method,
source sentence IDs, fragment spans, boundary flags, and summary counts.

## Construction rules

Chagatai is split in source order 70/10/20 before augmentation. Validation and
test contain sequential Chagatai only. Train uses:

- sequential concatenation of 2-4 neighboring source sentences;
- random concatenation of 2-5 distinct, nonadjacent, reordered sources;
- partial merge of the end of one sentence and the start of another.

In partial examples, the end of the first fragment remains EOS and the end of
the second fragment is not EOS because the original sentence continues.

The cleaning pass applies NFKC, removes punctuation, symbols, controls, and
combining marks, and rejects residual Latin, Cyrillic, CJK, URL, and
bibliography noise.

## Validation and provenance

All five configurations passed source-split leakage, exact sequence overlap,
label length, fragment reconstruction, auxiliary train-only, random ordering,
and partial-boundary checks. Build manifests and statistics are stored under
`metadata/<configuration>/`.

Cleaned source tables are available under `sources/<configuration>/`. They are
not loaded as model splits, but retain source locators and raw text for audit
and reconstruction.

The source corpora retain their original terms. No new license is asserted for
the combined dataset; users should verify the terms of each source before
redistribution or commercial use.
""".strip()
    return "\n".join(yaml_lines) + "\n\n" + body.format(
        table_rows="\n".join(table_rows)
    ) + "\n"


def prepare_export(
    builds_root: Path,
    output_dir: Path,
    rows_per_shard: int,
) -> None:
    manifests = load_manifests(builds_root)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ) as temporary_root:
        staging = Path(temporary_root) / output_dir.name
        staging.mkdir(parents=True)

        common_build = builds_root / "chagatai_only"
        common_manifest = manifests["chagatai_only"]
        for local_split, hub_split in (("dev", "validation"), ("test", "test")):
            common_csv = common_build / f"{local_split}.csv"
            common_hash = file_sha256(common_csv)
            for name in VARIANT_NAMES[1:]:
                candidate = builds_root / name / f"{local_split}.csv"
                if file_sha256(candidate) != common_hash:
                    raise ValueError(
                        f"{name}/{local_split}: split differs from chagatai_only"
                    )
            row_count, _ = convert_csv_to_parquet_shards(
                common_csv,
                staging / "data" / "common",
                hub_split,
                rows_per_shard,
                parse_json_columns=True,
            )
            expected = int(common_manifest["sequence_counts"][local_split])
            if row_count != expected:
                raise ValueError(
                    f"common/{local_split}: expected {expected} rows, got {row_count}"
                )

        for name in VARIANT_NAMES:
            build_dir = builds_root / name
            manifest = manifests[name]
            for local_split, hub_split in SPLIT_NAMES.items():
                row_count, _ = convert_csv_to_parquet_shards(
                    build_dir / f"{local_split}.csv",
                    staging / "data" / name,
                    hub_split,
                    rows_per_shard,
                    parse_json_columns=True,
                )
                expected = int(manifest["sequence_counts"][local_split])
                if row_count != expected:
                    raise ValueError(
                        f"{name}/{local_split}: expected {expected} rows, got {row_count}"
                    )

            source_count, _ = convert_csv_to_parquet_shards(
                build_dir / "source_sentences.csv",
                staging / "sources" / name,
                "source_sentences",
                rows_per_shard,
                parse_json_columns=False,
            )
            expected_sources = sum(
                int(value) for value in manifest["source_counts"].values()
            )
            if source_count != expected_sources:
                raise ValueError(
                    f"{name}: expected {expected_sources} sources, got {source_count}"
                )

            metadata_dir = staging / "metadata" / name
            metadata_dir.mkdir(parents=True)
            shutil.copy2(build_dir / "manifest.json", metadata_dir / "manifest.json")
            shutil.copy2(build_dir / "stats.csv", metadata_dir / "stats.csv")

        variants_manifest = builds_root / "variants_manifest.json"
        if variants_manifest.exists():
            shutil.copy2(variants_manifest, staging / "variants_manifest.json")
        (staging / "README.md").write_text(
            render_dataset_card(manifests),
            encoding="utf-8",
        )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.replace(output_dir)


def upload_to_hub(
    output_dir: Path,
    repo_id: str,
    *,
    private: bool = False,
    token: str | None = None,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"Ensuring repository exists: {repo_id} (dataset)...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=private,
    )
    print(f"Uploading files from {output_dir} to {repo_id}...")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        commit_message="Upload Chagatai SBD dataset (5 configurations, typed Parquet)",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare typed, sharded Parquet files for Hugging Face Hub."
    )
    parser.add_argument("--builds-root", type=Path, default=DEFAULT_BUILDS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rows-per-shard", type=int, default=50_000)
    parser.add_argument(
        "--upload-repo-id",
        type=str,
        default=None,
        help="Optional Hugging Face repo ID (e.g. chagatai-project/chagatai-sbd) to upload to.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repository as private if it does not already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows_per_shard <= 0:
        raise ValueError("rows-per-shard must be positive")
    prepare_export(args.builds_root, args.output_dir, args.rows_per_shard)
    total_bytes = sum(
        path.stat().st_size for path in args.output_dir.rglob("*") if path.is_file()
    )
    print(f"Prepared: {args.output_dir}")
    print(f"Files: {sum(1 for path in args.output_dir.rglob('*') if path.is_file())}")
    print(f"Size: {total_bytes / (1024 ** 2):.1f} MiB")

    if args.upload_repo_id:
        upload_to_hub(args.output_dir, args.upload_repo_id, private=args.private)
        print(
            f"Successfully uploaded to https://huggingface.co/datasets/{args.upload_repo_id}"
        )


if __name__ == "__main__":
    main()
