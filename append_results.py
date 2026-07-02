from __future__ import annotations

import argparse
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = BASE_DIR / "results.txt"


RUN_HEADING_RE = re.compile(r"^## Run \d+:", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append Kaggle-generated Run blocks to results.txt."
    )
    parser.add_argument(
        "blocks_file",
        type=Path,
        help="File produced by stanza.ipynb, e.g. results_run_blocks.txt.",
    )
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def run_headings(text: str) -> set[str]:
    return {line.strip() for line in RUN_HEADING_RE.findall(text)}


def main() -> None:
    args = parse_args()
    if not args.blocks_file.exists():
        raise SystemExit(f"Run blocks file not found: {args.blocks_file}")
    if not args.results_file.exists():
        raise SystemExit(f"Results file not found: {args.results_file}")

    blocks = args.blocks_file.read_text(encoding="utf-8").strip()
    if not blocks:
        raise SystemExit(f"Run blocks file is empty: {args.blocks_file}")

    existing = args.results_file.read_text(encoding="utf-8")
    duplicate = run_headings(existing) & run_headings(blocks)
    if duplicate:
        raise SystemExit(f"Refusing to append duplicate run heading(s): {sorted(duplicate)}")

    separator = "\n\n" if existing.endswith("\n") else "\n\n"
    args.results_file.write_text(existing.rstrip() + separator + blocks + "\n", encoding="utf-8")
    print(f"Appended {len(run_headings(blocks))} run block(s) to {args.results_file}")


if __name__ == "__main__":
    main()
