from __future__ import annotations

import argparse
from pathlib import Path

from stanza.models import tokenizer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "stanza_chg" / "tokenizer"
MODEL_DIR = BASE_DIR / "stanza_chg" / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Stanza tokenizer/SBD model for Chagatai."
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--report-steps", type=int, default=25)
    parser.add_argument("--early-stop-steps", type=int, default=500)
    parser.add_argument("--max-seqlen", type=int, default=800)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--save-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--save-name", default="chg_sic_tokenizer.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.main(
        [
            "--txt_file",
            str(DATA_DIR / "train.txt"),
            "--label_file",
            str(DATA_DIR / "train.toklabels"),
            "--dev_txt_file",
            str(DATA_DIR / "dev.txt"),
            "--dev_label_file",
            str(DATA_DIR / "dev.toklabels"),
            "--mwt_json_file",
            str(DATA_DIR / "mwt.json"),
            "--lang",
            "chg",
            "--shorthand",
            "chg_sic",
            "--save_dir",
            str(args.save_dir),
            "--save_name",
            args.save_name,
            "--mode",
            "train",
            "--device",
            args.device,
            "--steps",
            str(args.steps),
            "--eval_steps",
            str(args.eval_steps),
            "--report_steps",
            str(args.report_steps),
            "--max_steps_before_stop",
            str(args.early_stop_steps),
            "--max_seqlen",
            str(args.max_seqlen),
        ]
    )


if __name__ == "__main__":
    main()
