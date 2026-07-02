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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr0", type=float, default=2e-3)
    parser.add_argument("--anneal", type=float, default=0.995)
    parser.add_argument("--anneal-after", type=int, default=800)
    parser.add_argument("--dropout", type=float, default=0.33)
    parser.add_argument("--unit-dropout", type=float, default=0.33)
    parser.add_argument("--feat-dropout", type=float, default=0.05)
    parser.add_argument("--feat-unit-dropout", type=float, default=0.33)
    parser.add_argument("--tok-noise", type=float, default=0.0)
    parser.add_argument("--sent-drop-prob", type=float, default=0.0)
    parser.add_argument("--last-char-drop-prob", type=float, default=0.0)
    parser.add_argument("--last-char-move-prob", type=float, default=0.0)
    parser.add_argument("--punct-move-back-prob", type=float, default=0.0)
    parser.add_argument("--augment-final-punct-prob", type=float, default=0.0)
    parser.add_argument("--augment-mid-punct-prob", type=float, default=0.0)
    parser.add_argument("--split-mwt-prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--charlm",
        action="store_true",
        help="Enable Stanza contextual char-LM embeddings.",
    )
    parser.add_argument(
        "--charlm-shorthand",
        default=None,
        help="Optional shorthand of the CharLM training corpus.",
    )
    parser.add_argument("--charlm-forward-file", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--save-name", default="chg_sic_tokenizer.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_args = [
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
        "--batch_size",
        str(args.batch_size),
        "--lr0",
        str(args.lr0),
        "--anneal",
        str(args.anneal),
        "--anneal_after",
        str(args.anneal_after),
        "--dropout",
        str(args.dropout),
        "--unit_dropout",
        str(args.unit_dropout),
        "--feat_dropout",
        str(args.feat_dropout),
        "--feat_unit_dropout",
        str(args.feat_unit_dropout),
        "--tok_noise",
        str(args.tok_noise),
        "--sent_drop_prob",
        str(args.sent_drop_prob),
        "--last_char_drop_prob",
        str(args.last_char_drop_prob),
        "--last_char_move_prob",
        str(args.last_char_move_prob),
        "--punct_move_back_prob",
        str(args.punct_move_back_prob),
        "--augment_final_punct_prob",
        str(args.augment_final_punct_prob),
        "--augment_mid_punct_prob",
        str(args.augment_mid_punct_prob),
        "--split_mwt_prob",
        str(args.split_mwt_prob),
        "--seed",
        str(args.seed),
    ]

    use_charlm = args.charlm or args.charlm_forward_file is not None
    if use_charlm:
        if args.charlm_forward_file is None:
            raise SystemExit("--charlm requires --charlm-forward-file")
        if not args.charlm_forward_file.exists():
            raise SystemExit(f"CharLM model not found: {args.charlm_forward_file}")
        tokenizer_args.extend(
            ["--charlm", "--charlm_forward_file", str(args.charlm_forward_file)]
        )
        if args.charlm_shorthand is not None:
            tokenizer_args.extend(["--charlm_shorthand", args.charlm_shorthand])

    tokenizer.main(tokenizer_args)


if __name__ == "__main__":
    main()
