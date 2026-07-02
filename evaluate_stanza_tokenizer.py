from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stanza.models import tokenizer
from stanza.models.tokenization.data import TokenizationDataset
from stanza.models.tokenization.trainer import Trainer
from stanza.models.tokenization.utils import load_mwt_dict, output_predictions


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "stanza_chg" / "models" / "chg_sic_tokenizer.pt"
DEFAULT_DATA_DIR = BASE_DIR / "stanza_chg" / "tokenizer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Chagatai Stanza tokenizer on the full test split."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def precision_recall_f1(
    predictions: np.ndarray, gold: np.ndarray, positive_labels: set[int]
) -> tuple[float, float, float, int, int, int]:
    pred_positive = np.isin(predictions, list(positive_labels))
    gold_positive = np.isin(gold, list(positive_labels))

    tp = int(np.logical_and(pred_positive, gold_positive).sum())
    fp = int(np.logical_and(pred_positive, ~gold_positive).sum())
    fn = int(np.logical_and(~pred_positive, gold_positive).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1, tp, fp, fn


def main() -> None:
    cli_args = parse_args()
    if not cli_args.model.exists():
        raise SystemExit(f"Model not found: {cli_args.model}")

    text_path = cli_args.data_dir / "test.txt"
    label_path = cli_args.data_dir / "test.toklabels"
    mwt_path = cli_args.data_dir / "mwt.json"

    runtime_args = tokenizer.parse_args(
        [
            "--mode",
            "predict",
            "--txt_file",
            str(text_path),
            "--label_file",
            str(label_path),
            "--mwt_json_file",
            str(mwt_path),
            "--lang",
            "chg",
            "--shorthand",
            "chg_sic",
            "--device",
            cli_args.device,
        ]
    )

    trainer = Trainer(
        args=runtime_args,
        model_file=str(cli_args.model),
        device=cli_args.device,
        foundation_cache=None,
    )
    for key, value in trainer.args.items():
        if not key.endswith("_file") and key not in {
            "device",
            "mode",
            "save_dir",
            "load_name",
            "save_name",
        }:
            runtime_args[key] = value

    batches = TokenizationDataset(
        runtime_args,
        input_files={"txt": str(text_path), "label": str(label_path)},
        vocab=trainer.vocab,
        evaluation=True,
        dictionary=trainer.dictionary,
    )
    mwt_dict = load_mwt_dict(str(mwt_path))
    _, _, prediction_chunks, _ = output_predictions(
        None,
        trainer,
        batches,
        trainer.vocab,
        mwt_dict,
        runtime_args["max_seqlen"],
    )

    gold_chunks = batches.labels()
    predictions = np.concatenate(prediction_chunks)
    gold = np.concatenate(gold_chunks)

    token_metrics = precision_recall_f1(predictions, gold, {1, 2, 3, 4})
    sentence_metrics = precision_recall_f1(predictions, gold, {2, 4})

    exact = 0
    for pred_chunk, gold_chunk in zip(prediction_chunks, gold_chunks):
        pred_boundaries = np.isin(pred_chunk, [2, 4])
        gold_boundaries = np.isin(gold_chunk, [2, 4])
        exact += int(np.array_equal(pred_boundaries, gold_boundaries))

    def print_metrics(name: str, values: tuple[float, float, float, int, int, int]) -> None:
        precision, recall, f1, tp, fp, fn = values
        print(
            f"{name}: precision={precision:.4f} recall={recall:.4f} "
            f"f1={f1:.4f} tp={tp} fp={fp} fn={fn}"
        )

    print(f"Test samples: {len(gold_chunks)}")
    print_metrics("Token boundaries", token_metrics)
    print_metrics("Sentence boundaries", sentence_metrics)
    print(f"Sentence exact match: {exact}/{len(gold_chunks)} = {exact / len(gold_chunks):.4f}")


if __name__ == "__main__":
    main()
