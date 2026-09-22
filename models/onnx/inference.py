"""
Chagatai Sentence Boundary Detection (SBD) — Standalone ONNX Inference Engine.

Zero-dependency inference module (requires only `numpy` and `onnxruntime`).
Supports standalone SOTA and unified multi-model ensembles packaged into single .onnx files.
"""

import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
import numpy as np
import onnxruntime as ort

NEWLINE_WHITESPACE_RE = re.compile(r'\n[\s\u0080-\u009f]*\n')
WHITESPACE_RE = re.compile(r'[\s\u0080-\u009f]')
NUMERIC_RE = re.compile(r'^[\d]+([,\.]+[\d]+)*[,\.]*$')


class ChagataiSBD:
    """
    High-performance, dependency-free Sentence Boundary Detector for Chagatai text.
    
    Loads a single unified ONNX model (either standalone or soft-voting ensemble)
    and predicts sentence boundaries directly on unsegmented text.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/onnx/chagatai_sbd_tri_hybrid_stanza.onnx",
        vocab_path: Union[str, Path] = "models/onnx/vocab.json",
        providers: List[str] = None
    ):
        model_path = Path(model_path)
        vocab_path = Path(vocab_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

        # Load vocabulary mapping
        with open(vocab_path, "r", encoding="utf-8") as f:
            v_data = json.load(f)
            self.unit2id: Dict[str, int] = v_data["unit2id"]
            self.pad_id: int = v_data["pad_id"]
            self.unk_id: int = v_data["unk_id"]

        # Initialize ONNX Runtime session
        if providers is None:
            providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(model_path), providers=providers)

    def _preprocess_chunk(self, chunk: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
        # Substitute non-standard whitespaces
        chars = [WHITESPACE_RE.sub(' ', c) for c in chunk]

        # Collapse consecutive spaces (matches Stanza pipeline)
        filtered_chars: List[str] = []
        for i, c in enumerate(chars):
            if i > 0 and c == ' ' and filtered_chars[-1] == ' ':
                continue
            filtered_chars.append(c)

        N = len(filtered_chars)
        char_ids = np.zeros(N + 1, dtype=np.int64)
        feats = np.zeros((N + 1, 5), dtype=np.float32)

        for i, c in enumerate(filtered_chars):
            char_ids[i] = self.unit2id.get(c, self.unk_id)
            feats[i, 0] = 1.0 if c.startswith(' ') else 0.0
            feats[i, 1] = 1.0 if c[0].isupper() else 0.0
            feats[i, 2] = 1.0 if NUMERIC_RE.match(c) else 0.0
            feats[i, 3] = 1.0 if i == (N - 1) else 0.0
            feats[i, 4] = 1.0 if i == 0 else 0.0

        char_ids[N] = self.pad_id
        feats[N, :] = 0.0

        return filtered_chars, char_ids, feats

    def segment(self, text: str) -> List[str]:
        """
        Segment a document or multi-line text into individual sentences.
        
        Args:
            text: Raw Chagatai text string.
            
        Returns:
            List of segmented sentences.
        """
        if not text or not text.strip():
            return []

        chunks = [pt.rstrip() for pt in NEWLINE_WHITESPACE_RE.split(text.rstrip()) if pt.rstrip()]
        all_sentences: List[str] = []

        for chunk in chunks:
            filtered_chars, char_ids, feats = self._preprocess_chunk(chunk)
            N = len(filtered_chars)
            if N == 0:
                continue

            # Run ONNX inference
            c_input = char_ids[np.newaxis, :]  # shape: [1, N + 1]
            f_input = feats[np.newaxis, :]     # shape: [1, N + 1, 5]

            p_sent, is_tok, is_eos = self.session.run(
                None,
                {'char_ids': c_input, 'feats': f_input}
            )

            eos_mask = is_eos[0, :N].copy()
            # Guarantee last non-space character in paragraph closes the sentence
            eos_mask[-1] = True

            curr_sent = []
            for c, is_end in zip(filtered_chars, eos_mask):
                curr_sent.append(c)
                if is_end:
                    s_str = "".join(curr_sent).strip()
                    if s_str:
                        all_sentences.append(s_str)
                    curr_sent = []

            if curr_sent:
                s_str = "".join(curr_sent).strip()
                if s_str:
                    all_sentences.append(s_str)

        return all_sentences


if __name__ == '__main__':
    # Smoke test on a sample Chagatai paragraph
    sbd = ChagataiSBD("models/onnx/chagatai_sbd_tri_hybrid_stanza.onnx", "models/onnx/vocab.json")
    
    sample_text = (
        "سلطان ابوسعید میرزا شهادت تاپقاندین سونگ سمرقند ولایتینی میرزا احمد آلدی "
        "و هرات ولایتینی سلطان حسین میرزا ضبط قیلدی و اندیجان ولایتینی عمرشیخ میرزا توتتی"
    )
    
    print("\nOriginal text:")
    print(sample_text)
    
    sentences = sbd.segment(sample_text)
    print(f"\nDetected {len(sentences)} sentences:")
    for idx, s in enumerate(sentences, 1):
        print(f"  [{idx}] {s}")
