from __future__ import annotations

import re
import unicodedata
from collections import Counter


PARENTHETICAL_FOOTNOTE_RE = re.compile(
    r"[\(\[\{]\s*[0-9\u0660-\u0669\u06f0-\u06f9]+\s*[\)\]\}]"
)
SPACE_RE = re.compile(r"\s+")
URL_REMNANT_RE = re.compile(
    r"(?:^|\s)(?:https?|www|doi|isbn|html?|php|"
    r"com|org|net|edu|gov|io|ru|uk|cn|de|tr|xyz)(?:\s|$)",
    re.IGNORECASE,
)
MIN_ARABIC_LETTER_RATIO = 0.90
FORBIDDEN_AUXILIARY_SCRIPTS = ("LATIN", "CYRILLIC", "CJK")


def normalize_spaces(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def clean_sentence(text: str) -> str:
    """Normalize text and remove punctuation, symbols, marks, and controls."""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\u200c", " ").replace("\u200d", " ")
    text = PARENTHETICAL_FOOTNOTE_RE.sub(" ", text)

    cleaned: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category[0] == "M":
            continue
        if category[0] in {"P", "S"} or category == "Cf":
            cleaned.append(" ")
        else:
            cleaned.append(char)
    return normalize_spaces("".join(cleaned))


def has_word_character(token: str) -> bool:
    return any(char.isalpha() or char.isdigit() for char in token)


def letter_script_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "ARABIC" in name:
            counts["arabic"] += 1
        elif "LATIN" in name:
            counts["latin"] += 1
        elif "CYRILLIC" in name:
            counts["cyrillic"] += 1
        elif "CJK" in name or "IDEOGRAPH" in name:
            counts["cjk"] += 1
        else:
            counts["other"] += 1
    return counts


def arabic_letter_ratio(text: str) -> float:
    counts = letter_script_counts(text)
    total = sum(counts.values())
    return counts["arabic"] / total if total else 0.0


def source_noise_reasons(
    cleaned_text: str,
    *,
    reject_mixed_scripts: bool = True,
) -> tuple[str, ...]:
    """Return deterministic reasons for rejecting a cleaned source sentence."""
    reasons: list[str] = []
    counts = letter_script_counts(cleaned_text)
    if sum(counts.values()) == 0:
        reasons.append("no_letters")
    elif arabic_letter_ratio(cleaned_text) < MIN_ARABIC_LETTER_RATIO:
        reasons.append("low_arabic_letter_ratio")

    if reject_mixed_scripts:
        if counts["latin"]:
            reasons.append("latin_script")
        if counts["cyrillic"]:
            reasons.append("cyrillic_script")
        if counts["cjk"]:
            reasons.append("cjk_script")

    if URL_REMNANT_RE.search(cleaned_text):
        reasons.append("url_or_bibliography_remnant")
    return tuple(dict.fromkeys(reasons))
