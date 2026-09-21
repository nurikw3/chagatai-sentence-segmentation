"""Canonical, model-agnostic sentence-boundary dataset utilities."""

from .cleaning import clean_sentence
from .sources import SourceSentence

__all__ = ["SourceSentence", "clean_sentence"]
