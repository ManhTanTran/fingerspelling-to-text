"""Corpus-level character/word error metrics and edit-operation counts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def edit_counts(reference: Sequence[Any], hypothesis: Sequence[Any]) -> dict[str, int]:
    """Return Levenshtein distance and deterministic S/D/I alignment counts."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    distance = [[0] * cols for _ in range(rows)]
    back: list[list[str | None]] = [[None] * cols for _ in range(rows)]

    for i in range(1, rows):
        distance[i][0] = i
        back[i][0] = "D"
    for j in range(1, cols):
        distance[0][j] = j
        back[0][j] = "I"

    for i in range(1, rows):
        for j in range(1, cols):
            same = reference[i - 1] == hypothesis[j - 1]
            choices = [
                (distance[i - 1][j - 1] + (0 if same else 1), "M" if same else "S"),
                (distance[i - 1][j] + 1, "D"),
                (distance[i][j - 1] + 1, "I"),
            ]
            # Stable preference for diagonal, then deletion, then insertion.
            distance[i][j], back[i][j] = min(choices, key=lambda item: item[0])

    substitutions = deletions = insertions = 0
    i, j = len(reference), len(hypothesis)
    while i or j:
        operation = back[i][j]
        if operation in ("M", "S"):
            substitutions += operation == "S"
            i -= 1
            j -= 1
        elif operation == "D":
            deletions += 1
            i -= 1
        elif operation == "I":
            insertions += 1
            j -= 1
        else:
            raise RuntimeError("invalid edit-distance backtrace")

    return {
        "distance": distance[-1][-1],
        "substitutions": int(substitutions),
        "deletions": deletions,
        "insertions": insertions,
    }


def _rate(errors: int, reference_units: int) -> float | None:
    return errors / reference_units if reference_units else None


def corpus_error_metrics(
    references: Sequence[str], hypotheses: Sequence[str]
) -> dict[str, int | float | None]:
    """Compute micro-averaged corpus CER/WER, exact match, and S/D/I counts.

    Text is used as supplied: there is no hidden lowercasing, punctuation
    stripping, or whitespace normalization. CER includes spaces; WER splits on
    Python whitespace. A rate is ``None`` when the corresponding corpus has no
    reference units, because that denominator is undefined.
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    totals = {
        "char_edits": 0,
        "reference_characters": 0,
        "char_substitutions": 0,
        "char_deletions": 0,
        "char_insertions": 0,
        "word_edits": 0,
        "reference_words": 0,
        "word_substitutions": 0,
        "word_deletions": 0,
        "word_insertions": 0,
    }
    exact = 0
    for reference, hypothesis in zip(references, hypotheses):
        if not isinstance(reference, str) or not isinstance(hypothesis, str):
            raise TypeError("all references and hypotheses must be strings")
        chars = edit_counts(reference, hypothesis)
        words = edit_counts(reference.split(), hypothesis.split())
        totals["char_edits"] += chars["distance"]
        totals["reference_characters"] += len(reference)
        totals["char_substitutions"] += chars["substitutions"]
        totals["char_deletions"] += chars["deletions"]
        totals["char_insertions"] += chars["insertions"]
        totals["word_edits"] += words["distance"]
        totals["reference_words"] += len(reference.split())
        totals["word_substitutions"] += words["substitutions"]
        totals["word_deletions"] += words["deletions"]
        totals["word_insertions"] += words["insertions"]
        exact += reference == hypothesis

    count = len(references)
    return {
        "sample_count": count,
        "cer": _rate(totals["char_edits"], totals["reference_characters"]),
        "wer": _rate(totals["word_edits"], totals["reference_words"]),
        "exact_match": exact / count if count else None,
        **totals,
    }

