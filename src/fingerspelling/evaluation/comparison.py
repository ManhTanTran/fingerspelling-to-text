"""Row-wise comparison of fresh inference CSVs against saved references."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from fingerspelling.storage.integrity import sha256_file


def _read_prediction_rows(path: str | Path) -> list[dict[str, str]]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"reference", "prediction"}.issubset(reader.fieldnames):
            raise ValueError(f"prediction CSV must include reference and prediction columns: {source}")
        rows = list(reader)
    return rows


def compare_prediction_csvs(
    generated_csv: str | Path,
    reference_csv: str | Path,
    *,
    max_rows: int | None = None,
    max_mismatches: int = 25,
) -> dict[str, Any]:
    """Compare fresh predictions directly to an archived prediction artifact.

    Since the reference artifact has no source-index column, rows are aligned
    by notebook dataset order; reference-text equality is reported separately
    to expose any ordering or preprocessing divergence.
    """
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive when provided")
    if max_mismatches < 0:
        raise ValueError("max_mismatches cannot be negative")

    generated_path = Path(generated_csv).expanduser().resolve()
    reference_path = Path(reference_csv).expanduser().resolve()
    generated = _read_prediction_rows(generated_path)
    reference = _read_prediction_rows(reference_path)
    if max_rows is None:
        compared_generated = generated
        compared_reference = reference
    else:
        compared_generated = generated[:max_rows]
        compared_reference = reference[:max_rows]

    compared_count = min(len(compared_generated), len(compared_reference))
    reference_matches = 0
    prediction_matches = 0
    mismatch_total = 0
    mismatches: list[dict[str, Any]] = []
    for index in range(compared_count):
        generated_row = compared_generated[index]
        reference_row = compared_reference[index]
        reference_equal = generated_row["reference"] == reference_row["reference"]
        prediction_equal = generated_row["prediction"] == reference_row["prediction"]
        reference_matches += reference_equal
        prediction_matches += prediction_equal
        if not reference_equal or not prediction_equal:
            mismatch_total += 1
            if len(mismatches) < max_mismatches:
                mismatches.append({
                    "row": index + 1,
                    "reference_text_matches": reference_equal,
                    "prediction_matches": prediction_equal,
                    "dataset_reference": reference_row["reference"],
                    "inference_reference": generated_row["reference"],
                    "saved_prediction": reference_row["prediction"],
                    "inference_prediction": generated_row["prediction"],
                })

    generated_count_matches = (
        len(generated) == max_rows if max_rows is not None else len(generated) == len(reference)
    )
    reference_has_requested_rows = len(reference) >= max_rows if max_rows is not None else True
    all_rows_aligned = (
        generated_count_matches
        and reference_has_requested_rows
        and compared_count == len(compared_generated)
        and compared_count == len(compared_reference)
    )
    all_reference_text_matches = all_rows_aligned and reference_matches == compared_count
    all_predictions_match = all_rows_aligned and prediction_matches == compared_count
    return {
        "generated_csv": str(generated_path),
        "generated_csv_sha256": sha256_file(generated_path),
        "reference_csv": str(reference_path),
        "reference_csv_sha256": sha256_file(reference_path),
        "alignment": "row order; reference text equality is checked per row",
        "generated_rows": len(generated),
        "reference_rows": len(reference),
        "max_rows": max_rows,
        "compared_rows": compared_count,
        "row_counts_match": all_rows_aligned,
        "reference_text_matches": reference_matches,
        "prediction_matches": prediction_matches,
        "all_reference_text_matches": all_reference_text_matches,
        "all_predictions_match": all_predictions_match,
        "mismatch_count": mismatch_total,
        "mismatches": mismatches,
    }
