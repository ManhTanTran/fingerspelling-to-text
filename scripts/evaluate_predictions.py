"""Recompute corpus metrics from a prediction CSV; metrics are never hard-coded."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fingerspelling.evaluation.metrics import corpus_error_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--reference-column", default="reference")
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            parser.error("CSV has no header")
        missing = {
            name for name in (args.reference_column, args.prediction_column)
            if name not in reader.fieldnames
        }
        if missing:
            parser.error(f"CSV is missing required columns: {sorted(missing)}")
        rows = list(reader)

    report = corpus_error_metrics(
        [row[args.reference_column] for row in rows],
        [row[args.prediction_column] for row in rows],
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()

