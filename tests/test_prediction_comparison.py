import csv

from fingerspelling.evaluation.comparison import compare_prediction_csvs


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["reference", "prediction"])
        writer.writeheader()
        writer.writerows(rows)


def test_comparison_directly_checks_smoke_prefix_against_reference(tmp_path):
    reference = tmp_path / "reference.csv"
    generated = tmp_path / "inference.csv"
    _write_csv(reference, [
        {"reference": "HELLO", "prediction": "HELLO"},
        {"reference": "WORLD", "prediction": "WORD"},
        {"reference": "THREE", "prediction": "THREE"},
    ])
    _write_csv(generated, [
        {"reference": "HELLO", "prediction": "HELLO"},
        {"reference": "WORLD", "prediction": "WORLD"},
    ])

    report = compare_prediction_csvs(generated, reference, max_rows=2)

    assert report["row_counts_match"] is True
    assert report["all_reference_text_matches"] is True
    assert report["all_predictions_match"] is False
    assert report["prediction_matches"] == 1
    assert report["mismatch_count"] == 1
    assert report["mismatches"][0]["row"] == 2


def test_comparison_detects_order_or_preprocessing_divergence(tmp_path):
    reference = tmp_path / "reference.csv"
    generated = tmp_path / "inference.csv"
    _write_csv(reference, [{"reference": "A B", "prediction": "A B"}])
    _write_csv(generated, [{"reference": "AB", "prediction": "AB"}])

    report = compare_prediction_csvs(generated, reference)

    assert report["all_reference_text_matches"] is False
    assert report["all_predictions_match"] is False
    assert report["mismatches"][0]["reference_text_matches"] is False
