import pytest

from fingerspelling.evaluation.metrics import corpus_error_metrics, edit_counts


def test_edit_counts_report_substitution_deletion_and_insertion():
    assert edit_counts("abc", "axc") == {
        "distance": 1,
        "substitutions": 1,
        "deletions": 0,
        "insertions": 0,
    }
    assert edit_counts("abc", "ac")["deletions"] == 1
    assert edit_counts("abc", "abxc")["insertions"] == 1


def test_corpus_metrics_are_micro_averaged_not_mean_of_sample_rates():
    metrics = corpus_error_metrics(["a", "bbb"], ["x", "bb"])
    assert metrics["cer"] == pytest.approx(2 / 4)
    assert metrics["char_edits"] == 2
    assert metrics["reference_characters"] == 4
    assert metrics["exact_match"] == 0


def test_word_error_rate_uses_whitespace_tokens_and_reports_counts():
    metrics = corpus_error_metrics(["hello world"], ["hello there"])
    assert metrics["wer"] == pytest.approx(1 / 2)
    assert metrics["word_substitutions"] == 1
    assert metrics["word_deletions"] == 0
    assert metrics["word_insertions"] == 0


def test_empty_corpus_and_undefined_reference_denominator():
    assert corpus_error_metrics([], [])["cer"] is None
    metrics = corpus_error_metrics([""], ["new"])
    assert metrics["cer"] is None
    assert metrics["char_insertions"] == 3


def test_mismatched_lengths_and_non_string_inputs_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        corpus_error_metrics(["a"], [])
    with pytest.raises(TypeError, match="strings"):
        corpus_error_metrics([1], ["a"])

