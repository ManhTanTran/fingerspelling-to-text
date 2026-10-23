import pytest
import torch

from fingerspelling.decoding.ctc_greedy import (
    collapse_repeats_and_blanks,
    greedy_decode,
)


def test_ctc_collapses_adjacent_repeats_and_removes_blank():
    assert collapse_repeats_and_blanks([0, 1, 1, 0, 1, 2, 2, 0], blank_index=0) == [1, 1, 2]


def test_greedy_decode_uses_explicit_mapping_and_blank():
    logits = torch.tensor(
        [[9.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 7.0, 0.0], [8.0, 0.0, 0.0], [0.0, 6.0, 0.0]]
    )
    assert greedy_decode(logits, {1: "A", 2: "B"}, blank_index=0) == "AA"


def test_greedy_decode_respects_input_length():
    logits = torch.tensor([[0.0, 4.0], [0.0, 3.0], [0.0, 2.0]])
    assert greedy_decode(logits, ["_", "x"], blank_index=0, input_length=1) == "x"


def test_greedy_decode_rejects_invalid_length_and_missing_token():
    logits = torch.tensor([[0.0, 1.0]])
    with pytest.raises(ValueError, match="input_length"):
        greedy_decode(logits, ["_", "x"], blank_index=0, input_length=2)
    with pytest.raises(ValueError, match="missing"):
        greedy_decode(logits, ["_"], blank_index=0)

