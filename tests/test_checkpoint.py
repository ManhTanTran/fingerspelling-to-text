import pytest
import torch

from fingerspelling.inference import load_checkpoint
from fingerspelling.models import BiGRUCTC


def _payload(input_dim=126):
    tokens = {"a": 1, "b": 2}
    inverse = {index: char for char, index in tokens.items()}
    model = BiGRUCTC(input_dim=input_dim, hidden_size=4, num_layers=2, num_classes=3)
    return {
        "model_state_dict": model.state_dict(),
        "char_to_idx": tokens,
        "idx_to_char": inverse,
        "blank_idx": 0,
        "input_dim": input_dim,
        "hidden_size": 4,
        "num_layers": 2,
        "num_classes": 3,
        "epoch": 3,
    }


def test_checkpoint_loads_weights_only_and_strictly(tmp_path):
    path = tmp_path / "trusted.pt"
    torch.save(_payload(), path)

    loaded = load_checkpoint(path, device="cpu")

    assert loaded.input_dim == 126
    assert loaded.vocabulary.blank_index == 0
    assert loaded.vocabulary.decode_path([0, 1, 1, 0, 1, 2, 2]) == "aab"
    assert loaded.epoch == 3
    assert not loaded.model.training


def test_checkpoint_strictly_rejects_state_dict_shape_mismatch(tmp_path):
    payload = _payload()
    payload["model_state_dict"]["classifier.bias"] = torch.zeros(4)
    path = tmp_path / "bad_shape.pt"
    torch.save(payload, path)

    with pytest.raises(RuntimeError):
        load_checkpoint(path, device="cpu")


def test_checkpoint_rejects_inconsistent_vocabulary(tmp_path):
    payload = _payload()
    payload["blank_idx"] = 2
    path = tmp_path / "bad_vocab.pt"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="cover every non-blank"):
        load_checkpoint(path, device="cpu")

