import torch
import pytest

from fingerspelling.inference import Predictor, load_checkpoint
from fingerspelling.models import BiGRUCTC


def _save_checkpoint(path, input_dim):
    chars = {"a": 1, "b": 2}
    model = BiGRUCTC(input_dim=input_dim, hidden_size=4, num_layers=2, num_classes=3)
    torch.save({
        "model_state_dict": model.state_dict(),
        "char_to_idx": chars,
        "idx_to_char": {index: char for char, index in chars.items()},
        "blank_idx": 0,
        "input_dim": input_dim,
        "hidden_size": 4,
        "num_layers": 2,
        "num_classes": 3,
    }, path)


@pytest.mark.parametrize("input_dim", [63, 126])
def test_predictor_accepts_features_and_pose(tmp_path, input_dim):
    path = tmp_path / "model.pt"
    _save_checkpoint(path, input_dim)
    predictor = Predictor.from_checkpoint(path, device="cpu")

    prediction = predictor.predict_features(torch.zeros(3, input_dim))
    pose_prediction = predictor.predict_pose(torch.zeros(3, 21, 3).numpy())

    assert isinstance(prediction.text, str)
    assert prediction.latency_ms >= 0
    assert isinstance(pose_prediction.text, str)


def test_predictor_rejects_mismatched_or_empty_features(tmp_path):
    path = tmp_path / "model.pt"
    _save_checkpoint(path, 126)
    predictor = Predictor(load_checkpoint(path, device="cpu"))
    with pytest.raises(ValueError, match="shape"):
        predictor.predict_features(torch.zeros(2, 63))
    with pytest.raises(ValueError, match="at least one frame"):
        predictor.predict_features(torch.zeros(0, 126))

