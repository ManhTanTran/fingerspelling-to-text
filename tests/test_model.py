import pytest
import torch

from fingerspelling.models import BiGRUCTC


@pytest.mark.parametrize("input_dim", [63, 126])
def test_model_emits_expected_class_count_for_both_baselines(input_dim):
    model = BiGRUCTC(input_dim=input_dim, hidden_size=128, num_layers=2, num_classes=60)
    output = model(torch.zeros(2, 5, input_dim), torch.tensor([5, 3]))
    assert output.shape == (5, 2, 60)
    assert torch.isfinite(output).all()


def test_model_rejects_wrong_feature_dimension():
    model = BiGRUCTC(input_dim=126)
    with pytest.raises(ValueError, match="expected 126"):
        model(torch.zeros(1, 4, 63))


def test_model_packs_variable_length_sequences():
    model = BiGRUCTC(input_dim=63)
    output = model(torch.zeros(2, 5, 63), torch.tensor([5, 3]))
    torch.testing.assert_close(
        output[3:, 1], model.classifier.bias.expand(2, -1)
    )

