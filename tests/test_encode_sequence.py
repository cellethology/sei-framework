import pytest
import torch


from embeddings.util import encode_sequence


def test_encode_sequence_basic_one_hot():
    encoded = encode_sequence("ATGC", sequence_length=8)

    assert encoded.shape == (4, 8)
    assert encoded.dtype == torch.float32
    assert torch.equal(encoded[:, 4:], torch.zeros(4, 4))
    assert encoded[0, 0] == 1.0  # A at pos 0
    assert encoded[1, 1] == 1.0  # T at pos 1
    assert encoded[2, 2] == 1.0  # G at pos 2
    assert encoded[3, 3] == 1.0  # C at pos 3


def test_encode_sequence_is_case_insensitive():
    encoded = encode_sequence("atgc", sequence_length=4)

    assert torch.all(encoded == torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float32))


def test_encode_sequence_raises_on_unknown_bases():
    with pytest.raises(ValueError, match="Unknown base"):
        encode_sequence("AXTG", sequence_length=4)


def test_encode_sequence_truncates_to_max_length():
    encoded = encode_sequence("ATGCATGC", sequence_length=4)

    assert torch.count_nonzero(encoded) == 4
    assert encoded[0, 0] == 1.0
    assert encoded[1, 1] == 1.0
    assert encoded[2, 2] == 1.0
    assert encoded[3, 3] == 1.0
