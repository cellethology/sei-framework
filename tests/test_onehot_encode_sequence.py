import pytest
import torch

from embeddings.util import encode_sequence


def test_encode_sequence_basic_one_hot():
    encoded = encode_sequence("AGCT", sequence_length=8)

    assert encoded.shape == (4, 8)
    assert encoded.dtype == torch.float32
    assert torch.equal(encoded[:, 4:], torch.zeros(4, 4))
    assert encoded[0, 0] == 1.0  # A at pos 0 (A -> channel 0)
    assert encoded[2, 1] == 1.0  # G at pos 1 (G -> channel 2)
    assert encoded[1, 2] == 1.0  # C at pos 2 (C -> channel 1)
    assert encoded[3, 3] == 1.0  # T at pos 3 (T -> channel 3)


def test_encode_sequence_is_case_insensitive():
    encoded = encode_sequence("atgc", sequence_length=4)

    # Base mapping: A=0, C=1, G=2, T=3
    # Sequence "atgc": A(0) at pos 0, T(3) at pos 1, G(2) at pos 2, C(1) at pos 3
    assert torch.all(
        encoded
        == torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],  # A channel: A at pos 0
                [0.0, 0.0, 0.0, 1.0],  # C channel: C at pos 3
                [0.0, 0.0, 1.0, 0.0],  # G channel: G at pos 2
                [0.0, 1.0, 0.0, 0.0],  # T channel: T at pos 1
            ],
            dtype=torch.float32,
        )
    )


def test_encode_sequence_raises_on_unknown_bases():
    with pytest.raises(ValueError, match="Unknown base"):
        encode_sequence("AXTG", sequence_length=4)


def test_encode_sequence_raises_on_exceeding_max_length():
    with pytest.raises(ValueError, match="exceeds maximum allowed length"):
        encode_sequence("ATGCATGC", sequence_length=4)
