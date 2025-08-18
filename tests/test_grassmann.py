import torch

from grassmann.GrassmannDistribution import GrassmannBinary
from grassmann.utils import ARAI_REFERENCE_LAMBDA, ARAI_REFERENCE_SIGMA, check_valid_sigma


def test_GrassmannBinary():
    # define three example events
    x = torch.zeros((3, 5))
    x[0, 0] = 1

    x[1, 1] = 1

    x[2, 0] = 0
    x[2, 1] = 1
    x[2, 2] = 0
    x[2, 3] = 0
    x[2, 4] = 1

    # x should have these probs
    prob_x_check = torch.tensor([0.0232, 0.0018, 0.0101])

    gr = GrassmannBinary(ARAI_REFERENCE_SIGMA)

    assert torch.allclose(gr.lambd, ARAI_REFERENCE_LAMBDA, atol=1e-4)
    assert torch.allclose(gr.prob(x), prob_x_check, atol=1e-4)
    assert check_valid_sigma(ARAI_REFERENCE_SIGMA)
