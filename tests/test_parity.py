# pip install git+https://github.com/mackelab/grassmann_binary_distribution
import pytest
import torch
from grassmann_distribution.GrassmannDistribution import GrassmannBinary as ReferenceGrassmannBinary
from grassmann_distribution.GrassmannDistribution import MoGrassmannBinary as ReferenceMoGrassmannBinary

from grassmann.GrassmannDistribution import GrassmannBinary, MoGrassmannBinary
from grassmann.utils import ARAI_REFERENCE_SIGMA


def generate_valid_sigma(dim: int, seed: int, epsilon: float = 1e-4) -> torch.Tensor:
    torch.manual_seed(seed)
    B = torch.randn(dim, dim)
    C = torch.randn(dim, dim)

    # Make diagonals non-negative (relu on diag)
    B_diag = torch.relu(torch.diagonal(B))
    C_diag = torch.relu(torch.diagonal(C))
    B = B.clone()
    C = C.clone()
    B.diagonal().copy_(B_diag)
    C.diagonal().copy_(C_diag)

    # Make row-diagonally-dominant: set diag to sum(abs(row)) + eps
    B_row = torch.sum(torch.abs(B), dim=-1) + epsilon
    C_row = torch.sum(torch.abs(C), dim=-1) + epsilon
    idx = torch.arange(dim)
    B[idx, idx] = B_row
    C[idx, idx] = C_row

    lambd = B @ torch.inverse(C) + torch.eye(dim)
    sigma = torch.inverse(lambd)
    return sigma


SIGMA_CASES = [
    ARAI_REFERENCE_SIGMA.clone(),
    generate_valid_sigma(ARAI_REFERENCE_SIGMA.shape[0], seed=123),
    generate_valid_sigma(ARAI_REFERENCE_SIGMA.shape[0], seed=456),
]


@pytest.mark.parametrize("sigma", SIGMA_CASES)
def test_grassmann_parity_prob_mean_cov_corr(sigma: torch.Tensor):
    torch.manual_seed(0)
    dim = sigma.shape[0]

    ours = GrassmannBinary(sigma)
    ref = ReferenceGrassmannBinary(sigma)

    x = torch.bernoulli(0.5 * torch.ones(64, dim))

    p_ours = ours.prob(x)
    p_ref = ref.prob(x)

    assert torch.allclose(p_ours, p_ref, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ours.mean(), ref.mean(), rtol=1e-6, atol=1e-7)
    assert torch.allclose(ours.cov(), ref.cov(), rtol=1e-6, atol=1e-7)
    assert torch.allclose(ours.corr(), ref.corr(), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("sigma", SIGMA_CASES)
def test_grassmann_parity_conditional_sigma(sigma: torch.Tensor):
    torch.manual_seed(1)
    dim = sigma.shape[0]

    ours = GrassmannBinary(sigma)
    ref = ReferenceGrassmannBinary(sigma)

    batch = 16
    xc = torch.bernoulli(0.5 * torch.ones(batch, dim))
    # hold last 2 dims unobserved across batch for identical masks
    unobs = 2
    xc[:, -unobs:] = torch.nan

    cs_ours = ours.conditional_sigma(xc)
    cs_ref = ref.conditional_sigma(xc)

    assert torch.allclose(cs_ours, cs_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("sigma", SIGMA_CASES)
def test_mograssmann_parity_prob_mean_cov_corr_and_conditional(sigma: torch.Tensor):
    torch.manual_seed(2)
    dim = sigma.shape[0]

    # two identical components -> safe validity and easier parity
    sigmas = torch.stack([sigma, sigma], dim=0)
    mixing_p = torch.tensor([0.4, 0.6])

    ours = MoGrassmannBinary(sigmas, mixing_p)
    ref = ReferenceMoGrassmannBinary(sigmas, mixing_p)

    x = torch.bernoulli(0.5 * torch.ones(64, dim))

    p_ours = ours.prob(x)
    p_ref = ref.prob(x)
    assert torch.allclose(p_ours, p_ref, rtol=1e-5, atol=1e-6)

    assert torch.allclose(ours.mean(), ref.mean(), rtol=1e-6, atol=1e-7)
    assert torch.allclose(ours.cov(), ref.cov(), rtol=1e-5, atol=1e-6)
    assert torch.allclose(ours.corr(), ref.corr(), rtol=1e-5, atol=1e-6)

    # Conditional sigma for a chosen component (0)
    batch = 12
    xc = torch.bernoulli(0.5 * torch.ones(batch, dim))
    xc[:, -2:] = torch.nan
    cs_ours = ours.conditional_sigma(sigmas[0], xc)
    cs_ref = ref.conditional_sigma(sigmas[0], xc)
    assert torch.allclose(cs_ours, cs_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("sigma", SIGMA_CASES)
def test_grassmann_static_methods_equivalence(sigma: torch.Tensor):
    torch.manual_seed(3)
    dim = sigma.shape[0]

    gb = GrassmannBinary(sigma)
    x = torch.bernoulli(0.5 * torch.ones(32, dim))

    # prob vs static prob_grassmann
    assert torch.allclose(gb.prob(x), GrassmannBinary.prob_grassmann(x, sigma), rtol=1e-6, atol=1e-7)

    # cov vs static cov_grassmann
    assert torch.allclose(gb.cov(), GrassmannBinary.cov_grassmann(sigma), rtol=1e-6, atol=1e-7)

    # corr recompute from cov
    cov = gb.cov()
    std = torch.sqrt(torch.diag(cov))
    std_mat = torch.outer(std, std)
    corr_manual = cov / (std_mat + 1e-8)
    assert torch.allclose(gb.corr(), corr_manual, rtol=1e-6, atol=1e-7)

    # conditional_sigma_from vs method with mixed masks
    xc = torch.bernoulli(0.5 * torch.ones(10, dim))
    xc[::2, :2] = torch.nan
    xc[1::2, -2:] = torch.nan
    cs_from = GrassmannBinary.conditional_sigma_from(sigma, xc, gb._epsilon)
    cs_method = gb.conditional_sigma(xc)
    assert torch.allclose(cs_from, cs_method, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("sigma", SIGMA_CASES)
def test_grassmann_sampling_stats(sigma: torch.Tensor):
    torch.manual_seed(4)
    gb = GrassmannBinary(sigma)

    n = 3000
    samples = gb.sample(n)
    assert samples.shape == (n, gb.dim)
    assert samples.dtype == sigma.dtype
    # values are 0/1
    assert torch.all((samples == 0) | (samples == 1))
    # empirical means close to diag(sigma)
    emp_mean = samples.float().mean(0)
    assert torch.allclose(emp_mean, torch.diag(sigma).float(), atol=0.05, rtol=0.0)


@pytest.mark.parametrize("sigma", SIGMA_CASES)
def test_mograssmann_static_methods_equivalence(sigma: torch.Tensor):
    torch.manual_seed(5)
    dim = sigma.shape[0]
    sigmas = torch.stack([sigma, sigma * 0.95 + 0.05 * torch.eye(dim)], dim=0)
    mixing_p = torch.tensor([0.3, 0.7])

    mog = MoGrassmannBinary(sigmas, mixing_p)
    x = torch.bernoulli(0.5 * torch.ones(32, dim))

    # prob vs static prob_mograssmann
    assert torch.allclose(mog.prob(x), MoGrassmannBinary.prob_mograssmann(x, mixing_p, sigmas), rtol=1e-6, atol=1e-7)

    # cov vs static cov_mograssmann
    assert torch.allclose(mog.cov(), MoGrassmannBinary.cov_mograssmann(mixing_p, sigmas), rtol=1e-6, atol=1e-7)

    # corr vs static corr_mograssmann
    assert torch.allclose(mog.corr(), MoGrassmannBinary.corr_mograssmann(mixing_p, sigmas), rtol=1e-6, atol=1e-7)

    # conditional via helper parity (component 1)
    batch = 10
    xc = torch.bernoulli(0.5 * torch.ones(batch, dim))
    # ensure equal number of unobserved dims across rows, but different positions
    xc[::2, :2] = torch.nan
    xc[1::2, -2:] = torch.nan
    cs1 = mog.conditional_sigma(sigmas[1], xc)
    cs1_ref = GrassmannBinary.conditional_sigma_from(sigmas[1], xc, mog._epsilon)
    assert torch.allclose(cs1, cs1_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("sigma", SIGMA_CASES)
def test_mograssmann_sampling_stats(sigma: torch.Tensor):
    torch.manual_seed(6)
    dim = sigma.shape[0]
    sigmas = torch.stack([sigma, sigma], dim=0)
    mixing_p = torch.tensor([0.4, 0.6])

    mog = MoGrassmannBinary(sigmas, mixing_p)

    n = 4000
    samples = mog.sample(n)
    assert samples.shape == (n, dim)
    assert torch.all((samples == 0) | (samples == 1))
    emp_mean = samples.float().mean(0)
    theo_mean = mog.mean().float()
    assert torch.allclose(emp_mean, theo_mean, atol=0.05, rtol=0.0)
