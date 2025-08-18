import torch

from grassmann.fit_grassmann import (
    EstimateGrassmann,
    EstimateMoGrassmann,
    compute_sigma_from_BC,
    train_EstimateGrassmann,
)
from grassmann.GrassmannDistribution import GrassmannBinary, MoGrassmannBinary
from grassmann.utils import ARAI_REFERENCE_SIGMA


def test_compute_sigma_from_BC_shapes_and_validity():
    dim = ARAI_REFERENCE_SIGMA.shape[0]
    B = torch.randn(dim, dim)
    C = torch.randn(dim, dim)
    sigma = compute_sigma_from_BC(B, C)
    assert sigma.shape == (dim, dim)

    B_b = torch.randn(3, dim, dim)
    C_b = torch.randn(3, dim, dim)
    sigma_b = compute_sigma_from_BC(B_b, C_b)
    assert sigma_b.shape == (3, dim, dim)


def test_estimate_grassmann_forward_and_parity_prob():
    torch.manual_seed(0)
    dim = ARAI_REFERENCE_SIGMA.shape[0]
    est = EstimateGrassmann(dim)
    # make sigma stable
    est.sigma = compute_sigma_from_BC(est.B, est.C)

    x = torch.bernoulli(0.5 * torch.ones(64, dim))
    # forward returns mean logprob
    lp = est(x)
    assert torch.isfinite(lp)

    # parity with distribution prob
    p_est = est.prob_grassmann(x, est.sigma)
    p_ref = GrassmannBinary.prob_grassmann(x, est.sigma)
    assert torch.allclose(p_est, p_ref, rtol=1e-6, atol=1e-7)


def test_estimate_mograssmann_forward_and_parity_prob():
    torch.manual_seed(1)
    dim = ARAI_REFERENCE_SIGMA.shape[0]
    nc = 2
    est = EstimateMoGrassmann(dim, nc)
    est.sigma = compute_sigma_from_BC(est.B, est.C)
    est.p_mixing = torch.tensor([0.4, 0.6])

    x = torch.bernoulli(0.5 * torch.ones(64, dim))
    lp = est(x)
    assert torch.isfinite(lp)

    p_est = est.prob_mograssmann(x, est.sigma, est.p_mixing)
    p_ref = MoGrassmannBinary.prob_mograssmann(x, est.p_mixing, est.sigma)
    assert torch.allclose(p_est, p_ref, rtol=1e-6, atol=1e-7)


def test_train_estimate_grassmann_decreases_loss():
    torch.manual_seed(2)
    dim = ARAI_REFERENCE_SIGMA.shape[0]
    gb = GrassmannBinary(ARAI_REFERENCE_SIGMA)
    samples = gb.sample(1000)

    model = EstimateGrassmann(dim)
    history = train_EstimateGrassmann(model, samples, steps=100, verbose=False, batch_size=256, early_stop=False)
    # check last window lower than first window
    start = history[:50].mean()
    end = history[-50:].mean()
    assert end < start


def test_estimate_grassmann_learns_reference_sigma_well_enough():
    torch.manual_seed(3)
    dim = ARAI_REFERENCE_SIGMA.shape[0]
    ref = GrassmannBinary(ARAI_REFERENCE_SIGMA)
    samples = ref.sample(5000)

    # Initialize model and compute initial sigma, mean, cov
    model = EstimateGrassmann(dim)
    sigma_init = compute_sigma_from_BC(model.B, model.C)
    mean_ref = torch.diagonal(ARAI_REFERENCE_SIGMA)
    cov_ref = GrassmannBinary.cov_grassmann(ARAI_REFERENCE_SIGMA)
    mean_init = torch.diagonal(sigma_init)
    cov_init = GrassmannBinary.cov_grassmann(sigma_init)

    diag_rmse_init = torch.sqrt(torch.mean((mean_init - mean_ref) ** 2))
    cov_fro_init = torch.linalg.norm(cov_init - cov_ref)
    cov_fro_ref = torch.linalg.norm(cov_ref) + 1e-8

    # Train
    train_EstimateGrassmann(model, samples, steps=1000, verbose=False, batch_size=512, early_stop=False)

    # Evaluate post-training
    sigma_trained = compute_sigma_from_BC(model.B, model.C)
    mean_tr = torch.diagonal(sigma_trained)
    cov_tr = GrassmannBinary.cov_grassmann(sigma_trained)

    diag_rmse_tr = torch.sqrt(torch.mean((mean_tr - mean_ref) ** 2))
    cov_fro_tr = torch.linalg.norm(cov_tr - cov_ref)

    # Must improve significantly vs init
    assert diag_rmse_tr < diag_rmse_init * 0.7, f"diag_rmse_tr: {diag_rmse_tr}, diag_rmse_init: {diag_rmse_init}"
    assert cov_fro_tr < cov_fro_init * 0.7, f"cov_fro_tr: {cov_fro_tr}, cov_fro_init: {cov_fro_init}"

    # And meet loose absolute thresholds
    assert diag_rmse_tr < 0.2, f"diag_rmse_tr: {diag_rmse_tr}"
    assert (cov_fro_tr / cov_fro_ref) < 0.8, f"cov_fro_tr: {cov_fro_tr}, cov_fro_ref: {cov_fro_ref}"
