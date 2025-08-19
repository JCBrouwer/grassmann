import pytest
import torch

from grassmann.conditional_grassmann import MLP, GrassmannConditional
from grassmann.fit_grassmann import compute_sigma_from_BC
from grassmann.GrassmannDistribution import GrassmannBinary, MoGrassmannBinary
from grassmann.utils import ARAI_REFERENCE_SIGMA


def build_dummy_hidden(features: int, hidden_features: int):
    # simple linear network
    return MLP(input_dim=7, output_dim=hidden_features, num_fc_layers=2, num_hiddens=16)


@pytest.mark.parametrize("batch_size,num_components", [(3, 1), (2, 2)])
def test_get_grassmann_params_shapes(batch_size: int, num_components: int):
    torch.manual_seed(0)
    features = ARAI_REFERENCE_SIGMA.shape[0]
    hidden_features = 8
    hidden = build_dummy_hidden(features, hidden_features)
    model = GrassmannConditional(features, hidden_features, hidden, num_components=num_components)

    context = torch.randn(batch_size, 7)
    mixing_p, sigma = model.get_grassmann_params(context)
    assert mixing_p.shape == (batch_size, num_components)
    assert sigma.shape == (batch_size, num_components, features, features)


def test_prob_parity_with_base_mixture():
    torch.manual_seed(1)
    features = ARAI_REFERENCE_SIGMA.shape[0]
    hidden_features = 8
    hidden = build_dummy_hidden(features, hidden_features)
    model = GrassmannConditional(features, hidden_features, hidden, num_components=2)

    batch = 5
    context = torch.randn(batch, 7)
    mixing_p, sigma = model.get_grassmann_params(context)

    # Build base mixture outputs per batch item and compare
    x = torch.bernoulli(0.5 * torch.ones(batch, features))
    p_cond = model.prob(x, context)
    p_base = torch.empty(batch)
    for b in range(batch):
        p_base[b] = MoGrassmannBinary.prob_mograssmann(x[b : b + 1], mixing_p[b], sigma[b]).squeeze(0)
    assert torch.allclose(p_cond, p_base, rtol=1e-6, atol=1e-7)


def test_cov_corr_parity():
    torch.manual_seed(2)
    features = ARAI_REFERENCE_SIGMA.shape[0]
    hidden_features = 8
    hidden = build_dummy_hidden(features, hidden_features)
    model = GrassmannConditional(features, hidden_features, hidden, num_components=2)

    context = torch.randn(4, 7)
    mixing_p, sigma = model.get_grassmann_params(context)

    # Base cov/corr per batch item
    cov_base = torch.stack([MoGrassmannBinary.cov_mograssmann(mixing_p[b], sigma[b]) for b in range(context.shape[0])])
    corr_base = torch.stack([
        MoGrassmannBinary.corr_mograssmann(mixing_p[b], sigma[b]) for b in range(context.shape[0])
    ])

    cov = model.cov(context)
    corr = model.corr(context)
    assert torch.allclose(cov, cov_base, rtol=1e-6, atol=1e-7)
    assert torch.allclose(corr, corr_base, rtol=1e-6, atol=1e-7)


def test_compute_sigma_matches_shared_projection():
    torch.manual_seed(3)
    features = ARAI_REFERENCE_SIGMA.shape[0]
    hidden_features = 8
    hidden = build_dummy_hidden(features, hidden_features)
    model = GrassmannConditional(features, hidden_features, hidden, num_components=3)

    batch = 2
    B = torch.randn(batch, 3, features, features)
    C = torch.randn(batch, 3, features, features)
    sigma_model = model.compute_sigma(B, C)
    sigma_ref = compute_sigma_from_BC(B.reshape(-1, features, features), C.reshape(-1, features, features)).reshape(
        batch, 3, features, features
    )
    assert torch.allclose(sigma_model, sigma_ref, rtol=1e-6, atol=1e-7)


def test_sample_shapes_and_values():
    torch.manual_seed(4)
    features = ARAI_REFERENCE_SIGMA.shape[0]
    hidden_features = 8
    hidden = build_dummy_hidden(features, hidden_features)
    model = GrassmannConditional(features, hidden_features, hidden, num_components=2)

    context = torch.randn(1, 7)
    samples = model.sample(500, context)
    assert samples.shape == (500, features)
    assert torch.all((samples == 0) | (samples == 1))


def test_conditional_sigma_delegation():
    torch.manual_seed(5)
    dim = ARAI_REFERENCE_SIGMA.shape[0]
    sigma = ARAI_REFERENCE_SIGMA
    batch = 6
    xc = torch.bernoulli(0.5 * torch.ones(batch, dim))
    # same number of NaNs across rows for helper precondition
    xc[:, :2] = torch.nan
    cs1 = GrassmannConditional.conditional_sigma(sigma, xc)
    cs2 = GrassmannBinary.conditional_sigma_from(sigma, xc, 1e-4)
    assert torch.allclose(cs1, cs2, rtol=1e-6, atol=1e-7)
