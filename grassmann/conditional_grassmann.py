from typing import Tuple

import torch
from torch import Tensor, nn

from grassmann.fit_grassmann import compute_sigma_from_BC
from grassmann.GrassmannDistribution import GrassmannBinary, MoGrassmannBinary

"""
conditional version of a MoGr distribtuion 
using https://github.com/mackelab/pyknos/blob/main/pyknos/mdn/mdn.py as a template
"""


class GrassmannConditional(nn.Module):
    """
    Conditional density for Grassmann distribution
    """

    def __init__(
        self,
        features: int,
        hidden_features: int,
        hidden_net: nn.Module,
        num_components=1,
        custom_initialization=False,
        embedding_net=None,
    ):
        """Conditional Grassmann with possibly multiple components
        Args:
            features: Dimension of output density.
            hidden_features: Dimension of final layer of `hidden_net`.
            hidden_net: A Module which outputs final hidden representation before
                paramterization layers (i.e sigma, mixing coefficient).
            num_components: Number of Grassmann components.
            custom_initialization: XXX not yet implemented
            embedding_net: not yet implemented
        """

        super().__init__()

        self._features = features
        self._hidden_features = hidden_features
        self._num_components = num_components

        # Modules
        self._hidden_net = hidden_net

        self._logits_layer = nn.Linear(hidden_features, num_components)  # unnormalized mixing coefficients

        self._BC_layer = nn.Linear(
            hidden_features, num_components * 2 * features**2
        )  # parameterization layer for sigma

        # XXX docstring text
        # embedding_net: NOT IMPLEMENTED
        #         A `nn.Module` which has trainable parameters to encode the
        #         context (conditioning). It is trained jointly with the Grassmann.
        if embedding_net is not None:
            raise NotImplementedError

        # Constant for numerical stability.
        self._epsilon = 1e-4  # 1e-2

        # Initialize mixture coefficients and precision factors sensibly.
        if custom_initialization:
            self._initialize()

    def get_grassmann_params(self, context: Tensor) -> Tuple[Tensor, Tensor]:
        """Return mixing coefficients and sigma
        Args:
            context: Input to the MDN, leading dimension is batch dimension.
        Returns:
            mixing_p: (batch, num_components)
            sigma: (batch, num_components, features, features)
        """

        h = self._hidden_net(context)

        # Logits and B,C are unconstrained and are obtained directly from the
        # output of a linear layer.
        logits = self._logits_layer(h)
        # apply softmax to get normalized mixing coeffiecients
        mixing_p = torch.softmax(logits, 1)

        BC = self._BC_layer(h).view(-1, self._num_components, 2, self._features, self._features)

        sigma = self.compute_sigma(BC[:, :, 0, :, :], BC[:, :, 1, :, :])

        return mixing_p, sigma

    def get_mean(self, context: Tensor) -> Tensor:
        """
        computes the means for the given context
        """
        mixing_p, sigma = self.get_grassmann_params(context)
        return torch.sum(torch.diagonal(sigma, dim1=-1, dim2=-2) * mixing_p.unsqueeze(-1), -2)

    def cov(self, context: Tensor) -> Tensor:
        """
        computes the cov for the given context
        returns:
            cov (batch,dim,dim)
        """
        # get sigmas
        mixing_p, sigma = self.get_grassmann_params(context)
        return MoGrassmannBinary.cov_mograssmann(mixing_p, sigma)

    def corr(self, context: Tensor) -> Tensor:
        """
        computes the corr for the given context
        returns:
            corr (batch,dim,dim)
        """
        mixing_p, sigma = self.get_grassmann_params(context)
        return MoGrassmannBinary.corr_mograssmann(mixing_p, sigma)

    def compute_sigma(self, B: Tensor, C: Tensor) -> Tensor:
        """
        Compute sigma from unconstrained B and C using shared projection.
        B, C: (batch, num_components, dim, dim)
        Returns: sigma (batch, num_components, dim, dim)
        """
        batch, n_comp, dim, _ = B.shape
        B_flat = B.reshape(batch * n_comp, dim, dim)
        C_flat = C.reshape(batch * n_comp, dim, dim)
        sigma_flat = compute_sigma_from_BC(B_flat, C_flat, epsilon=self._epsilon)
        return sigma_flat.reshape(batch, n_comp, dim, dim)

    def prob(self, inputs: Tensor, context: Tensor) -> Tensor:
        """Return MoGrass(inputs|context) where MoG is a mixture of Grassmann density.
        The MoGrass's parameters (mixture coefficients, Sigma) are the
        outputs of a neural network.
        Args:
            inputs: Input variable, leading dim interpreted as batch dimension.
            context: Conditioning variable, leading dim interpreted as batch dimension.
        Returns:
            probability of inputs given context under a MoG model. (NOT in log space)
        """
        mixing_p, sigmas = self.get_grassmann_params(context)
        return self.prob_mograssmann(inputs, mixing_p, sigmas)

    def forward(self, inputs: Tensor, context: Tensor) -> Tensor:
        """alias for self.prob
        ---
        Return MoGrass(inputs|context) where MoG is a mixture of Grassmann density.
        The MoGrass's parameters (mixture coefficients, Sigma) are the
        outputs of a neural network.
        Args:
            inputs: Input variable, leading dim interpreted as batch dimension.
            context: Conditioning variable, leading dim interpreted as batch dimension.
        Returns:
            probability of inputs given context under a MoG model. (NOT in log space)
        """
        return self.prob(inputs, context)

    @staticmethod
    def prob_mograssmann(inputs: Tensor, mixing_p: Tensor, sigmas: Tensor) -> Tensor:
        """Vectorized probability via base batched implementation."""
        return MoGrassmannBinary.prob_mograssmann(inputs, mixing_p, sigmas)

    def sample(self, num_samples: int, context: Tensor) -> Tensor:
        """
        Return num_samples independent samples from MoGrass( . | context).
        Generates num_samples samples for EACH item in context batch i.e. returns
        (num_samples * batch_size) samples in total.
        Args:
            num_samples: Number of samples to generate.
            context: Conditioning variable, leading dimension is batch dimension.
                only for batch_dim = 1 implemented.
        Returns:
            Generated samples: (num_samples, output_dim) with leading batch dimension.
        """

        # only one context at a time is implemente!
        assert context.shape[0] == 1

        # Get necessary quantities.
        mixing_p, sigmas = self.get_grassmann_params(context)
        return self.sample_mograssmann(num_samples, mixing_p.squeeze(0), sigmas.squeeze(0))

    @staticmethod
    def conditional_sigma(sigma: Tensor, xc: Tensor, epsilon: float = 1e-4) -> Tensor:
        """
        Delegates to GrassmannBinary.conditional_sigma_from for parity and vectorization.
        """
        return GrassmannBinary.conditional_sigma_from(sigma, xc, epsilon)

    @staticmethod
    def sample_mograssmann(num_samples: int, mixing_p: Tensor, sigma: Tensor) -> Tensor:
        """Reuse base mixture sampling."""
        return MoGrassmannBinary(sigma.detach(), mixing_p.detach()).sample(num_samples)

    def _initialize(self) -> None:
        """
        Initialize MDN so that mixture coefficients are approximately uniform,
        and covariances are approximately the identity.
        """

        raise NotImplementedError

        # Initialize mixture coefficients to near uniform.
        self._logits_layer.weight.data = self._epsilon * torch.randn(self._num_components, self._hidden_features)
        self._logits_layer.bias.data = self._epsilon * torch.randn(self._num_components)

        # Initialize diagonal of precision factors to inverse of softplus at 1.
        self._unconstrained_diagonal_layer.weight.data = self._epsilon * torch.randn(
            self._num_components * self._features, self._hidden_features
        )
        self._unconstrained_diagonal_layer.bias.data = torch.log(
            torch.exp(torch.tensor([1 - self._epsilon])) - 1
        ) * torch.ones(self._num_components * self._features) + self._epsilon * torch.randn(
            self._num_components * self._features
        )

        # Initialize off-diagonal of precision factors to zero.
        self._upper_layer.weight.data = self._epsilon * torch.randn(
            self._num_components * self._num_upper_params, self._hidden_features
        )
        self._upper_layer.bias.data = self._epsilon * torch.randn(self._num_components * self._num_upper_params)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_fc_layers: int = 3,
        num_hiddens: int = 128,
    ):
        """multi-layer NN
        Args:
            input_dim: Dimensionality of input
            num_layers: Number of layers of the network.
            output_dim: output dim, should correspond to hidden_features
        """
        super().__init__()
        self.output_dim = output_dim
        self.num_hiddens = num_hiddens

        # construct fully connected layers
        fc_layers = [nn.Linear(input_dim, num_hiddens), nn.ReLU()]
        for _ in range(num_fc_layers - 1):
            fc_layers.append(nn.Linear(num_hiddens, num_hiddens))
            fc_layers.append(nn.ReLU())

        self.fc_subnet = nn.Sequential(*fc_layers)

        self.final_layer = nn.Linear(num_hiddens, output_dim)

    def forward(self, x):
        """Network forward pass.
        Args:
            x: Input tensor (batch_size, input_dim)
        Returns:
            Network output (batch_size, output_dim).
        """

        embedding = self.fc_subnet(x)

        out = self.final_layer(embedding)

        return out
