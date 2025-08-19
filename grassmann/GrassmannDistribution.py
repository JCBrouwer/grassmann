import numpy as np
import torch
from torch import Tensor

from grassmann.utils import check_valid_sigma


class GrassmannBinary:
    """
    Implementation of a multivariate binary probability distribution in the Grassmann formalism
    (Arai, Takashi. Multivariate binary probability distribution in the Grassmann formalism. PhysRevE. 2021.)
    """

    def __init__(
        self,
        sigma: Tensor,
        lambd: Tensor | None = None,
    ):
        assert len(sigma.shape) == 2
        assert check_valid_sigma(sigma)
        self.dim = sigma.shape[0]
        self.sigma = sigma
        if lambd is None:
            self.lambd = torch.inverse(sigma)  # this is not used at the moment!
        self._epsilon = 1e-4  # small value to avoid singular matrices

    def prob(self, x):
        # Todo: make more efficient by only computing once per oberved states?
        return self.prob_grassmann(x, self.sigma)

    @staticmethod
    def prob_grassmann(
        x: Tensor,
        sigma: Tensor,
    ) -> Tensor:
        """
        Return the probability of `x` under a GrassmannBinary with specified parameters.
        This implementation supports an arbitrary number of batch dimensions.

        Args:
            x: Binary inputs with shape (..., d).
            sigma: Grassmann parameter with shape (..., d, d) or (d, d), broadcastable to x.
        Returns:
            probabilities with shape broadcast(x.shape[:-1], sigma.shape[:-2]).
        """
        # Validate last dimensions
        assert x.shape[-1] == sigma.shape[-1] == sigma.shape[-2]

        dim = sigma.shape[-1]

        # Off-diagonal part: multiply each column j by (-1)^(1 - x_j) which is -1 if x_j=0 else 1
        # Use a stable equivalent: sign = 2*x - 1  in { -1, 1 }
        sign = 2 * x - 1  # (..., d)
        off_diag = sigma * sign[..., None, :]  # (..., d, d)

        eye = torch.eye(dim, dtype=sigma.dtype, device=sigma.device)
        off_diag = off_diag * (1 - eye)  # zero diagonal

        # Diagonal part: sigma_ii^x_i * (1 - sigma_ii)^(1 - x_i) reduces to x*sigma_ii + (1-x)*(1-sigma_ii)
        diag_sigma = torch.diagonal(sigma, dim1=-2, dim2=-1)  # (..., d)
        diag_val = x * diag_sigma + (1 - x) * (1 - diag_sigma)  # (..., d)

        m = off_diag + torch.diag_embed(diag_val)  # (..., d, d)

        return torch.linalg.det(m)  # (...)

    def mean(self):
        """
        returns the expected value based on self.sigma
        """
        return torch.diagonal(self.sigma)

    @staticmethod
    def cov_grassmann(sigma):
        """
        calculates the covariance for a Grassmann distribution.
        Args:
            sigma (Tensor): Grassmann parameter with shape (..., d, d)

        Returns:
            Tensor: covariance with shape (..., d, d)
        """
        # off-diagonal: -sigma_ij * sigma_ji
        cov = -(sigma * sigma.transpose(-1, -2))
        # diagonal: p * (1 - p), where p = diag(sigma)
        diag = torch.diagonal(sigma, dim1=-2, dim2=-1)
        cov.diagonal(dim1=-2, dim2=-1).copy_(diag * (1 - diag))
        return cov

    def cov(self):
        """
        returns covariance matrix based on self.sigma
        """
        return GrassmannBinary.cov_grassmann(self.sigma)

    def corr(self):
        """
        returns corr matrix based on self.sigma
        """
        cov = self.cov()
        std = torch.sqrt(torch.diag(cov))
        std_mat = torch.outer(std, std)
        return cov / (std_mat + 1e-8)

    @staticmethod
    def conditional_sigma_from(sigma: Tensor, xc: Tensor, epsilon: float) -> Tensor:
        """
        Vectorized conditional sigma computation given a sigma and batch of partially observed xc.
        xc contains NaNs for unobserved entries.
        Returns a batch of conditional sigmas for the remaining (unobserved) dimensions.
        """
        batch_size, _ = xc.shape
        dim = sigma.shape[0]

        dim_r_per_row = torch.isnan(xc).sum(1)
        assert torch.all(torch.eq(dim_r_per_row, dim_r_per_row[0]))
        dim_r = int(dim_r_per_row[0].item())
        dim_c = dim - dim_r

        device = sigma.device
        dtype = sigma.dtype

        sigma_r = torch.empty((batch_size, dim_r, dim_r), dtype=dtype, device=device)
        masks = ~torch.isnan(xc)

        # Fast path: identical masks across batch
        if torch.all(masks[0] == masks):
            mask = masks[0]
            remaining_mask = ~mask

            if dim_c == 0:
                sigma_rr = sigma[remaining_mask][:, remaining_mask]
                sigma_r[:] = sigma_rr
                return sigma_r

            sigma_rr = sigma[remaining_mask][:, remaining_mask]
            sigma_rc = sigma[remaining_mask][:, mask]
            sigma_cr = sigma[mask][:, remaining_mask]
            sigma_cc = sigma[mask][:, mask]

            x_c = xc[:, mask]
            eye_c = torch.eye(dim_c, dtype=dtype, device=device)
            A = sigma_cc.expand(batch_size, dim_c, dim_c).clone()
            A = A - torch.diag_embed(1 - x_c) + eye_c * epsilon
            B = sigma_cr.expand(batch_size, dim_c, dim_r)
            X = torch.linalg.solve(A, B)
            sigma_r[:] = sigma_rr - torch.matmul(sigma_rc, X)
            return sigma_r

        # General path: group by unique masks
        unique_masks, inverse_indices = torch.unique(masks, dim=0, return_inverse=True)
        for group_index in range(unique_masks.shape[0]):
            row_selector = inverse_indices == group_index
            if not torch.any(row_selector):
                continue

            mask = unique_masks[group_index]
            remaining_mask = ~mask

            if dim_c == 0:
                sigma_rr = sigma[remaining_mask][:, remaining_mask]
                sigma_r[row_selector] = sigma_rr
                continue

            sigma_rr = sigma[remaining_mask][:, remaining_mask]
            sigma_rc = sigma[remaining_mask][:, mask]
            sigma_cr = sigma[mask][:, remaining_mask]
            sigma_cc = sigma[mask][:, mask]

            x_c = xc[row_selector][:, mask]
            n_g = x_c.shape[0]
            eye_c = torch.eye(dim_c, dtype=dtype, device=device)
            A = sigma_cc.expand(n_g, dim_c, dim_c).clone()
            A = A - torch.diag_embed(1 - x_c) + eye_c * epsilon
            B = sigma_cr.expand(n_g, dim_c, dim_r)
            X = torch.linalg.solve(A, B)
            sigma_r[row_selector] = sigma_rr - torch.matmul(sigma_rc, X)

        return sigma_r

    def conditional_sigma(self, xc: Tensor) -> Tensor:
        """
        returns the conditional grassmann matrix for the remaining dimensions, given xc
            xc: Tensor of full dim, with "nan" in remaining positions. (batch_size x d)
        """
        return GrassmannBinary.conditional_sigma_from(self.sigma, xc, self._epsilon)

    def sample(
        self,
        num_samples: int,
    ) -> Tensor:
        """
        Return samples of a GrassmannBinary with specified parameters.
        Args:
            num_samples: Number of samples to generate.
        Returns:
            Tensor: Samples from the GrassmannBinary.
        """

        samples = torch.zeros((num_samples, self.dim)) * torch.nan

        # sample first dim. simple bernoulli from sigma_00
        samples[:, 0] = torch.bernoulli(self.sigma[0, 0].expand(num_samples))

        for i in range(1, self.dim):
            sigma_c = self.conditional_sigma(samples)
            samples[:, i] = torch.bernoulli(sigma_c[:, 0, 0])

        return samples


"""
Mixture of Grassmann 
"""


class MoGrassmannBinary:
    """
    Mixture of GrassmannBinary
    """

    def __init__(self, sigma: Tensor, mixing_p: Tensor):
        """
        Args:
            sigma (Tensor): (nc,dim,dim) parameters for mogr
            mixing_p (Tensor): mixing coefficients for mogr, should sum up to one
        """
        assert len(sigma.shape) == 3
        for i in range(sigma.shape[0]):
            assert check_valid_sigma(sigma[i])
        assert sigma.shape[0] == mixing_p.shape[0]
        self.dim = sigma.shape[1]
        self.nc = sigma.shape[0]  # number of components
        self.sigma = sigma
        self.mixing_p = mixing_p
        self._epsilon = 1e-4  # small value to avoid singular matrices

    def prob(self, x):
        """
        evaluates the probability of the mogr, based on self.sigma and self.mixing_p
        Args:
            x (Tensor): samples (batch, dim)

        Returns:
            Tensor: prbabilities (batch)
        """
        # Todo: make more efficient by only computing once per oberved states?
        return self.prob_mograssmann(x, self.mixing_p, self.sigma)

    @staticmethod
    def prob_mograssmann(
        inputs: Tensor,
        mixing_p: Tensor,
        sigmas: Tensor,
    ) -> Tensor:
        """
        Return the probability of `inputs` under a MoGrassmann with specified parameters.
        Supports arbitrary leading batch dimensions shared/broadcast across arguments.

        Args:
            inputs: 01-tensors at which to evaluate the MoGrassmann. Shape (..., d).
            mixing_p: Weights for each mixture component. Shape (..., n_components).
            sigmas: MoGrassmann parameters. Shape (..., n_components, d, d).
        Returns:
            probabilities with shape broadcast(inputs.shape[:-1], mixing_p.shape[:-1], sigmas.shape[:-3]).
        """
        # Basic shape checks on the trailing dimensions
        d = inputs.shape[-1]
        assert sigmas.shape[-1] == d and sigmas.shape[-2] == d
        assert mixing_p.shape[-1] == sigmas.shape[-3]

        # Compute per-component probabilities, broadcasting inputs across the component axis
        per_component = GrassmannBinary.prob_grassmann(inputs.unsqueeze(-2), sigmas)  # (..., K)

        # Weighted sum across components
        return torch.sum(per_component * mixing_p, dim=-1)

    def mean(self):
        """
        returns the expected value based on self.sigma
        """
        return torch.sum(torch.diagonal(self.sigma, dim1=-1, dim2=-2) * self.mixing_p.unsqueeze(-1), -2)

    @staticmethod
    def cov_mograssmann(mixing_p, sigma) -> Tensor:
        """
        Computes the covariance for a mixture for the given mixing coefficients and parameters.
        Supports arbitrary leading batch dimensions.

        Args:
            mixing_p: (..., n_components)
            sigma: (..., n_components, d, d)
        Returns:
            cov: (..., d, d)
        """
        # Basic shape checks on the trailing dimensions
        n_comp = mixing_p.shape[-1]
        assert sigma.shape[-3] == n_comp

        # Means per component (..., K, d)
        means = torch.diagonal(sigma, dim1=-1, dim2=-2)

        # Within-component covariance (..., K, d, d)
        cov_per_comp = GrassmannBinary.cov_grassmann(sigma)
        cov_within = torch.sum(cov_per_comp * mixing_p.unsqueeze(-1).unsqueeze(-1), dim=-3)

        # Between-component covariance
        mixture_mean = torch.sum(means * mixing_p.unsqueeze(-1), dim=-2)  # (..., d)
        diffs = means - mixture_mean.unsqueeze(-2)  # (..., K, d)
        weighted_diffs = diffs * mixing_p.unsqueeze(-1)  # (..., K, d)
        cov_between = torch.einsum("...kd,...ke->...de", weighted_diffs, diffs)

        return cov_within + cov_between

    def cov(self):
        """
        returns covariance matrix based on self.sigma
        """
        cov = MoGrassmannBinary.cov_mograssmann(self.mixing_p, self.sigma)

        return cov

    def corr(self):
        """
        returns corr matrix based on self.sigma
        """
        cov = self.cov()
        std = torch.sqrt(torch.diag(cov))
        std_mat = torch.outer(std, std)

        return cov / (std_mat + 1e-8)

    @staticmethod
    def corr_mograssmann(mixing_p, sigma) -> Tensor:
        """
        computes the correlation matrix, supporting arbitrary leading batch dimensions.
        inputs:
            mixing_p (..., n_components)
            sigma (..., n_components, dim, dim)
        returns:
            corr (..., dim, dim)
        """
        # compute cov, including all components
        cov = MoGrassmannBinary.cov_mograssmann(mixing_p, sigma)
        std = torch.sqrt(torch.diagonal(cov, dim1=-1, dim2=-2))  # (..., d)
        std_mat = std.unsqueeze(-1) * std.unsqueeze(-2)  # (..., d, d)
        return cov / (std_mat + 1e-8)

    def conditional_sigma(self, sigma: Tensor, xc: Tensor) -> Tensor:
        """
        returns the conditional grassmann matrix for the remaining dimensions, given xc
            xc: Tensor of full dim, with "nan" in remaining positions. (batch_size x d)
        """
        return GrassmannBinary.conditional_sigma_from(sigma, xc, self._epsilon)

    def sample(self, num_samples: int) -> Tensor:
        """
        Return samples of a moGrassmannBinary with specified parameters.
        Args:
            num_samples: Number of samples to generate.
        Returns:
            Tensor: Samples from the GrassmannBinary.
        """

        # sample how many samples from each component
        ns = torch.tensor(np.random.multinomial(num_samples, self.mixing_p))

        samples = torch.zeros((num_samples, self.dim)) * torch.nan

        count = 0
        for j, n in enumerate(ns):
            n_int = int(n.item())
            if n_int > 0:
                # sample first dim. simple bernoulli from sigma_00
                samples[count : count + n_int, 0] = torch.bernoulli(self.sigma[j][0, 0].expand(n_int))

                for i in range(1, self.dim):
                    sigma_c = self.conditional_sigma(self.sigma[j], samples[count : count + n_int])
                    samples[count : count + n_int, i] = torch.bernoulli(sigma_c[:, 0, 0])

            count += n_int

        return samples[torch.randperm(num_samples)]
