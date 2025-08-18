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
    ) -> Tensor:  # n x d  # (d x d)
        """
        Return the probability of `x` under a GrassmannBinary with specified parameters.
        As standalone method.
        Args:
            x: Location at which to evaluate the Grassmann, aka binary vector.
            sigma:
        Returns:
            probabilities of each input.
        """
        assert len(x.shape) == 2  # check dim: batch, d

        batch_size = x.shape[0]
        dim = sigma.shape[0]

        m = torch.zeros((batch_size, dim, dim))

        # vectorized version
        m = sigma.repeat(batch_size, 1, 1) * ((-1) ** (1 - x)).repeat(1, dim).view(batch_size, dim, dim)
        m = m * (1 - torch.eye(dim, dim).repeat(batch_size, 1, 1))  # replace diag with 0
        m = m + (
            torch.eye(dim).repeat(batch_size, 1, 1)
            * (torch.diag(sigma).repeat(batch_size, 1) ** x).repeat(1, dim).view(batch_size, dim, dim)
            * (torch.diag(1 - sigma).repeat(batch_size, 1) ** (1 - x)).repeat(1, dim).view(batch_size, dim, dim)
        )

        p = torch.det(m)

        return p

    def mean(self):
        """
        returns the expected value based on self.sigma
        """
        return torch.diagonal(self.sigma)

    @staticmethod
    def cov_grassmann(sigma):
        """
        calculates the cov a a grassmann distribution
        Args:
            sigma (Tensor): parameter of gr

        Returns:
            Tensor: cov
        """
        # off-diagonal: -sigma_ij * sigma_ji
        cov = -(sigma * sigma.T)
        # diagonal: p * (1 - p), where p = diag(sigma)
        diag = torch.diagonal(sigma)
        cov.diagonal().copy_(diag * (1 - diag))
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
        Unlike the `prob()` method, this method is fully detached from the neural
        network and can be used independent of the neural net in case the MoGrassmann
        parameters are already known.
        Args:
            inputs: 01-tensors at which to evaluate the MoGrassmann. (batch_size, parameter_dim)
            mixing_p: weights of each component of the MoGrassmann. Shape: (num_components).
            sigmas: Parameters of each MoGrassmann, shape (num_components, parameter_dim, parameter_dim).
        Returns:
            probabilities of each input.
        """
        assert len(inputs.shape) == 2  # check dim: batch, dim
        assert sigmas.shape[0] == mixing_p.shape[0]  # check: n_components

        num_components = mixing_p.shape[0]

        # Reuse single-component probability and sum across components
        per_comp = [GrassmannBinary.prob_grassmann(inputs, sigmas[k]) for k in range(num_components)]
        probs = torch.stack(per_comp, dim=-1)  # (batch_size, num_components)
        return probs @ mixing_p

    def mean(self):
        """
        returns the expected value based on self.sigma
        """
        return torch.sum(torch.diagonal(self.sigma, dim1=-1, dim2=-2) * self.mixing_p.unsqueeze(-1), -2)

    @staticmethod
    def cov_mograssmann(mixing_p, sigma) -> Tensor:
        """
        computes the cov for the given mixing coefficients and sigma
        as standalone
        returns:
            mixing_p (n_components)
            cov (dim,dim)
        """
        # get dims
        n_comp = mixing_p.shape[-1]

        assert sigma.shape[0] == n_comp
        # check if mixing coefficients sum up to 1
        assert torch.isclose(torch.sum(mixing_p), torch.ones(1), atol=1e-4)

        # Means per component
        means = torch.diagonal(sigma, dim1=-1, dim2=-2)  # (n_comp, dim)

        # Reuse GrassmannBinary.cov_grassmann per component
        cov_per_comp = torch.stack(
            [GrassmannBinary.cov_grassmann(sigma[k]) for k in range(n_comp)], dim=0
        )  # (n_comp, dim, dim)

        # Weighted within-component covariance
        cov_within = torch.sum(cov_per_comp * mixing_p.unsqueeze(-1).unsqueeze(-1), dim=0)

        # Mixture mean and between-component covariance
        mixture_mean = torch.sum(means * mixing_p.unsqueeze(-1), dim=0)  # (dim)
        diffs = means - mixture_mean  # (n_comp, dim)
        cov_of_means_per_comp = torch.einsum("ni,nj->nij", diffs, diffs)
        cov_between = torch.sum(cov_of_means_per_comp * mixing_p.unsqueeze(-1).unsqueeze(-1), dim=0)

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
        computes the corr
        inputs:
            mixing_p (batch,n_components)
            sigma (num_components, dim, dim)
        returns:
            cov (batch,dim,dim)
        """
        # compute cov, including all components
        cov = MoGrassmannBinary.cov_mograssmann(mixing_p, sigma)
        std = torch.sqrt(torch.diagonal(cov, dim1=-1, dim2=-2))
        std_mat = torch.outer(std, std)
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
