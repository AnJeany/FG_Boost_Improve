"""============================================================================
Gaussian process regression using random Fourier features. Based on "Random
Features for Large-Scale Kernel Machines" by Rahimi and Recht (2007).

For more, see the accompanying blog post:
http://gregorygundersen.com/blog/2019/12/23/random-fourier-features/
code from : https://github.com/gwgundersen/random-fourier-features/blob/master/rffgpr.py
============================================================================"""

import torch

# ------------------------------------------------------------------------------

class RFFGaussianProcess:
    def __init__(self, rff_dim=10, sigma=1.0):
        """Gaussian process regression using random Fourier features.

        rff_dim : Dimension of random feature.
        sigma :   sigma^2 is the variance.
        """
        self.rff_dim = rff_dim
        self.sigma   = sigma
        self.alpha_  = None
        self.b_      = None
        self.W_      = None

    def _get_rffs(self, X, return_vars=False):
        """Return random Fourier features based on data X, as well as random
        variables W and b.
        """
        N, D = X.shape
        if self.W_ is not None:
            W, b = self.W_, self.b_
        else:
            W = torch.normal(mean=0, std=1, size=(self.rff_dim, D))
            b = torch.rand(self.rff_dim) * 2 * torch.pi

        B = b.unsqueeze(1).repeat(1, N)
        norm = 1. / torch.sqrt(torch.tensor(self.rff_dim, dtype=torch.float32))
        Z = norm * torch.sqrt(torch.tensor(2.0)) * torch.cos(self.sigma * W @ X.T.float() + B)
        if return_vars:
            return Z, W, b
        return Z

    def _get_rvs(self, D):
        """On first call, return random variables W and b. Else, return cached
        values.
        """
        if self.W_ is not None:
            return self.W_, self.b_
        W = torch.normal(mean=0, std=1, size=(self.rff_dim, D))
        b = torch.rand(self.rff_dim) * 2 * torch.pi
        return W, b

