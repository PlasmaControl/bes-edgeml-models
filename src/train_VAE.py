import torch
import logging

logger = logging.getLogger(__name__)


class ELBOLoss:
    """
    Implementation of ELBO loss for variational autoencoder.

    :param beta: Disentanglement parameter.
    :param reduction: specifies the reduction to apply to the output (under construction)
    """

    def __init__(self, beta=1.0, reduction=None):
        self.beta = beta
        self.reduction = reduction

    def __call__(self, data_in, reconstruction, sample, mu, logvar, logscale):
        # Likelihood
        recon_loss = self.gaussian_likelihood(reconstruction, data_in, logscale)

        # kl
        std = torch.exp(logvar / 2)
        kl = self.kl_divergence(sample, mu, std)

        # elbo
        elbo = (self.beta * kl - recon_loss)

        if self.reduction == 'mean':
            elbo = elbo.mean()

        return elbo, kl, recon_loss

    def kl_divergence(self, sample, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)

        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(sample)
        log_pz = p.log_prob(sample)

        # kl
        kl = (log_qzx - log_pz)

        # sum over last dim to go from single dim distribution to multi-dim
        kl = kl.sum(-1)

        return kl

    def gaussian_likelihood(self, x_hat, x, logscale):
        # learned external parameter
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)

        return log_pxz.sum(dim=(1, 2, 3, 4))
