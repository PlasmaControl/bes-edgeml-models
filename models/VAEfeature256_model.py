import argparse
from typing import Tuple, Union
from collections import OrderedDict

import torch
import torch.nn as nn


class VAEfeature256Model(nn.Module):

    def __init__(self, args: argparse.Namespace, num_filters: int = 10,
                 fc_units: Union[int, Tuple[int, int]] = (40, 20), latent_dim: int = 10, beta: float = 1.0,
                 logscale=nn.Parameter(torch.Tensor([0.0])), dropout_rate: float = 0.4, negative_slope: float = 0.02,
                 maxpool_size: int = 2, ):
        """
                8x8 + time feature blocks followed by fully-connected layers. This function
                takes in a 4-dimensional tensor of size: `(1, signal_window_size, 8, 8)`
                performs maxpooling to downsample the spatial dimension by half, perform a
                3-d convolution with a filter size identical to the spatial dimensions of the
                input to avoid the sliding of the kernel over the input. Fully connected layers
                reduce to a latent space. The decoder structure is taken as the mirror image of
                the encoder.

                Args:
                -----
                    fc_units (Union[int, Tuple[int, int]], optional): Number of hidden
                        units in each layer. Defaults to (40, 20).
                    dropout_rate (float, optional): Fraction of total hidden units that will
                        be turned off for drop out. Defaults to 0.2.
                    negative_slope (float, optional): Slope of LeakyReLU activation for negative
                        `x`. Defaults to 0.02.
                    maxpool_size (int, optional): Size of the kernel used for maxpooling. Use
                        0 to skip maxpooling. Defaults to 2.
                    num_filters (int, optional): Dimensionality of the output space.
                        Essentially, it gives the number of output kernels after convolution.
                        Defaults to 10.
                    latent_dim (int, optional): Size of the latent space between ecoder and decoder.
                        Defaults to 10.
                    beta (float, optional): Strength of disentanglement for training beta-VAE
                        Defaults to 1.0
                """

        super(VAEfeature256Model, self).__init__()
        self.args = args
        self.latent_dim = latent_dim
        self.beta = beta
        self.logscale = logscale

        pool_size = [1, maxpool_size, maxpool_size]
        filter_size = (int(self.args.signal_window_size), 4, 4)

        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    ('maxpool', nn.MaxPool3d(kernel_size=pool_size)),
                    ('conv', nn.Conv3d(in_channels=1, out_channels=num_filters, kernel_size=filter_size)),
                    ('dropout3d', nn.Dropout3d(p=dropout_rate)),
                    ('relu3d', nn.LeakyReLU(negative_slope=negative_slope)),
                    ('flatten', nn.Flatten()),
                    ('fc1', nn.Linear(in_features=num_filters, out_features=fc_units[0])),
                    ('dropout1', nn.Dropout(p=dropout_rate)),
                    ('relu1', nn.LeakyReLU(negative_slope=negative_slope)),
                    ('fc2', nn.Linear(in_features=fc_units[0], out_features=fc_units[1])),
                    ('dropout2', nn.Dropout(p=dropout_rate)),
                    ('relu2', nn.LeakyReLU(negative_slope=negative_slope)),
                    ('fc3', nn.Linear(in_features=fc_units[1], out_features=self.latent_dim * 2))])

        )

        self.decoder = nn.Sequential(OrderedDict(
                [('fc1_decoder', nn.Linear(in_features=self.latent_dim, out_features=fc_units[1])),
                        ('dropout1_decoder', nn.Dropout(p=dropout_rate)),
                        ('relu1_decoder', nn.LeakyReLU(negative_slope=negative_slope)),
                        ('fc2_decoder', nn.Linear(in_features=fc_units[1], out_features=fc_units[0])),
                        ('dropout2_decoder', nn.Dropout(p=dropout_rate)),
                        ('relu2_decoder', nn.LeakyReLU(negative_slope=negative_slope)),
                        ('fc3_decoder', nn.Linear(in_features=fc_units[0], out_features=num_filters)),
                        ('dropout3_decoder', nn.Dropout(p=dropout_rate)),
                        ('relu3_decoder', nn.LeakyReLU(negative_slope=negative_slope)), (
                'tconv_decoder', nn.Linear(in_features=num_filters, out_features=self.args.signal_window_size * 8 * 8)),
                        ('sigmoid_decoder', nn.Sigmoid())]))

        self.apply(self.init_weights_)

    @staticmethod
    def init_weights_(m):
        print(m)
        if type(m) == nn.Linear:
            with torch.no_grad():
                torch.nn.init.uniform_(m.weight, -0.08, 0.08)

    def encode(self, x):
        dist = self.encoder(x).view(-1, 2, self.latent_dim)
        mu = dist[:, 0, :]
        logvar = dist[:, 1, :]
        return mu, logvar

    def parameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar / 2)  # standard deviation
            eps = torch.randn_like(std)
            sample = mu + (eps * std)  # sampling as if coming from the input space
            return sample
        else:
            return mu

    def decode(self, z):
        return self.decoder(z).view(-1, 1, self.args.signal_window_size, 8, 8)

    def forward(self, x):
        mu, logvar = self.encode(x)
        sample = self.parameterize(mu, logvar)
        reconstruction = self.decode(sample)

        return reconstruction, mu, logvar, sample
