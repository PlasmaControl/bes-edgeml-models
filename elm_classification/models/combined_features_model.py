import argparse
from typing import Tuple, Union

import numpy as np
import pywt
from scipy.fft import rfft
import torch
import torch.nn as nn


class CombinedFeatureModel(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class RawFeatureModel(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        dropout_rate: float = 0.4,
        negative_slope: float = 0.02,
        maxpool_size: int = 2,
        num_filters: int = 10,
    ):
        """
        Use the raw BES channels values as features. 8x8 + time feature blocks
        followed by fully-connected layers. This function takes in a 5-dimensional
        tensor of size: `(N, 1, signal_window_size, 8, 8)`, N=batch_size and
        performs maxpooling to downsample the spatial dimension by half, perform a
        3-d convolution with a filter size identical to the spatial dimensions of the
        input to avoid the sliding of the kernel over the input. In the end, we will
        have a feature map that can be concatenated with other features.

        Args:
        -----
            args (argparse.Namespace): Command line arguments containing the information
                about signal_window.
            dropout_rate (float, optional): Fraction of total hidden units that will
                be turned off for drop out. Defaults to 0.2.
            negative_slope (float, optional): Slope of LeakyReLU activation for negative
                `x`. Defaults to 0.02.
            maxpool_size (int, optional): Size of the kernel used for maxpooling. Use
                0 to skip maxpooling. Defaults to 2.
            num_filters (int, optional): Dimensionality of the output space.
                Essentially, it gives the number of output kernels after convolution.
                Defaults to 10.
        """
        super(RawFeatureModel, self).__init__()
        pool_size = [1, maxpool_size, maxpool_size]
        self.args = args
        filter_size = (self.args.signal_window_size, 4, 4)
        self.maxpool = nn.MaxPool3d(kernel_size=pool_size)
        self.conv = nn.Conv3d(
            in_channels=1, out_channels=num_filters, kernel_size=filter_size
        )
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout3d = nn.Dropout3d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(x)
        x = self.dropout3d(self.conv(x))
        x = self.relu(x)
        x = torch.flatten(x, 1)

        return x


class FFTFeatureModel(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        dropout_rate: float = 0.4,
        negative_slope: float = 0.02,
        num_filters: int = 10,
    ):
        """
        Use the raw BES channels values as input and perform a Fast Fourier Transform
        to input signals. This function takes in a 5-dimensional
        tensor of size: `(N, 1, signal_window_size, 8, 8)`, N=batch_size and
        performs a FFT followed by an absolute value of the input tensor. It
        then performs a 3-d convolution with a filter size identical to the spatial
        dimensions of the input to avoid the sliding of the kernel over the input.
        In the end, we will have a feature map that can be concatenated with other features.

        Args:
        -----
            args (argparse.Namespace): Command line arguments containing the information
                about signal_window.
            dropout_rate (float, optional): Fraction of total hidden units that will
                be turned off for drop out. Defaults to 0.2.
            negative_slope (float, optional): Slope of LeakyReLU activation for negative
                `x`. Defaults to 0.02.
            maxpool_size (int, optional): Size of the kernel used for maxpooling. Use
                0 to skip maxpooling. Defaults to 2.
            num_filters (int, optional): Dimensionality of the output space.
                Essentially, it gives the number of output kernels after convolution.
                Defaults to 10.
        """
        self.args = args
        temporal_size = int(self.args.signal_window_size // 2) + 1
        filter_size = (temporal_size, 8, 8)
        self.conv = nn.Conv3d(
            in_channels=1, out_channels=num_filters, kernel_size=filter_size
        )
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout3d = nn.Dropout3d(p=dropout_rate)

    def forward(self, x):
        # apply FFT to the input along the time dimension
        x = torch.abs(torch.fft.rfft(x, dim=2))
        x = self.dropout3d(self.conv(x))
        x = self.relu(x)
        x = torch.flatten(x, 1)

        return x


class CWTFeatureModel(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass
