import argparse
from typing import Tuple, Union
from collections import OrderedDict

import torch
import torch.nn as nn


class FeatureModel(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        fc_units: Union[int, Tuple[int, int]] = (40, 20),
        dropout_rate: float = 0.4,
        negative_slope: float = 0.02,
        filter_size: tuple = (16, 8, 8),
        num_filters: int = 32,
    ):
        """
        8x8 + time feature blocks followed by fully-connected layers. This function
        takes in a 4-dimensional tensor of size: `(1, signal_window_size, 8, 8)`
        performs maxpooling to downsample the spatial dimension by half, perform a
        3-d convolution with a filter size identical to the spatial dimensions of the
        input to avoid the sliding of the kernel over the input. Finally, it adds a
        couple of fully connected layers on the output of the 3-d convolution.

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
        """
        super(FeatureModel, self).__init__()
        self.args = args
        filter_size = (int(self.args.signal_window_size), filter_size[1], filter_size[2])
        self.conv = nn.Conv3d(
            in_channels=1, out_channels=num_filters, kernel_size=filter_size
        )
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout3d = nn.Dropout3d(p=dropout_rate)
        input_features = 32 if self.args.signal_window_size == 8 else 32
        self.fc1 = nn.Linear(
            in_features=input_features, out_features=fc_units[0]
        )
        self.fc2 = nn.Linear(in_features=fc_units[0], out_features=fc_units[1])
        self.fc3 = nn.Linear(in_features=fc_units[1], out_features=1)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.layers = OrderedDict([
            ('conv', self.conv),
            ('fc1', self.fc1),
            ('fc2', self.fc2),
            ('fc3', self.fc3)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout3d(self.conv(x))
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    signal_window_size = 16
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_window_size", type=int)

    args = parser.parse_args(["--signal_window_size", str(signal_window_size)])

    model = FeatureModel(args)
    x = torch.rand(4, 1, signal_window_size, 8, 8)
    y = model(x)
    print(y.shape)
