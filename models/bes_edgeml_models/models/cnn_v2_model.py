import argparse
from typing import Tuple, Union

import torch
import torch.nn as nn


class CNNV2Model(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        fc_units: Union[int, Tuple[int, int]] = (64, 32),
        dropout_rate: float = 0.3,
        negative_slope: float = 0.02,
        filter_size: int = 3,
        num_channels: Tuple[int, int] = (8, 16, 32),
    ):
        """
        CNN layers followed by fully-connected layers.

        Args:
        -----
            fc_units (Union[int, Tuple[int, int]], optional): Number of hidden
                units in each layer. Defaults to (40, 20).
            dropout_rate (float, optional): Fraction of total hidden units that will
                be turned off for drop out. Defaults to 0.2.
            negative_slope (float, optional): Slope of LeakyReLU activation for negative
                `x`. Defaults to 0.02.
            filter_size (int, optional): Size of the convolution kernel/filter. Defaults
                to 3.
            num_channels (Tuple[int, int], optional): Dimensionality of the output space.
                Essentially, it gives the number of output kernels after convolution.
                Defaults to (4, 8).
        """
        super(CNNV2Model, self).__init__()
        self.args = args
        filter = (filter_size, filter_size, filter_size)
        self.conv1 = nn.Conv3d(
            in_channels=1, out_channels=num_channels[0], kernel_size=filter
        )
        self.conv2 = nn.Conv3d(
            in_channels=num_channels[0],
            out_channels=num_channels[1],
            kernel_size=filter,
        )
        self.conv3 = nn.Conv3d(
            in_channels=num_channels[1],
            out_channels=num_channels[2],
            kernel_size=filter,
        )
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout3d = nn.Dropout3d(p=dropout_rate)
        input_features = 256 if self.args.signal_window_size == 8 else 1280
        self.fc1 = nn.Linear(
            in_features=input_features, out_features=fc_units[0]
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(in_features=fc_units[0], out_features=fc_units[1])
        self.fc3 = nn.Linear(in_features=fc_units[1], out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(self.dropout3d(x))
        x = self.conv2(x)
        x = self.relu(self.dropout3d(x))
        x = self.conv3(x)
        x = self.relu(self.dropout3d(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)

        return x
