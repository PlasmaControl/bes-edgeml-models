from typing import Tuple, Union
import torch
import torch.nn as nn
from torchinfo import summary

import config


class CNNModel(nn.Module):
    def __init__(
        self,
        fc_units: Union[int, Tuple[int, int]] = (40, 20),
        dropout_rate: float = 0.3,
        negative_slope: float = 0.02,
        filter_size: int = 3,
        num_channels: Tuple[int, int] = (4, 8),
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
        super(CNNModel, self).__init__()
        filter1 = (8, filter_size, filter_size)
        self.conv1 = nn.Conv3d(
            in_channels=1, out_channels=num_channels[0], kernel_size=filter1
        )
        filter2 = (1, filter_size, filter_size)
        self.conv2 = nn.Conv3d(
            in_channels=num_channels[0],
            out_channels=num_channels[1],
            kernel_size=filter2,
        )
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout3d = nn.Dropout3d(p=dropout_rate)
        self.fc1 = nn.Linear(in_features=128, out_features=fc_units[0])
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(in_features=fc_units[0], out_features=fc_units[1])
        self.fc3 = nn.Linear(in_features=fc_units[1], out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(self.dropout3d(x))
        x = self.conv2(x)
        x = self.relu(self.dropout3d(x))
        x = x.view(config.batch_size, 1, -1)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)

        return x


class FeatureModel(nn.Module):
    def __init__(
        self,
        fc_units: Union[int, Tuple[int, int]] = (40, 20),
        dropout_rate: float = 0.3,
        negative_slope: float = 0.02,
        filter_size: tuple = (8, 4, 4),
        maxpool_size: int = 2,
        num_filters: int = 10,
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
        pool_size = [1, maxpool_size, maxpool_size]
        self.maxpool = nn.MaxPool3d(kernel_size=pool_size)
        self.conv = nn.Conv3d(
            in_channels=1, out_channels=num_filters, kernel_size=filter_size
        )
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout3d = nn.Dropout3d(p=dropout_rate)
        self.fc1 = nn.Linear(in_features=10, out_features=fc_units[0])
        self.fc2 = nn.Linear(in_features=fc_units[0], out_features=fc_units[1])
        self.fc3 = nn.Linear(in_features=fc_units[1], out_features=1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(x)
        x = self.dropout3d(self.conv(x))
        x = self.relu(x)
        x = x.view(config.batch_size, 1, -1)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)

        return x


def get_params(model: Union[CNNModel, FeatureModel]) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_details(model, x, input_size):
    print("\t\t\t\tMODEL SUMMARY")
    summary(model, input_size=input_size)
    print(f"Output size: {model(x).shape}")
    print(f"Model contains {get_params(model)} trainable parameters!")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureModel()
    input_size = (4, 1, 8, 8, 8)
    x = torch.rand(*input_size)
    model, x = model.to(device), x.to(device)
    model_details(model, x, input_size)
