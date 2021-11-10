import argparse
from typing import Tuple, Union

import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
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
        self.args = args
        filter1 = (8, filter_size, filter_size)
        in_channels = 6 if self.args.data_preproc == "gradient" else 1
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=num_channels[0],
            kernel_size=filter1,
        )
        filter2 = (1, filter_size, filter_size)
        self.conv2 = nn.Conv3d(
            in_channels=num_channels[0],
            out_channels=num_channels[1],
            kernel_size=filter2,
        )
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout3d = nn.Dropout3d(p=dropout_rate)
        if self.args.signal_window_size == 8:
            input_features = 128
        elif self.args.signal_window_size == 16:
            input_features = 1152
        elif self.args.signal_window_size == 32:
            input_features = 3200
        elif self.args.signal_window_size == 64:
            input_features = 7296
        elif self.args.signal_window_size == 128:
            input_features = 15488
        elif self.args.signal_window_size == 256:
            input_features = 31872
        else:
            raise ValueError(
                "Input features for given signal window size are not parsed!"
            )
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
        x = torch.flatten(x, 1)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_preproc", type=str, default="unprocessed")
    parser.add_argument("--signal_window_size", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(
        [
            "--signal_window_size",
            "256",
        ],  # ["--device", "cpu"]
    )
    shape = (16, 1, 256, 8, 8)
    x = torch.ones(*shape)
    device = torch.device(
        "cpu"
    )  # "cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    model = CNNModel(args)
    print(summary(model, input_size=shape, device="cpu"))

    for param in list(model.named_parameters()):
        print(
            f"param name: {param[0]},\nshape: {param[1].shape}, requires_grad: {param[1].requires_grad}"
        )
    print(
        f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    y = model(x)
    # print(y)
    print(y.shape)
