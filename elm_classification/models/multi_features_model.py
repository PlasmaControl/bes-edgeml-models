import argparse
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class RawFeatureModel(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        dropout_rate: float = 0.4,
        negative_slope: float = 0.02,
        maxpool_size: int = 2,
        num_filters: int = 48,
    ):
        """
        Use the raw BES channels values as features. This function takes in a 5-dimensional
        tensor of size: `(N, 1, signal_window_size, 8, 8)`, N=batch_size and
        performs maxpooling to downsample the spatial dimension by half, perform a
        3-d convolution with a filter size identical to the spatial dimensions of the
        input to avoid the sliding of the kernel over the input. Finally, a feature map
        is generated that can be concatenated with other features.

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
        spatial_dim = int(8 // maxpool_size)
        filter_size = (self.args.signal_window_size, spatial_dim, spatial_dim)
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
        num_filters: int = 48,
    ):
        """
        Use the raw BES channels values as input and perform a Fast Fourier Transform
        to input signals. This function takes in a 5-dimensional
        tensor of size: `(N, 1, signal_window_size, 8, 8)`, N=batch_size and
        performs a FFT followed by an absolute value of the input tensor. It
        then performs a 3-d convolution with a filter size identical to the spatial
        dimensions of the input so that the receptive field is the same
        size as input in both spatial and temporal axes. Again, we will use the
        feature map and combine it with other features before feeding it into a
        classifier.

        Args:
        -----
            args (argparse.Namespace): Command line arguments containing the information
                about signal_window.
            dropout_rate (float, optional): Fraction of total hidden units that will
                be turned off for drop out. Defaults to 0.2.
            negative_slope (float, optional): Slope of LeakyReLU activation for negative
                `x`. Defaults to 0.02.
            num_filters (int, optional): Dimensionality of the output space.
                Essentially, it gives the number of output kernels after convolution.
                Defaults to 10.
        """
        super(FFTFeatureModel, self).__init__()
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
        x = x.to(self.args.device)  # needed for PowerPC architecture
        x = torch.abs(torch.fft.rfft(x, dim=2))
        x = self.dropout3d(self.conv(x))
        x = self.relu(x)
        x = torch.flatten(x, 1)

        return x


class CWTFeatureModel(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        dropout_rate: float = 0.4,
        negative_slope: float = 0.02,
        num_filters: int = 48,
    ):
        """
        Use features from the output of continuous wavelet transform. The model architecture
        is similar to the `RawFeatureModel`. This model takes in a 6-dimensional
        tensor of size: `(N, 1, signal_window_size, n_scales, 8, 8)`, where `N`=batch_size, and
        `n_scales`=number of different scales used (which are equal to `signal_window_size`).
        For each signal block, only the scales and BES channels for the leading time
        steps are used as model input which is a 5-dimensional tensor of size (`N, 1, n_scales, 8, 8)`.
        The model then performs a 3-d convolution with a filter size identical to
        the spatial dimensions of the input so that the receptive field is the same
        size as input in both spatial and temporal axes. In the end, we will have
        a feature map that can be concatenated with other features.

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
        super(CWTFeatureModel, self).__init__()
        self.args = args
        # filter_size = (
        #     (self.args.signal_window_size, 8, 8)
        #     if self.args.signal_window_size <= 64
        #     else (int(self.args.signal_window_size / 2), 8, 8)
        # )
        filter_size = (int(np.log2(1024)) + 1, 8, 8)
        self.conv = nn.Conv3d(
            in_channels=1, out_channels=num_filters, kernel_size=filter_size
        )
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout3d = nn.Dropout3d(p=dropout_rate)

    def forward(self, x):
        x = x[:, :, -1, ...]  # take only the last time step
        x = self.dropout3d(self.conv(x))
        x = self.relu(x)
        x = torch.flatten(x, 1)

        return x


class MultiFeaturesModel(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        raw_features_model: RawFeatureModel,
        fft_features_model: FFTFeatureModel,
        cwt_features_model: CWTFeatureModel,
        dropout_rate: float = 0.4,
        negative_slope: float = 0.02,
    ):
        """Encapsulate all the feature models to create a composite model that
        uses all the feature maps. It takes in the class instances of
        `RawFeatureModel`, `FFTFeatureModel` and `CWTFeatureModel` along with the
        dropout rate and negative slope of the LeakyReLU activation. Once all
        the feature maps are computed, it concatenates them together and pass them
        through a couple of fully connected layers wich act as the classifier.

        Args:
        -----
            args (argparse.Namespace): Command line arguments.
            raw_features_model (RawFeatureModel): Instance of `RawFeatureModel` class.
            fft_features_model (FFTFeatureModel): Instance of `FFTFeatureModel` class.
            cwt_features_model (CWTFeatureModel): Instance of `CWTFeatureModel` class.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.4.
            negative_slope (float, optional): Slope of activation functions. Defaults to 0.02.
        """
        super(MultiFeaturesModel, self).__init__()
        self.args = args
        self.raw_features_model = raw_features_model
        self.fft_features_model = fft_features_model
        self.cwt_features_model = cwt_features_model
        input_features = 144 if self.args.use_fft else 96
        self.fc1 = nn.Linear(in_features=input_features, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x_raw, x_cwt):
        # extract raw and cwt processed signals
        if self.args.use_fft:
            raw_features = self.raw_features_model(x_raw)
            fft_features = self.fft_features_model(x_raw)
            cwt_features = self.cwt_features_model(x_cwt)
            x = torch.cat([raw_features, fft_features, cwt_features], dim=1)
        else:
            raw_features = self.raw_features_model(x_raw)
            cwt_features = self.cwt_features_model(x_cwt)
            x = torch.cat([raw_features, cwt_features], dim=1)

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
            "16",
        ],  # ["--device", "cpu"]
    )
    shape_raw = (16, 1, 16, 8, 8)
    shape_cwt = (16, 1, 16, 16, 8, 8)
    x_raw = torch.randn(*shape_raw)
    x_cwt = torch.randn(*shape_cwt)

    device = torch.device(
        "cpu"
    )  # "cuda" if torch.cuda.is_available() else "cpu")
    x_raw = x_raw.to(device)
    x_cwt = x_cwt.to(device)
    raw_model = RawFeatureModel(args)
    fft_model = FFTFeatureModel(args)
    cwt_model = CWTFeatureModel(args)
    model = MultiFeaturesModel(args, raw_model, fft_model, cwt_model)
    print(summary(model, input_size=(shape_raw, shape_cwt), device="cpu"))

    for param in list(model.named_parameters()):
        print(
            f"param name: {param[0]},\nshape: {param[1].shape}, requires_grad: {param[1].requires_grad}"
        )
    print(
        f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    y = model(x_raw, x_cwt)
    print(y)
    print(y.shape)
