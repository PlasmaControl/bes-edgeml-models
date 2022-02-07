import argparse
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from custom_wavelet import transform


class _BaseFeatureModel(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(_BaseFeatureModel, self).__init__()
        self.args = args
        assert np.ceil(np.log2(self.args.signal_window_size)) == np.floor(
            np.log2(self.args.signal_window_size)
        ), "Size of signal window should be a power of 2."
        assert self.args.mf_maxpool_size in [
            1,
            2,
            4,
        ], "Max pool size can only be 1 (no pooling), 2, and 4."
        if self.args.mf_maxpool_size > 1:
            self.maxpool = nn.MaxPool3d(
                kernel_size=[1, self.args.mf_maxpool_size, self.args.mf_maxpool_size],
            )
        else:
            self.maxpool = None
        self.relu = nn.LeakyReLU(negative_slope=self.args.mf_relu_negative_slope)
        self.dropout3d = nn.Dropout3d(p=self.args.mf_dropout_rate)
        self.num_filters = None
        self.conv = None

    def _conv_dropout_relu_flatten(self, x: torch.Tensor) -> torch.Tensor:
        if self.maxpool:
            x = self.maxpool(x)
        if self.conv is not None:
            x = self.relu(self.dropout3d(self.conv(x)))
            x = torch.flatten(x, 1)
            return x
        else:
            raise NotImplementedError("Convolution layer is not implemented.")


class RawFeatureModel(_BaseFeatureModel):
    def __init__(self, args):
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
        """
        super(RawFeatureModel, self).__init__(args)
        self.num_filters = self.args.raw_num_filters
        spatial_dim = int(8 // self.args.mf_maxpool_size)
        filter_size = (self.args.signal_window_size, spatial_dim, spatial_dim)
        self.conv = nn.Conv3d(
            in_channels=1, out_channels=self.num_filters, kernel_size=filter_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_dropout_relu_flatten(x)

        return x


class FFTFeatureModel(_BaseFeatureModel):
    def __init__(self, args):
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
        """
        super(FFTFeatureModel, self).__init__(args)
        self.num_filters = self.args.fft_num_filters
        temporal_size = int(self.args.signal_window_size // 2) + 1
        filter_size = (temporal_size, 8, 8)
        self.conv = nn.Conv3d(
            in_channels=1, out_channels=self.num_filters, kernel_size=filter_size
        )

    def forward(self, x):
        # apply FFT to the input along the time dimension
        x = x.to(self.args.device)  # needed for PowerPC architecture
        x = torch.abs(torch.fft.rfft(x, dim=2))
        x = self._conv_dropout_relu_flatten(x)

        return x


class CWTFeatureModel(_BaseFeatureModel):
    def __init__(self, args):
        """
        Use features from the output of continuous wavelet transform. The model architecture
        is similar to the `RawFeatureModel`. This model takes in a 6-dimensional
        tensor of size: `(N, 1, signal_window_size, 8, 8)`, where `N`=batch_size.
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
        """
        super(CWTFeatureModel, self).__init__(args)
        self.num_filters = self.args.wt_num_filters
        filter_size = (len(self.args.scales), 8, 8)
        self.conv = nn.Conv3d(
            in_channels=1, out_channels=self.num_filters, kernel_size=filter_size
        )

    def forward(self, x):
        if self.args.scales is not None:
            # get CWT batch wise
            x_cwt = transform.continuous_wavelet_transform(
                self.args.signal_window_size, self.args.scales, x, self.args.device
            )
        else:
            raise ValueError(
                "Using continuous wavelet transform but iterable containing scales is not parsed!"
            )
        x_cwt = self._conv_dropout_relu_flatten(x_cwt)

        return x_cwt


class MultiFeaturesModel(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        raw_features_model: Union[RawFeatureModel, None] = None,
        fft_features_model: Union[FFTFeatureModel, None] = None,
        cwt_features_model: Union[CWTFeatureModel, None] = None,
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
        input_features = 0
        for model in [
            self.raw_features_model,
            self.fft_features_model,
            self.cwt_features_model,
        ]:
            if model is not None:
                input_features += model.num_filters
        self.fc1 = nn.Linear(in_features=input_features, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)
        self.dropout = nn.Dropout(p=self.args.mf_dropout_rate)
        self.relu = nn.LeakyReLU(negative_slope=self.args.mf_relu_negative_slope)

    def forward(self, x):
        # extract raw and cwt processed signals
        raw_features = self.raw_features_model(x) if self.raw_features_model else None
        fft_features = self.fft_features_model(x) if self.fft_features_model else None
        cwt_features = self.cwt_features_model(x) if self.cwt_features_model else None

        active_features_list = [
            features
            for features in [raw_features, fft_features, cwt_features]
            if features is not None
        ]
        x = torch.cat(active_features_list, dim=1)

        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_preproc", type=str, default="unprocessed")
    parser.add_argument("--signal_window_size", type=int)
    parser.add_argument("--scales", nargs="+", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(
        [
            "--signal_window_size",
            "16",
            "--scales",
            "1",
            "2",
            "4",
            "8",
            "16",
        ],
    )
    shape_raw = (16, 1, 16, 8, 8)
    shape_cwt = (16, 1, 5, 8, 8)
    x_raw = torch.randn(*shape_raw)
    x_cwt = torch.randn(*shape_cwt)

    device = torch.device("cpu")  # "cuda" if torch.cuda.is_available() else "cpu")
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
