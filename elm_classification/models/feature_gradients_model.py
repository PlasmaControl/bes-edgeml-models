import argparse
from typing import Tuple, Union

import torch
import torch.nn as nn


class SpatialFeatures(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        num_filters: int = 10,
        maxpool_size: int = 2,
    ):
        super(SpatialFeatures, self).__init__()
        pool_size = [1, maxpool_size, maxpool_size]
        self.args = args
        self.maxpool = nn.MaxPool3d(kernel_size=pool_size)
        self.filter_size = (8, 4, 4)
        self.conv_spatial = nn.Conv3d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=self.filter_size,
        )

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv_spatial(x)
        return torch.flatten(x, 1)


class TemporalFeatures(nn.Module):
    def __init__(self):
        super(TemporalFeatures, self).__init__()
        self.conv_temporal = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5
        )

    def forward(self, x):
        dxdt = x[:, :, -1, ...] - x[:, :, 0, ...]
        dxdt = self.conv_temporal(dxdt)
        return torch.flatten(dxdt, 1)


class FeatureGradientsModel(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        spatial: SpatialFeatures,
        temporal: TemporalFeatures,
        dropout_rate: float = 0.4,
        negative_slope: float = 0.02,
    ):
        super(FeatureGradientsModel, self).__init__()
        self.args = args
        self.spatial = spatial
        self.temporal = temporal
        if self.args.signal_window_size == 16:
            input_features = 602
        # elif self.args.signal_window_size == 32:
        #     input_features = 250
        # elif self.args.signal_window_size == 64:
        #     input_features = 570
        # elif self.args.signal_window_size == 128:
        #     input_features = 1210
        # elif self.args.signal_window_size == 256:
        #     input_features = 2490
        # elif self.args.signal_window_size == 512:
        #     input_features = 5050
        else:
            raise ValueError(
                "Input features for given signal window size are not parsed!"
            )
        self.fc1 = nn.Linear(in_features=input_features, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, input):
        spatial_features = self.spatial(input)
        temporal_features = self.temporal(input)
        # print(f"Spatial features: {spatial_features.shape}")
        # print(f"Temporal features: {temporal_features.shape}")
        x = torch.cat([spatial_features, temporal_features], dim=1)
        x = self.dropout(self.fc1(x))
        x = self.relu(x)
        x = self.dropout(self.fc2(x))
        x = self.relu(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_window_size", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(
        [
            "--signal_window_size",
            "16",
        ],  # ["--device", "cpu"]
    )
    shape = (4, 1, 16, 8, 8)
    x = torch.rand(*shape)
    device = torch.device(
        "cpu"
    )  # "cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    spatial = SpatialFeatures(args)
    temporal = TemporalFeatures()
    model = FeatureGradientsModel(args, spatial, temporal)
    print(summary(model, input_size=shape, device="cpu"))

    for param in list(model.named_parameters()):
        print(
            f"param name: {param[0]},\nshape: {param[1].shape}, requires_grad: {param[1].requires_grad}"
        )
    print(
        f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    y = model(x)
    print(y)
    print(y.shape)
