import argparse

import numpy as np
import torch
import torch.nn as nn


class FeatureExtractorModel(nn.Module):
    def __init__(
        self, args: argparse.Namespace, num_filters=16, filter_size=(16, 5, 5)
    ):
        super(FeatureExtractorModel, self).__init__()
        self.args = args
        self.conv = nn.Conv3d(
            in_channels=1, out_channels=num_filters, kernel_size=filter_size
        )
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 1, 1))
        input_features = 144  # if self.args.signal_window_size == 8 else 144
        self.fc = nn.Linear(in_features=input_features, out_features=1)
        self.relu = nn.LeakyReLU(negative_slope=0.02)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    window_size = 16
    x = torch.rand(1, 1, window_size, 8, 8)
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_window_size", type=int)
    args = parser.parse_args(
        [
            "--signal_window_size",
            str(window_size),
        ],
    )

    model = FeatureExtractorModel(args, filter_size=(window_size, 5, 5))
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # tensors = list(model.conv.parameters())
    # w = model.conv.weight
    # print(w.shape)
    # model.conv.register_forward_hook(get_activation("conv"))
    # print(f"Model contains {total_params} trainable parameters!")
    y = model(x)
    print(y.shape)
