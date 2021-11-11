import argparse

import torch
import torch.nn as nn


class CNNGradientModel(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(CNNGradientModel, self).__init__()
        self.args = args
        in_channels = 3
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=8, kernel_size=3
        )
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu = nn.LeakyReLU(negative_slope=0.05)
        self.dropout2d = nn.Dropout2d(p=0.4)
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout2d(x)
        x = self.relu(self.conv2(x))
        x = self.dropout2d(x)
        x = self.relu(self.conv3(x))
        x = self.dropout2d(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
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
            "64",
        ],  # ["--device", "cpu"]
    )
    shape = (16, 3, 8, 8)
    x = torch.rand(*shape)
    device = torch.device(
        "cpu"
    )  # "cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    model = CNNGradientModel(args)
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
