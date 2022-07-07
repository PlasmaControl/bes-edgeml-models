import argparse

import torch
import torch.nn as nn


class CNN2DModel(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(CNN2DModel, self).__init__()
        self.args = args
        in_channels = 6 if self.args.data_preproc == "gradient" else 1
        if self.args.signal_window_size == 8:
            input_size = 512
        elif self.args.signal_window_size == 16:
            input_size = 1024
        elif self.args.signal_window_size == 32:
            input_size = 2048
        elif self.args.signal_window_size == 64:
            input_size = 4096
        elif self.args.signal_window_size == 128:
            input_size = 8192
        else:
            raise ValueError(
                "Input features for given signal window size are not parsed!"
            )
        projection_size = int(in_channels * input_size)
        self.project2d = torch.empty(
            projection_size,
            dtype=torch.float32,
            device=args.device,
        ).view(in_channels, -1, 8, 8)
        nn.init.normal_(self.project2d)
        self.project2d = nn.Parameter(self.project2d, requires_grad=True)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=8, kernel_size=3
        )
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.act = nn.LeakyReLU(negative_slope=0.05)
        self.dropout2d = nn.Dropout2d(p=0.4)
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # create the projection of the 3D tensor on a 2D tensor
        x = x[:, :, ...] * self.project2d
        # sum it across the temporal dimension
        x = torch.sum(x, axis=2)
        x = self.act(self.conv1(x))
        x = self.dropout2d(x)
        x = self.act(self.conv2(x))
        x = self.dropout2d(x)
        x = self.act(self.conv3(x))
        x = self.dropout2d(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.act(self.fc2(x))
        x = self.dropout(x)
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
            "--data_preproc",
            "wavelet",
            "--signal_window_size",
            "128",
        ],  # ["--device", "cpu"]
    )
    shape = (16, 1, 128, 8, 8)
    x = torch.ones(*shape)
    device = torch.device(
        "cpu"
    )  # "cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    model = CNN2DModel(args)
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
