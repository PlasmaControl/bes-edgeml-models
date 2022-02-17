import argparse

import torch
import torch.nn as nn


def feature_extractor(
    args: argparse.Namespace,
    in_channels: int,
    out_channels: list,
) -> nn.Sequential:
    kernel_size = 3 if args.signal_window_size == 16 else 5
    block = nn.Sequential(
        nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels[0],
            kernel_size=kernel_size,
        ),
        nn.LeakyReLU(negative_slope=0.05),
        nn.MaxPool1d(kernel_size=2),
        nn.Conv1d(
            in_channels=out_channels[0],
            out_channels=out_channels[1],
            kernel_size=kernel_size,
        ),
        nn.LeakyReLU(negative_slope=0.05),
        nn.MaxPool1d(kernel_size=2),
    )
    return block


def intermediate_fc(fc_units: list) -> nn.Sequential:
    block = nn.Sequential(
        nn.Linear(in_features=fc_units[0], out_features=fc_units[1]),
        nn.Linear(in_features=fc_units[1], out_features=fc_units[2]),
        nn.Linear(in_features=fc_units[2], out_features=1),
    )
    return block


class MTSCNNModel(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        in_channels: int,
        out_channels: list,
        diag_fc_units: list,
        detect_fc_units: list,
    ):
        super(MTSCNNModel, self).__init__()
        self.args = args
        self.conv_block = {}
        self.fc_block = {}
        for i in range(64):
            self.conv_block[f"ch_{i+1}"] = feature_extractor(
                self.args,
                in_channels=in_channels,
                out_channels=out_channels,
            )
            self.fc_block[f"ch_{i+1}"] = intermediate_fc(diag_fc_units)
        self.conv_block = nn.ModuleDict(self.conv_block)
        self.fc_block = nn.ModuleDict(self.fc_block)
        self.fc1 = nn.Linear(
            in_features=detect_fc_units[0], out_features=detect_fc_units[1]
        )
        self.fc2 = nn.Linear(
            in_features=detect_fc_units[1], out_features=detect_fc_units[2]
        )

    def forward(self, x):
        x = torch.flatten(x, 3)
        logits = []
        for i in range(64):
            xi = self.conv_block[f"ch_{i+1}"](x[..., i])
            # print(f"Shape after conv block: {xi.shape}")
            xi = torch.flatten(xi, 1)
            # print(f"Shape after flattening: {xi.shape}")
            xi = self.fc_block[f"ch_{i+1}"](xi)
            # print(f"Shape after fc block: {xi.shape}")
            logits.append(xi)
        logits = torch.cat(logits, dim=1)
        # print(f"Shape after concat: {logits.shape}")
        logits = self.fc1(logits)
        logits = self.fc2(logits)
        return logits


if __name__ == "__main__":
    from torchinfo import summary

    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_preproc", type=str, default="unprocessed")
    parser.add_argument("--signal_window_size", type=int)
    # parser.add_argument("--device", default="cpu")
    args = parser.parse_args(
        [
            "--signal_window_size",
            "16",
        ],  # ["--device", "cpu"]
    )
    shape = (4, 1, 16, 8, 8)
    x = torch.rand(*shape)
    model = MTSCNNModel(
        args,
        in_channels=1,
        out_channels=[4, 8],
        diag_fc_units=[16, 16, 64],
        detect_fc_units=[64, 32, 1],
    )
    print(summary(model, input_size=shape, device="cpu"))

    for param in list(model.named_parameters()):
        print(
            f"param name: {param[0]},\nshape: {param[1].shape}, requires_grad: {param[1].requires_grad}"
        )
    print(
        f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    y = model(x)
    print(y.shape)
