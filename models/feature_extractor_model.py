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
        self.activation = {}
        self.conv = nn.Conv3d(
            in_channels=1, out_channels=num_filters, kernel_size=filter_size
        )
        input_features = 16 if self.args.signal_window_size == 8 else 256
        self.fc = nn.Linear(in_features=input_features, out_features=1)
        self.relu = nn.LeakyReLU(negative_slope=0.02)

    # def forward_hook(self):
    #     def hook(module, input, output):
    #         self.activation[self.name] = output.detach()
    #
    #     return hook

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# if __name__ == "__main__":
#     x = torch.rand(1, 1, 16, 8, 8)
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--signal_window_size", type=int)
#     args = parser.parse_args(
#         [
#             "--signal_window_size",
#             "16",
#         ],
#     )

#     model = FeatureExtractorModel(args, filter_size=(16, 5, 5))
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     tensors = list(model.conv.parameters())
#     w = model.conv.weight
#     print(w.shape)
#     model.conv.register_forward_hook(get_activation("conv"))
#     print(f"Model contains {total_params} trainable parameters!")
#     y = model(x)
#     print(y.shape)
#     print(activation["conv"].shape)
