import argparse

import torch
import torch.nn as nn


class CNN2DModel(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(CNN2DModel, self).__init__()
        self.args = args
        projection_size = 512 if self.args.signal_window_size == 8 else 1024
        self.project2d = torch.empty(
            projection_size, dtype=torch.float32, requires_grad=True
        ).view(-1, 8, 8)
        nn.init.normal_(self.project2d)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5)
        self.act = nn.Hardswish()
        self.dropout2d = nn.Dropout2d(p=0.4)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(in_features=64, out_features=16)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(in_features=16, out_features=1)

    def forward(self, x):
        # create the projection of the 3D tensor on a 2D tensor
        x = x[:, 0, ...] * self.project2d
        x = torch.sum(x, axis=1)
        # add dimension for input channels
        x.unsqueeze_(1)
        x = self.act(self.conv1(x))
        x = self.dropout2d(x)
        x = self.act(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# if __name__ == "__main__":
#     x = torch.ones(16, 1, 8, 8, 8)
#     model = CNN2DModel()
#     y = model(x)
#     print(y)
#     print(y.shape)
