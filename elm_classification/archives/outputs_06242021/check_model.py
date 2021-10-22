import argparse

import torch
import torch.nn as nn

from cnn_v2_model import CNNV2Model


class CheckModel(CNNV2Model):
    def check(self):
        model = nn.Sequential(self.conv1, self.conv2, self.conv3)
        return model

    def print_model(self):
        print(CNNV2Model(self.args))


if __name__ == "__main__":
    x = torch.rand(4, 1, 16, 12, 12)
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_window_size")
    args = parser.parse_args(["--signal_window_size", "16"])
    check_model = CheckModel(args=args)
    check_model.print_model()
    model = check_model.check()
    y = model(x)
    print(y.shape)
    print(torch.flatten(y, 1).shape)
