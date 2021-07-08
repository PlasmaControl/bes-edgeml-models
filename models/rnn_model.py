import argparse

import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=args.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


# if __name__ == "__main__":
#     import sys
#     import os

#     sys.path.append(os.getcwd())
#     from options.train_arguments import TrainArguments
#     from src import data, utils

#     args, parser = TrainArguments().parse(verbose=True)
#     utils.test_args_compat(args, parser)
#     LOGGER = utils.get_logger(script_name="test_lstm_logs")
#     model = RNNModel()
#     data_obj = data.Data(args, logger=LOGGER)
#     # create train, valid and test data
#     train_data, valid_data, test_data = data_obj.get_data(
#         shuffle_sample_indices=args.shuffle_sample_indices
#     )
#     # create datasets
#     transforms = None
#     train_dataset = data.ELMDataset(
#         args, *train_data, transform=transforms, logger=LOGGER
#     )

#     valid_dataset = data.ELMDataset(
#         args, *valid_data, transform=transforms, logger=LOGGER
#     )

#     # training and validation dataloaders
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         drop_last=True,
#     )

#     valid_loader = torch.utils.data.DataLoader(
#         valid_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         drop_last=True,
#     )
#     inp, labels = next(iter(train_loader))
#     print(inp.shape)
#     inp = inp.squeeze()
#     inp = torch.flatten(inp, -2)
#     y = model(inp)
#     print(y.shape)
#     print(y.squeeze())
