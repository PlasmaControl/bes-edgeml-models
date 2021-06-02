import os
import time

import torch
import torch.nn as nn

import numpy as np

import config, data, utils


LOGGER = utils.get_logger(script_name=__name__, log_file="output_logs.log")


def train_loop(data_obj: data.Data, fold: int):
    LOGGER.info("-" * 30)
    LOGGER.info(f"       Training fold: {fold}       ")
    LOGGER.info("-" * 30)

    # create train, valid and test data
    train_data, valid_data, _ = data_obj.get_data(
        shuffle_sample_indices=True, fold=fold
    )

    # create datasets
    train_dataset = data.ELMDataset(
        *train_data, config.signal_window_size, config.label_look_ahead
    )

    valid_dataset = data.ELMDataset(
        *valid_data, config.signal_window_size, config.label_look_ahead
    )

    # training and validation dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, valid_loader


if __name__ == "__main__":
    data_obj = data.Data(kfold=True)
    train_loader, _ = train_loop(data_obj, fold=0)
    image, label = next(iter(train_loader))
    LOGGER.info(f"Input size: {image.shape}")
    LOGGER.info(f"Target: {label}")
