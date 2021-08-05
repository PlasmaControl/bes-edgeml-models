import logging
import torch
from torch.utils.data import DataLoader
import argparse
import os

import src.utils
import src.data


def get_dataloader(args: argparse.Namespace,
                   logger: logging.Logger,
                   use_saved=True):
    # Name file to store dataloader from args
    dl_fname = f'dataloader_{args.data_mode}_lookahead_{args.label_look_ahead}_batchsize_{args.batch_size}.pt'

    try:
        # if the file already exists
        dl_fpath = os.path.join('data/saved_dataloaders', dl_fname)
        logger.info(f'Found dataloader object at {dl_fpath}')
        # then load the dataloader
        train_dataloader = torch.load(dl_fpath)
        logger.info(f'Loaded dataloader object')
    except FileNotFoundError:
        if use_saved:
            logger.info(f'Dataloader {dl_fname} not found. Creating dataloader from source data.')

        data_ = src.data.Data(args, logger)
        train_data, valid_data, test_data = data_.get_data(shuffle_sample_indices=True)

        # import data
        train_dataset = src.data.ELMDataset(
            args,
            *train_data,
            logger=logger,
            transform=None
        )

        # create dataloader objects from dataset
        batch_size = args.batch_size
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)

        torch.save(train_dataloader, f'data/saved_dataloaders/{dl_fname}')
        logger.info(f'Saved dataloader object at data/saved_dataloaders/{dl_fname}')

    return train_dataloader


def get_model(args: argparse.Namespace,
              logger: logging.Logger):
    _, model_cpt_path = src.utils.create_output_paths(args)
    model_cpt_file = os.path.join(model_cpt_path,
                                  f'{args.model_name}_{args.data_mode}_lookahead_{args.label_look_ahead}.pth')

    logger.info(f'Found {args.model_name} state dict at {model_cpt_file}.')
    model_cls = src.utils.create_model(args.model_name)
    model = model_cls(args)
    state_dict = torch.load(model_cpt_file)['model']
    model.load_state_dict(state_dict)
    logger.info(f'Loaded {args.model_name} state dict.')

    return model
