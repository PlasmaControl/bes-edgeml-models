import logging
import torch
from torch.utils.data import DataLoader
import argparse
import os
import re

try:
    import src.utils
    import src.data
except:
    pass


def get_dataloader(args: argparse.Namespace,
                   logger: logging.Logger,
                   use_saved=True):
    # Name file to store dataloader from args
    if args.generated:
        data_name_ = args.data_mode + '_' + re.split('[_.]', args.input_file)[-2]
    else:
        data_name_ = args.data_mode
    dl_fname = f'dataloader_{data_name_}_lookahead_{args.label_look_ahead}_batchsize_{args.batch_size}.pt'

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
    if args.generated:
        model_name_ = args.model_name + '_' + re.split('[_.]', args.input_file)[-2]
    else:
        model_name_ = args.model_name
    model_cpt_file = os.path.join(model_cpt_path,
                                  f'{model_name_}_{args.data_mode}_lookahead_{args.label_look_ahead}.pth')

    logger.info(f'Found {model_name_} state dict at {model_cpt_file}.')
    model_cls = src.utils.create_model(args.model_name)
    model = model_cls(args)
    state_dict = torch.load(model_cpt_file)['model']
    model.load_state_dict(state_dict)
    logger.info(f'Loaded {model_name_} state dict.')

    return model


def generate_data():
    import numpy as np
    from scipy.signal import square
    # test
    gen_signals = np.ones((2000, 64))
    gen_labels = np.empty((2000,))
    gen_time = np.arange(0, 2000)
    gen_signals[:1500] = gen_signals[:1500] * np.cos(np.linspace(0, 2 * np.pi * 15, 1500, endpoint=False))[:,
                                              np.newaxis]
    gen_signals[1500:] = gen_signals[1500:] * (square(np.linspace(0, 2 * np.pi * 5, 500, endpoint=False)))[:,
                                              np.newaxis]
    gen_labels[:1500] = 0
    gen_labels[1500:] = 1

    return gen_labels, gen_signals.T, gen_time


if __name__ == '__main__':
    import h5py as hf

    num_shots = 200
    data_path = '/home/jazimmerman/PycharmProjects/bes-edgeml-models/bes-edgeml-models/data/'
    f = hf.File(data_path + 'generated_data_square.hdf5', 'w-')
    for x in range(num_shots):
        labels, signals, time = generate_data()

        shot_grp = f.create_group(f'{x:05}')
        labels_subgrp = shot_grp.create_dataset('labels', labels.shape, data=labels)
        signals_subgrp = shot_grp.create_dataset('signals', signals.shape, data=signals)
        time_subgrp = shot_grp.create_dataset('time', time.shape, data=time)
    f.close()
