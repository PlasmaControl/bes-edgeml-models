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


def generate_data(type: str, add_noise=False):
    import numpy as np
    from scipy.signal import square

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    gen_signals = np.ones((2000, 64))
    gen_labels = np.zeros((2000,))
    gen_time = np.arange(0, 2000)

    if type == 'square':
        gen_signals[:1500] = gen_signals[:1500] * np.cos(np.linspace(0, 2 * np.pi * 15, 1500, endpoint=False))[:,
                                                  np.newaxis]
        gen_signals[1500:] = gen_signals[1500:] * (square(np.linspace(0, 2 * np.pi * 5, 500, endpoint=False)))[:,
                                                  np.newaxis]
        gen_labels[:1500] = 0
        gen_labels[1500:] = 1

    if type == 'sigmoid':
        arr = np.linspace(-1, 1, 2000, endpoint=False)
        sigarr = sigmoid(50 * (arr - 0.6))
        gen_signals = gen_signals * sigarr[:, np.newaxis]
        gen_labels[sigarr > 0.03] = 1

    if type == 'fm':
        arr = np.linspace(-1, 1, 2000, endpoint=False)
        sigarr = sigmoid(50 * (arr - 0.6))
        gen_labels[sigarr > 0.03] = 1

        # Samples per second
        sps = 2000

        # Duration
        duration_s = 1.0

        # ac: amplitude of the carrier. Should be kept at 1.0 in this script
        # you would modify it if you were micing it with, or modulating other,
        # waveforms.

        # carrier_hz: Frequency of the carrier
        # fm_hz: Frequency of the frequency modulator
        # k_p: deviation constant
        carrier_amplitude = 1.0
        carrier_hz = 100.0
        fm_hz = 10.0
        k = 2.0

        # Our final waveform is going to be calculated as the cosine of carrier and
        # frequency modulated terms.

        # First, define our range of sample numbers
        each_sample_number = np.arange(duration_s * sps)

        # Create the term that create the carrier
        carrier = 2 * np.pi * each_sample_number * carrier_hz / sps

        # Now create the term that is the frequency modulator
        # modulator = k * np.sin(2 * np.pi * each_sample_number * fm_hz / sps)
        modulator = k * np.cumsum(sigarr) / 25

        # Now create the modulated waveform, and attenuate it
        waveform = np.sin(carrier + modulator)
        gen_signals = gen_signals * waveform[:, np.newaxis]

    if add_noise:
        noise = np.random.normal(0, 0.05, gen_signals.shape)
        gen_signals += noise
        gen_signals = NormalizeData(gen_signals)

    return gen_labels, gen_signals.T, gen_time


if __name__ == '__main__':
    import h5py as hf

    num_shots = 200
    data_path = '/home/jazimmerman/PycharmProjects/bes-edgeml-models/bes-edgeml-models/data/'
    f = hf.File(data_path + 'generated_data_fm1.hdf5', 'w-')
    for x in range(num_shots):
        labels, signals, time = generate_data(type='fm', add_noise=True)

        shot_grp = f.create_group(f'{x:05}')
        labels_subgrp = shot_grp.create_dataset('labels', labels.shape, data=labels)
        signals_subgrp = shot_grp.create_dataset('signals', signals.shape, data=signals)
        time_subgrp = shot_grp.create_dataset('time', time.shape, data=time)
    f.close()
