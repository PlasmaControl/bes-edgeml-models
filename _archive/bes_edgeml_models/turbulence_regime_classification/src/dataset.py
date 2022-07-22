import argparse
import logging
import re
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch

from bes_edgeml_models.base.src.dataset import MultiSourceDataset


class TurbulenceDataset(MultiSourceDataset):
    def __init__(self,
                 args: argparse.Namespace,
                 logger: Union[logging.Logger, None] = None,
                 ):
        """PyTorch dataset class to get the ELM data and corresponding labels
        according to the sample_indices. The signals are grouped by `signal_window_size`
        which stacks the time data points and return a data chunk of size:
        (`signal_window_sizex8x8`). The dataset also returns the label which
        corresponds to the label of the last time step of the chunk. Implements weak shuffling,
        i.e. each batch is sampled randomly, however, the data points within a batch are contiguous.

        :param args: Argparse object containing command line args.
        :type args: argparse.Namespace
        :param logger: Logger object to log the dataset creation process.
        :type logger: logging.getLogger
        """

        super().__init__(args, logger)
        if not self.args.dataset_to_ram:
            # used for __getitem__ when reading from HDF5
            self.hf2np_signals = np.empty((64, self.args.batch_size + self.args.signal_window_size - 1))
            self.hf2np_labels = np.empty((self.args.batch_size,))


    def _get_from_hdf5(self, index):
        try:
            # Get correct index with respect to HDF5 file.
            hf = self.hf_opened[np.nonzero(self.valid_indices <= index[0])[0][-1]]
        except TypeError:
            raise AttributeError('HDF5 files have not been opened! Use TurbulenceDataset.open() ')

        idx_offset = self.valid_indices[(self.valid_indices <= index[0])][-1]# Adjust index relative to specific HDF5
        hf_index = [i - idx_offset + self.args.signal_window_size for i in index]
        hf_index = list(range(hf_index[0] - self.args.signal_window_size + 1, hf_index[0])) + hf_index
        hf['signals'].read_direct(self.hf2np_signals, np.s_[:, hf_index], np.s_[...])
        signal_windows = self._roll_window(self.hf2np_signals.transpose(), self.args.signal_window_size, self.args.batch_size)

        hf['labels'].read_direct(self.hf2np_labels, np.s_[hf_index[-self.args.batch_size:]], np.s_[...])

        return torch.tensor(signal_windows).unsqueeze(1), torch.tensor(self.hf2np_labels)


    def _retrieve_filepaths(self, input_dir=None):
        """
        Get filenames of all labeled files.
        :param input_dir: (optional) Change the input data directory.
        :return: all shot numbers, all shot file paths.
        :rtype: (list, list)
        """
        if input_dir:
            self.args.input_data_dir = input_dir
        dir = Path(self.args.input_data_dir) / 'labeled_datasets'
        assert dir.exists(), f'Directory {dir} does not exist. Have you made labeled datasets?'
        shots = {}
        for file in (dir.iterdir()):
            try:
                shot_num = re.findall(r'_(\d+).+.hdf5', str(file))[0]
            except IndexError:
                continue
            if shot_num not in shots.keys():
                shots[shot_num] = file

        # Keeps them in the same order for __getitem__
        self.input_files = [shots[key] for key in sorted(shots.keys())]
        self.shot_nums = list(sorted(shots.keys()))

        return self.shot_nums, self.input_files

    def _get_f_lengths(self):
        """
        Get lengths of all hdf5 files.
        :rtype: np.array
        """
        fs = []
        for f in self.input_files:
            with h5py.File(f, 'r') as ds:
                fs.append(len(ds['labels']) - self.args.signal_window_size - self.args.batch_size)
        return np.array(fs, dtype=int)

    def load_datasets(self):
        """Load datasets into RAM"""
        if self.istrain_:
            s = 'Training '
        elif self.isvalid_:
            s = 'Validation '
        else:
            s = ' '
        self.logger.info(f"Loading {s}datasets into RAM.")
        signals, labels = [], []

        self.open()
        for i, (sn, hf) in enumerate(zip(self.shot_nums, self.hf_opened)):
            print(f'\rProcessing shot {sn} ({i+1}/{len(self.shot_nums)})', end=' ')
            signals_np = np.array(hf['signals']).transpose()
            labels_np = np.array(hf['labels'])
            print(f'{signals_np.nbytes + labels_np.nbytes} bytes!')
            signals.append(signals_np)
            labels.append(labels_np)
        self.close()

        self.signals = signals
        self.labels = labels
        self.logger.info(f'{s}datasets loaded successfully.')

        return