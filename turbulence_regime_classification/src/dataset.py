import copy
import traceback
import h5py
import pandas as pd
import torch
import argparse
import logging
import numpy as np
import re
from pathlib import Path
from typing import Union
from sklearn.model_selection import train_test_split


class TurbulenceDataset(torch.utils.data.Dataset):
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

        self.args = args
        self.logger = logger

        assert Path(self.args.input_data_dir).exists()
        self.logger.info(f'Loading files from {self.args.input_data_dir}')

        self.shot_nums, self.input_files = self._retrieve_filepaths()
        self.logger.info(f'\tFound {len(self.input_files)} files!')

        self.f_lengths = self._get_f_lengths()
        self.hf_cumsum = np.cumsum(np.concatenate((np.array([0]), self.f_lengths)))[:-1]

        self.open_ = False
        self.signals = None
        self.labels = None
        if not self.args.dataset_to_ram:
            # used for __getitem__ when reading from HDF5
            self.hf2np_signals = np.empty((64, self.args.batch_size + self.args.signal_window_size - 1))
            self.hf2np_labels = np.empty((self.args.batch_size,))

        self.hf_opened = None

    def __len__(self):
        return sum(self.f_lengths)


    def __getitem__(self, index: int):

        if self.args.dataset_to_ram:
            return self._get_from_ram(index)
        else:
            return self._get_from_hdf5(index)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            # return False # uncomment to pass exception through

    def _get_from_ram(self, index):

        hf = self.signals[np.nonzero(self.hf_cumsum <= index[0])[0][-1]]
        hf_labels = self.labels[np.nonzero(self.hf_cumsum <= index[0])[0][-1]]

        idx_offset = self.hf_cumsum[(self.hf_cumsum <= index[0])][-1] - self.args.signal_window_size  # Adjust index relative to specific HDF5
        hf_index = [i - idx_offset for i in index]
        hf_index = list(range(hf_index[0] - self.args.signal_window_size + 1, hf_index[0])) + hf_index

        signal_windows = self._roll_window(hf[hf_index], self.args.signal_window_size, self.args.batch_size)
        labels = hf_labels[-self.args.batch_size:]

        return torch.tensor(signal_windows).unsqueeze(1), torch.tensor(labels)

    def _get_from_hdf5(self, index):
        try:
            # Get correct index with respect to HDF5 file.
            hf = self.hf_opened[np.nonzero(self.hf_cumsum <= index[0])[0][-1]]
        except TypeError:
            raise AttributeError('HDF5 files have not been opened! Use TurbulenceDataset.open() ')

        idx_offset = self.hf_cumsum[(self.hf_cumsum <= index[0])][-1] - self.args.signal_window_size # Adjust index relative to specific HDF5
        hf_index = [i - idx_offset for i in index]
        hf_index = list(range(hf_index[0] - self.args.signal_window_size + 1, hf_index[0])) + hf_index
        try:
            hf['signals'].read_direct(self.hf2np_signals, np.s_[:, hf_index], np.s_[...])
        except Exception as e:
             pass
        signal_windows = self._roll_window(self.hf2np_signals.transpose(), self.args.signal_window_size, self.args.batch_size)

        hf['labels'].read_direct(self.hf2np_labels, np.s_[hf_index[-self.args.batch_size:]], np.s_[...])

        return torch.tensor(signal_windows).unsqueeze(1), torch.tensor(self.hf2np_labels)

    def _roll_window(self, arr, sws, bs):
        """
        Helper function to return rolling window view of array.
        :param arr: Array to be rolled
        :type arr: np.ndarray
        :param sws: Signal window size
        :type sws: int
        :param bs: Batch size
        :type bs: int
        :return: Rolling window view of array
        :rtype: np.ndarray
        """
        return np.lib.stride_tricks.sliding_window_view(arr.view(), sws, axis=0)\
            .swapaxes(-1, 1)\
            .reshape(bs, -1, 8, 8)

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
                fs.append(len(ds['labels']) - self.args.signal_window_size + 1)
        return np.array(fs)

    def train_test_split(self, test_frac: float, seed=None):
        """
        Splits full dataset into train and test sets. Will only split by input file. Returns copies of
        class that only contain inputs from random selection of input files.
        :param test_frac: Fraction of dataset for test set.
        :param seed: Numpy random seed. Default None.
        :return: train_set, test_set
        :rtype: tuple(TurbulenceDataset, TurbulenceDataset)
        """
        np.random.seed(seed)
        shots_files = np.array(list(zip(self.shot_nums, self.input_files)))
        sf_idx = np.arange(len(shots_files))
        n_test = int(np.floor(len(shots_files) * test_frac))
        test_idx = np.random.choice(sf_idx, n_test, replace=False)
        train_idx = np.array([i for i in sf_idx if i not in test_idx])

        test_set = shots_files[test_idx]
        train_set = shots_files[train_idx]

        train = copy.deepcopy(self)
        train.shot_nums = [i[0] for i in train_set]
        train.input_files = [i[1] for i in train_set]
        train.f_lengths = train._get_f_lengths()
        train.hf_cumsum = np.cumsum(np.concatenate((np.array([0]), train.f_lengths)))[:-1]

        test = copy.deepcopy(self)
        test.shot_nums = [i[0] for i in test_set]
        test.input_files = [i[1] for i in test_set]
        test.f_lengths = test._get_f_lengths()
        test.hf_cumsum = np.cumsum(np.concatenate((np.array([0]), test.f_lengths)))[:-1]

        return train, test

    def load_datasets(self):
        self.logger.info("Loading datasets into RAM.")
        signals, labels = [], []
        for i, (sn, hf) in enumerate(zip(self.shot_nums, self.hf_opened)):
            print(f'\rProcessing shot {sn} ({i+1}/{len(self.shot_nums)})', end=' ')
            signals_np = np.array(hf['signals']).transpose()
            labels_np = np.array(hf['labels'])
            print(f'{signals_np.nbytes + labels_np.nbytes} bytes!')
            signals.append(signals_np)
            labels.append(labels_np)

        self.signals = signals
        self.labels = signals

        self.logger.info(' Datasets loaded successfully.')

        return

    def open(self):
        """
        Open all the datasets in self.args.data_dir for access.
        """
        if not self.args.dataset_to_ram:
            self.open_ = True
            inputs = []
            for f in self.input_files:
                inputs.append(h5py.File(f, 'r'))
            self.hf_opened = inputs

        return self

    def close(self):
        """
        Close all the data sets previously opened.
        :return:
        """
        self.open_ = False
        for f in self.hf_opened:
            f.close()
        return

    @staticmethod
    def make_labels(data, df_loc):
        df = pd.read_excel(df_loc)
        time = np.array(data['time'])
        signals = np.array(data['signals'])
        labels = np.zeros_like(time)
        for i, row in df.iterrows():
            tstart = row['tstart (ms)']
            tstop = row['tstop (ms)']
            label = row[[col for col in row.index if 'mode' in col]].values.argmax() + 1
            labels[np.nonzero((time > tstart) & (time < tstop))] = label
            signals = labels[np.nonzero(labels)[0]]
            labels = labels[np.nonzero(labels)[0]] - 1

        return signals.tolist(), labels.tolist()




