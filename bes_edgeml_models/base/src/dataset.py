import argparse
import copy
import logging
import traceback
import h5py
from typing import Union
from pathlib import Path

import torch
import numpy as np

try:
    from ..options.train_arguments import TrainArguments
    from ..src import utils
except ImportError:
    from bes_edgeml_models.base.options.train_arguments import TrainArguments
    from bes_edgeml_models.base.src import utils


class ELMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args: argparse.Namespace,
        signals: np.ndarray,
        labels: np.ndarray,
        sample_indices: np.ndarray,
        window_start: np.ndarray,
        logger: Union[logging.getLogger, None] = None,
    ):
        """PyTorch dataset class to get the ELM data and corresponding labels
        according to the sample_indices. The signals are grouped by `signal_window_size`
        which stacks the time data points and return a data chunk of size:
        (`signal_window_sizex8x8`). The dataset also returns the label which
        corresponds to the label of the last time step of the chunk.

        Args:
        -----
            args (argparse.Namespace): Argparse object containing command line args.
            signals (np.ndarray): BES signals obtained after data preprocessing.
            labels (np.ndarray): Corresponding targets.
            sample_indices (np.ndarray): Indices of the inputs obtained after oversampling.
            window_start (np.ndarray): Start index of each ELM event.
            logger (logging.getLogger): Logger object to log the dataset creation process.
            phase (str): Dataset creation phase - 'Training', 'Validation' or 'Testing'. Defaults to 'Training'.
        """
        self.args = args
        self.signals = signals
        self.labels = labels
        self.sample_indices = sample_indices
        self.window_start = window_start
        if logger is not None:
            self.logger = logger
            self.logger.info(f"------>  Creating pytorch dataset")
            self.logger.info(f"  Signals shape: {signals.shape}")
            self.logger.info(f"  Labels shape: {labels.shape}")
            self.logger.info(f"  Sample indices shape: {sample_indices.shape}")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx: int):
        time_idx = self.sample_indices[idx]
        signal_window = self.signals[
            time_idx : time_idx + self.args.signal_window_size
        ]
        label = self.labels[
            time_idx
            + self.args.signal_window_size
            + self.args.label_look_ahead
            - 1
        ]
        if self.args.data_preproc == "gradient":
            signal_window = np.transpose(signal_window, axes=(3, 0, 1, 2))
        elif self.args.data_preproc == "rnn":
            signal_window = signal_window.reshape(
                self.args.signal_window_size, -1
            )
        else:
            signal_window = signal_window[np.newaxis, ...]
        signal_window = torch.as_tensor(signal_window, dtype=torch.float32)
        label = torch.as_tensor(label)

        return signal_window, label

class MultiSourceDataset(torch.utils.data.Dataset):

    def __init__(self,
                 args: argparse.Namespace,
                 logger: Union[logging.Logger, None] = None,
                 ):
        self.args = args
        self.logger = logger

        assert Path(self.args.input_data_dir).exists()
        self.logger.info(f'Loading files from {self.args.input_data_dir}')

        self.shot_nums, self.input_files = self._retrieve_filepaths()
        self.logger.info(f'Found {len(self.input_files)} files!')

        # Some flags for operations and checks
        self.open_ = False
        self.istrain_ = False
        self.istest_ = False
        self.isvalid_ = False
        self.frac_ = 1

        self.f_lengths = self._get_f_lengths()
        self.valid_indices = np.cumsum(np.concatenate((np.array([0]), self.f_lengths)))[:-1]

        self.signals = None
        self.labels = None
        self.hf_opened = None

    def __len__(self):
        return int(sum(self.f_lengths))

    def __getitem__(self, index: int):
        if self.args.dataset_to_ram:
            return self._get_from_ram(index)
        else:
            return self._get_from_hdf5(index)

    def __enter__(self):
        if not self.args.dataset_to_ram:
            self.open()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self.open_:
            self.logger.info('Closing all open datasets')
            self.close()
            if exc_type is not None:
                traceback.print_exception(exc_type, exc_value, tb)
                # return False # uncomment to pass exception through
        return

    def open(self):
        """
        Open all the datasets in self.args.data_dir for access.
        """
        self.open_ = True
        hf_opened = []
        for f in self.input_files:
            hf_opened.append(h5py.File(f, 'r'))
        self.hf_opened = hf_opened

        return self

    def close(self):
        """
        Close all the data sets previously opened.
        :return:
        """
        if self.open_:
            self.open_ = False
            self.logger.info('Closing all open hdf5 files.')
            for f in self.hf_opened:
                f.close()
        self.hf_opened = None
        return

    def _get_from_ram(self, index):

        hf = self.signals[np.nonzero(self.valid_indices <= index[0])[0][-1]]
        hf_labels = self.labels[np.nonzero(self.valid_indices <= index[0])[0][-1]]

        idx_offset = self.valid_indices[(self.valid_indices <= index[0])][-1].astype(int)  # Adjust index relative to specific HDF5
        hf_index = [i - idx_offset + self.args.signal_window_size for i in index]
        hf_index = list(range(hf_index[0] - self.args.signal_window_size + 1, hf_index[0])) + hf_index

        signal_windows = self._roll_window(hf[hf_index], self.args.signal_window_size, self.args.batch_size)
        labels = hf_labels[hf_index[-self.args.batch_size:]]

        return torch.tensor(signal_windows).unsqueeze(1), torch.tensor(labels)

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

    def _set_state(self, state: str):
        self.istrain_ = False
        self.istest_ = False
        self.isvalid_ = False
        if state == 'train':
            self.istrain_ = True
            self.frac_ = 1 - (self.args.fraction_valid + self.args.fraction_test)
        elif state == 'valid':
            self.isvalid_ = True
            self.frac_ = self.args.fraction_valid
        elif state == 'test':
            self.istest_ = True
            self.frac_ = self.args.fraction_test
        else:
            pass

    def train_test_split(self, test_frac: float, seed=None):
        """
        Splits full dataset into train and test sets. Returns copies of self.
        :param test_frac: Fraction of dataset for test set.
        :param seed: Numpy random seed. Default None.
        :return: train_set, test_set
        :rtype: tuple(TurbulenceDataset, TurbulenceDataset)
        """
        np.random.seed(seed)
        shots_files = np.array(list(zip(self.shot_nums, self.input_files)))
        test_idx, train_idx = [0], [0]
        if len(shots_files) != 1:
            sf_idx = np.arange(len(shots_files), dtype=np.int32)
            n_test = int(np.floor(len(shots_files) * test_frac))
            n_test = n_test if n_test else 1
            test_idx = np.random.choice(sf_idx, n_test, replace=False)
            train_idx = np.array([i for i in sf_idx if i not in test_idx], dtype=np.int32)

        test_set = shots_files[test_idx]
        train_set = shots_files[train_idx]

        train = copy.deepcopy(self)
        train._set_state('train')
        train.shot_nums = [i[0] for i in train_set]
        train.input_files = [i[1] for i in train_set]
        train.f_lengths = train._get_f_lengths()
        train.valid_indices = np.cumsum(np.concatenate((np.array([0]), train.f_lengths)))[:-1]

        test = copy.deepcopy(self)
        test._set_state('valid')
        test.shot_nums = [i[0] for i in test_set]
        test.input_files = [i[1] for i in test_set]
        test.f_lengths = test._get_f_lengths()
        test.valid_indices = np.cumsum(np.concatenate((np.array([0]), test.f_lengths)))[:-1]

        return train, test




if __name__=="__main__":
    arg_list = [
        '--use_all_data', 
    ]
    args = TrainArguments().parse(arg_list=arg_list)
    LOGGER = utils.get_logger(script_name=__name__)
    data_cls = utils.create_data_class(args.data_preproc)
    data_obj = data_cls(args, LOGGER)
    elm_indices, all_data = data_obj.get_data(verbose=True)
    all_dataset = ELMDataset(
        args, *all_data[0:4], logger=LOGGER, phase="training"
    )
