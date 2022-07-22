import argparse
import logging
import pickle
import re
from pathlib import Path
from typing import Union

import h5py
import numpy as np

from bes_edgeml_models.base.src.dataset import MultiSourceDataset


class VelocimetryDataset(MultiSourceDataset):
    def __init__(self,
                 args: argparse.Namespace,
                 logger: Union[logging.Logger, None] = None,
                 ):
        """PyTorch dataset class to get the ELM data and corresponding velocimetry calculations.
        The signals are grouped by `signal_window_size` which stacks the time data points
        and return a data chunk of size: (`signal_window_sizex8x8`). The dataset also returns the label which
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
            self.hf2np_vZ = np.empty((self.args.batch_size, 8, 8))
            self.hf2np_vR = np.empty((self.args.batch_size, 8, 8))

    def _get_from_hdf5(self, index):
        return NotImplementedError('Can not get from hdf5.')

    def _retrieve_filepaths(self, input_dir=None):
        """
        Get filenames of all labeled files.
        :param input_dir: (optional) Change the input data directory.
        :return: all shot numbers, all shot file paths.
        :rtype: (list, list)
        """
        if input_dir:
            self.args.input_data_dir = input_dir
        dir = Path(self.args.input_data_dir)
        assert dir.exists(), f'Directory {dir} does not exist. Have you made datasets?'
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
                fs.append(np.around(len(ds['vR']) * self.frac_) - self.args.signal_window_size - self.args.batch_size)
        assert all([l >= 0 for l in fs]), "There are not enough data points to make dataset."
        return np.array(fs)

    def save_test_data(self) -> None:
        """
        Load and save test data to pickle file at location specified in args
        Returns: None

        """

        output = {}

        self.open()
        if len(self.shot_nums) == 1:
            hf = self.hf_opened[0]
            # Load and save test data to pickle file
            n_indices = len(hf['vR'])
            i_start = np.floor((1 - self.args.fraction_test) * n_indices).astype(int)
            # Load test data
            sx_t_data = np.s_[i_start:n_indices-1]
            sx_s_t_data = np.s_[:, i_start:n_indices-1]

            arr_len = sx_t_data.stop - sx_t_data.start
            hf2np_s = np.empty((64, arr_len))
            hf2np_vR = np.empty((arr_len, 8, 8))
            hf2np_vZ = np.empty((arr_len, 8, 8))

            hf['signals'].read_direct(hf2np_s, sx_s_t_data, np.s_[...])
            hf['vR'].read_direct(hf2np_vR, sx_t_data, np.s_[...])
            hf['vZ'].read_direct(hf2np_vZ, sx_t_data, np.s_[...])
        else:
            return

        output['signals'] = hf2np_s.transpose().reshape((-1, 8, 8))
        output['vZ'] = hf2np_vZ
        output['vR'] = hf2np_vR
        with open(Path(self.args.output_dir)/'test_data.pkl', 'w+b') as f:
            pickle.dump(output, f)
        return

    def load_datasets(self):
        """Load datasets into RAM"""
        # Only used for displaying strings in console.
        if self.istrain_:
            s = 'Training '
        elif self.isvalid_:
            s = 'Validation '
        else:
            s = ' '
        self.logger.info(f"Loading {s}datasets into RAM.")
        signals, labels = [], []

        self.open()
        if len(self.shot_nums) == 1:
            hf = self.hf_opened[0]
            n_indices = len(hf['vR'])
            # Indices for start and stop of validation and test sets
            i_start = np.floor((1 - (self.args.fraction_test + self.args.fraction_valid)) * n_indices).astype(int)
            i_stop = np.floor((1 - self.args.fraction_test) * n_indices).astype(int)

            if self.isvalid_:
                sx = np.s_[i_start:i_stop]
                sx_s = np.s_[:, i_start:i_stop]
            else:
                sx = np.s_[0:i_start]
                sx_s = np.s_[:, 0:i_start]

            # read_direct is faster and more memory efficient
            arr_len = sx.stop - sx.start
            hf2np_s = np.empty((64, arr_len))
            hf2np_vR = np.empty((arr_len, 8, 8))
            hf2np_vZ = np.empty((arr_len, 8, 8))

            hf['signals'].read_direct(hf2np_s, sx_s, np.s_[...])
            hf['vR'].read_direct(hf2np_vR, sx, np.s_[...])
            hf['vZ'].read_direct(hf2np_vZ, sx, np.s_[...])
            signals.append(hf2np_s.transpose())
            labels.append(np.concatenate((hf2np_vR.reshape(-1, 64), hf2np_vZ.reshape(-1, 64)), axis=1))
        else:
            for i, (sn, hf) in enumerate(zip(self.shot_nums, self.hf_opened)):
                #TODO: This might be wrong, but we're not using it for now (June 23, 2022)
                print(f'\rProcessing shot {sn} ({i+1}/{len(self.shot_nums)})', end=' ')
                signals_np = np.array(hf['signals']).transpose()
                vr_labels_np = np.array(hf['vR']).reshape((-1, 64))
                vz_labels_np = np.array(hf['vZ']).reshape((-1, 64))
                labels_np = np.concatenate((vr_labels_np, vz_labels_np), axis=1)
                print(f'{signals_np.nbytes + labels_np.nbytes} bytes!')
                signals.append(signals_np)
                labels.append(labels_np)
        self.close()

        self.signals = signals
        self.labels = labels
        self.logger.info(f'{s}datasets loaded successfully.')

        return