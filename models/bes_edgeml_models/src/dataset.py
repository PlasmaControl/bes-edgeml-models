import argparse
import logging
from typing import Union

import torch
import numpy as np

try:
    from ..options.train_arguments import TrainArguments
    from ..src import utils
except ImportError:
    from models.bes_edgeml_models.options.train_arguments import TrainArguments
    from models.bes_edgeml_models.src import utils


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
