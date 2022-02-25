import argparse
import logging

import torch
import numpy as np


class ELMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args: argparse.Namespace,
        signals: np.ndarray,
        labels: np.ndarray,
        sample_indices: np.ndarray,
        window_start: np.ndarray,
        logger: logging.getLogger,
        phase: str = "training",
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
        self.logger = logger
        self.logger.info(f"------>  Creating pytorch dataset for {phase} ")
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
        ].astype("int")
        if self.args.data_preproc == "gradient":
            signal_window = np.transpose(signal_window, axes=(3, 0, 1, 2))
        elif self.args.data_preproc == "rnn":
            signal_window = signal_window.reshape(
                self.args.signal_window_size, -1
            )
        else:
            signal_window = signal_window[np.newaxis, ...]
        signal_window = torch.as_tensor(signal_window, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.long)

        return signal_window, label

