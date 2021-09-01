import argparse
import logging
from typing import Callable

import torch
import numpy as np
import albumentations as A


class ELMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args: argparse.Namespace,
        signals: np.ndarray,
        labels: np.ndarray,
        sample_indices: np.ndarray,
        window_start: np.ndarray,
        logger: logging.getLogger,
        transform: Callable = None,
        phase: str = "training",
    ):
        """PyTorch dataset class to get the ELM data and corresponding labels
        according to the sample_indices. The signals are grouped by `signal_window_size`
        which stacks the time data points and return a data chunk of size:
        (`signal_window_sizex8x8`). The dataset also returns the label which
        corresponds to the label of the last time step of the chunk.

        Args:
        -----
            signals (np.ndarray): Input data of size 8x8.
            labels (np.ndarray): Corresponding targets.
            sample_indices (np.ndarray): Indices of the inputs obtained after
                oversampling.
            window_start (np.ndarray): Start index of each ELM event.
            signal_window_size (int): Number of time data points to be used for
                stacking.
            label_look_ahead (int): Label look ahead to find which time step label
                is to used.
            stack_elm_events (bool): Whether to use stacked ELM events. It stitches
                the input 3d-tensor together to get a 2d tensor representation on
                which larger CNNs can be trained. Defaults to False.
        """
        self.args = args
        self.signals = signals
        self.labels = labels
        self.sample_indices = sample_indices
        self.window_start = window_start
        self.transform = transform
        self.logger = logger
        self.logger.info("-" * 40)
        self.logger.info(f" Creating pytorch dataset for {phase} ")
        self.logger.info("-" * 40)
        self.logger.info(f"Signals shape: {signals.shape}")
        self.logger.info(f"Labels shape: {labels.shape}")
        self.logger.info(f"Sample indices shape: {sample_indices.shape}")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx: int):
        elm_idx = self.sample_indices[idx]
        signal_window = self.signals[
            elm_idx : elm_idx + self.args.signal_window_size
        ]
        label = self.labels[
            elm_idx
            + self.args.signal_window_size
            + self.args.label_look_ahead
            - 1
        ].astype("int")
        if self.args.stack_elm_events:
            if self.args.signal_window_size == 8:
                signal_window = np.hsplit(
                    np.concatenate(signal_window, axis=-1), 2
                )
                signal_window = np.concatenate(signal_window)
            elif self.args.signal_window_size == 16:
                signal_window = np.hsplit(
                    np.concatenate(signal_window, axis=-1), 4
                )
                signal_window = np.concatenate(signal_window)
            else:
                raise ValueError(
                    f"Expected signal window size as 8 or 16 but got {self.args.signal_window_size}."
                )
            if self.transform:
                signal_window = self.transform(image=signal_window)["image"]
        if self.args.add_noise:
            noise = np.random.normal(
                loc=self.args.mu,
                scale=self.args.sigma,
                size=signal_window.shape,
            )
            signal_window += noise
        if self.args.use_gradients:
            signal_window = np.transpose(signal_window, axes=(3, 0, 1, 2))
        else:
            signal_window = signal_window[np.newaxis, ...]
        signal_window = torch.as_tensor(signal_window, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.long)

        return signal_window, label


def get_transforms(args):
    return A.Compose([A.Resize(args.size, args.size)])
