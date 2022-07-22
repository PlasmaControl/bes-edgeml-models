import numpy as np
import torch


class ELM_Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        signals: np.ndarray, 
        labels: np.ndarray, 
        sample_indices: np.ndarray, 
        window_start: np.ndarray,
        signal_window_size: int,
        label_look_ahead: int = 0,  # =0 for time-to-ELM regression; >=0 for classification prediction
    ) -> None:
        self.signals = signals
        self.labels = labels
        self.sample_indices = sample_indices
        self.window_start = window_start
        self.signal_window_size = signal_window_size
        self.label_look_ahead = label_look_ahead

    def __len__(self):
        return self.sample_indices.size

    def __getitem__(self, idx: int):
        time_idx = self.sample_indices[idx]
        # BES signal window data
        signal_window = self.signals[
            time_idx : time_idx + self.signal_window_size
        ]
        signal_window = signal_window[np.newaxis, ...]
        signal_window = torch.as_tensor(signal_window, dtype=torch.float32)
        # label for signal window
        label = self.labels[
            time_idx
            + self.signal_window_size
            + self.label_look_ahead
            - 1
        ]
        label = torch.as_tensor(label)

        return signal_window, label
