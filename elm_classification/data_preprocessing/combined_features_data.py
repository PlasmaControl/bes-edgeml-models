"""
Data class to package BES data for training using PyTorch after combining 
features from raw unprocessed, wavelet and fast fourier transform.
"""
from typing import Tuple

import numpy as np
import pywt
from scipy.fft import rfft

try:
    from .base_data import BaseData
except ImportError:
    from base_data import BaseData


class CombinedFeaturesData(BaseData):
    def _preprocess_data(
        self,
        elm_indices: np.ndarray = None,
        shuffle_sample_indices: bool = False,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        signals = None
        window_start = None
        elm_start = None
        elm_stop = None
        valid_t0 = []
        labels = []

        # get ELM indices from the data file if not provided
        if elm_indices is None:
            elm_indices = self.elm_indices

        # iterate through all the ELM indices
        for elm_index in elm_indices:
            elm_key = f"{elm_index:05d}"
            elm_event = self.hf[elm_key]
            _signals = np.array(elm_event["signals"], dtype=np.float32)

            # transposing so that time dimension comes forward
            _signals = np.transpose(_signals, (1, 0)).reshape(-1, 8, 8)
            _labels = np.array(elm_event["labels"], dtype=np.float32)

            if self.args.truncate_inputs:
                active_elm_indices = np.where(_labels > 0)[0]
                elm_end_index = (
                    active_elm_indices[-1] + self.args.truncate_buffer
                )
                _signals = _signals[:elm_end_index, ...]
                _labels = _labels[:elm_end_index]
            if len(_labels) < 2000:
                continue
            else:
                # apply continuous wavelet transforms to the signal first
                coeffs = pywt.wavedec(
                    _signals, wavelet="db2", mode="symmetric", axis=0
                )
                uthresh = 1
                coeffs[1:] = (
                    pywt.threshold(i, value=uthresh, mode="hard")
                    for i in coeffs[1:]
                )
                _cwt_signals = pywt.waverec(
                    coeffs, wavlete="db2", mode="symmetric", axis=0
                )

                # apply fast Fourier transform to the signal
                _fft_signals = rfft(_signals, axis=0)

                if self.args.normalize_data:
                    _signals = _signals.reshape(-1, 64)
                    _cwt_signals = _cwt_signals.reshape(-1, 64)
                    _fft_signals = _fft_signals.reshape(-1, 64)
