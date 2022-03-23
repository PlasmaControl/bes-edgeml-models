"""
Data class to package BES data for training with regression algorithm.
changes target variable form class based to time based.
"""
from typing import Tuple

import numpy as np
import h5py

try:
    from .unprocessed_data import UnprocessedData
except ImportError:
    from unprocessed_data import UnprocessedData

class RegressionData(UnprocessedData):

    def _get_valid_indices(
        self,
        _signals: np.ndarray,
        _labels: np.ndarray,
        window_start_indices: np.ndarray = None,
        elm_start_indices: np.ndarray = None,
        elm_stop_indices: np.ndarray = None,
        valid_t0: np.ndarray = None,
        labels: np.ndarray = None,
        signals: np.ndarray = None,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:

        """Helper function to concatenate the signals and labels for the ELM events
                for a given mode. It also creates allowed indices to sample from with respect
                to signal window size and label look ahead. See the README to know more
                about it.

                Args:
                -----
                    _signals (np.ndarray): NumPy array containing the inputs.
                    _labels (np.ndarray): NumPy array containing the labels.
                    window_start_indices (np.ndarray, optional): Array containing the
                        start indices of the ELM events till (t-1)th time data point. Defaults
                        to None.
                    elm_start_indices (np.ndarray, optional): Array containing the start
                        indices of the active ELM events till (t-1)th time data point.
                        Defaults to None.
                    elm_stop_indices (np.ndarray, optional): Array containing the end
                        vertices of the active ELM events till (t-1)th time data point.
                        Defaults to None.
                    valid_t0 (np.ndarray, optional): Array containing all the allowed
                        vertices of the ELM events till (t-1)th time data point. Defaults
                        to None.
                    labels (torch.Tensor, optional): Tensor containing the targets of the ELM
                        events till (t-1)th time data point. Defaults to None.
                    signals (torch.Tensor, optional): Tensor containing the input signals of
                        the ELM events till (t-1)th time data point. Defaults to None.

                Returns:
                --------
                    Tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray ]: Tuple containing
                        signals, labels, valid_t0, start and stop indices appended with current
                        time data point.
                """
        # indices for active elm times in each elm event
        active_elm_indices = np.nonzero(_labels >= 0.5)[0]

        # `t0` is first index (or earliest time, or trailing time point) for signal window
        # `_valid_t0` denotes valid `t0` time points for signal window
        # initialize to zeros
        _valid_t0 = np.zeros(active_elm_indices[0], dtype=np.int32)
        # largest `t0` index with signal window in pre-ELM period
        largest_t0 = active_elm_indices[0] - self.args.signal_window_size
        if largest_t0 < 0:
            return None
        # `t0` time points up to `largest_t0` are valid
        _valid_t0[:largest_t0 + 1] = 1

        # # `t0` within sws+la of end are invalid
        # _valid_t0 = np.ones(_labels.shape, dtype=np.int32)
        # sws_plus_la = self.args.signal_window_size + self.args.label_look_ahead
        # _valid_t0[ -sws_plus_la + 1 : ] = 0

        if signals is None:
            # lists for elm event start, active elm start, active elm stop
            window_start_indices = np.array([0])
            elm_start_indices = active_elm_indices[0]
            elm_stop_indices = active_elm_indices[0]
            # concat valid_t0, signals, and labels
            valid_t0 = _valid_t0
            signals = _signals[:active_elm_indices[0]]
            labels = np.arange(active_elm_indices[0], 0, -1)
        else:
            # append lists for elm event start, active elm start, active elm stop
            last_index = len(labels) - 1
            window_start_indices = np.append(window_start_indices, last_index + 1)
            elm_start_indices = np.append(elm_start_indices, active_elm_indices[0] + last_index + 1)
            elm_stop_indices = np.append(elm_stop_indices, active_elm_indices[0] + last_index + 1)
            # concat on axis 0 (time dimension)
            valid_t0 = np.concatenate([valid_t0, _valid_t0])
            signals = np.concatenate([signals, _signals[:active_elm_indices[0]]], axis=0)
            labels = np.concatenate([labels, np.arange(active_elm_indices[0], 0, -1)], axis=0)

        if self.args.regress_log:
            labels = np.log(labels)

        return (signals, labels, valid_t0, window_start_indices, elm_start_indices, elm_stop_indices,)