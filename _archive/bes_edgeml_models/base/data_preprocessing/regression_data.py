"""
Data class to package BES data for training with regression algorithm.
changes target variable form class based to time based.
"""
from typing import Tuple

import numpy as np

try:
    from .unprocessed_data import UnprocessedData
except ImportError:
    from models.bes_edgeml_models.data_preprocessing.unprocessed_data import UnprocessedData


class RegressionData(UnprocessedData):

    def _get_valid_indices(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        verbose=False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            labels (torch.Tensor, optional): Tensor containing the labels of the ELM
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
        active_elm_indices = np.nonzero(labels == 1)[0]
        active_elm_start_index = active_elm_indices[0]

        # concat on axis 0 (time dimension)
        valid_t0 = np.ones(active_elm_start_index-1, dtype=np.int32)
        valid_t0[-self.args.signal_window_size + 1:] = 0
        labels = np.arange(active_elm_start_index, 1, -1, dtype=float)
        signals = signals[:active_elm_start_index-1, :, :]

        if self.args.regression == 'log':
            if np.any(labels == 0):
                assert False
            labels = np.log10(labels, dtype=float)

        if verbose:
            self.logger.info(f'  Total time points {labels.size}')
            self.logger.info(f'  Pre-ELM time points {active_elm_indices[0]}')
            self.logger.info(f'  Active ELM time points {active_elm_indices.size}')
            self.logger.info(f'  Post-ELM time points {labels.size - active_elm_indices[-1]-1}')
            self.logger.info(f'  Cound valid t0: {np.count_nonzero(valid_t0)}')
            self.logger.info(f'  Cound invalid t0: {np.count_nonzero(valid_t0-1)}')

        return (signals, labels, valid_t0)
