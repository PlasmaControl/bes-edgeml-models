"""
Data class to package BES data with their gradients. These gradients are then 
added as separate channels in addition to the original signals.  
"""
from typing import Tuple

import numpy as np

from base_data import BaseData


class GradientData(BaseData):
    def _preprocess_data(
        self,
        elm_indices: np.ndarray = None,
        shuffle_sample_indices: bool = False,
        is_test_data: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Helper function to preprocess the data: reshape the input signal, use
        allowed indices to upsample the class minority labels [active ELM events].

        Args:
        -----
            elm_indices (np.ndarray, optional): ELM event indices for the corresponding
                mode. Defaults to None.
            shuffle_sample_indices (bool, optional): Whether to shuffle the sample
                indices. Defaults to False.

        Returns:
        --------
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing
                original signals, correponding labels, sample indices obtained
                after upsampling and start index for each ELM event.
        """
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
            _signals = np.array(
                elm_event["signals"], dtype=self.args.signal_dtype
            )
            # transposing so that the time dimension comes forward
            _signals = np.transpose(_signals, (1, 0)).reshape(-1, 8, 8)
            _labels = np.array(
                elm_event["labels"], dtype=self.args.signal_dtype
            )
            if self.args.normalize_data:
                _signals = _signals.reshape(-1, 64)
                _signals[:, :33] = _signals[:, :33] / 10.0
                _signals[:, 33:] = _signals[:, 33:] / 5.0
                _signals = _signals.reshape(-1, 8, 8)

            if self.args.truncate_inputs:
                active_elm_indices = np.where(_labels > 0)[0]
                elm_start_index = active_elm_indices[0]
                if is_test_data:
                    elm_end_index = active_elm_indices[-1]
                else:
                    elm_end_index = elm_start_index + 75
                _signals = _signals[:elm_end_index, ...]
                _labels = _labels[:elm_end_index]

            # calculate gradients along y(axis=0) and x(axis=1)
            # suppress time derivatives for now
            _, dsignals_dy, dsignals_dx = np.gradient(_signals)
            _, d2signals_dy2, _ = np.gradient(dsignals_dy)
            _, d2signals_dxdy, d2signals_dx2 = np.gradient(dsignals_dx)
            _signals = np.array(
                [
                    _signals,
                    dsignals_dx,
                    dsignals_dy,
                    d2signals_dx2,
                    d2signals_dy2,
                    d2signals_dxdy,
                ]
            )
            _signals = np.transpose(_signals, axes=(1, 2, 3, 0))

            # get all the allowed indices till current time step
            indices_data = self._get_valid_indices(
                _signals=_signals,
                _labels=_labels,
                window_start_indices=window_start,
                elm_start_indices=elm_start,
                elm_stop_indices=elm_stop,
                valid_t0=valid_t0,
                labels=labels,
                signals=signals,
            )
            (
                signals,
                labels,
                valid_t0,
                window_start,
                elm_start,
                elm_stop,
            ) = indices_data

        _labels = np.array(labels)

        # valid indices for data sampling
        valid_indices = np.arange(valid_t0.size, dtype="int")
        valid_indices = valid_indices[valid_t0 == 1]
        sample_indices = valid_indices
        # sample_indices = self._oversample_data(
        #     _labels, valid_indices, elm_start, elm_stop
        # )

        if shuffle_sample_indices:
            np.random.shuffle(sample_indices)

        self.logger.info(
            "Data tensors -> signals, labels, valid_indices, sample_indices, window_start_indices:"
        )
        for tensor in [
            signals,
            labels,
            valid_indices,
            sample_indices,
            window_start,
        ]:
            tmp = f" shape {tensor.shape}, dtype {tensor.dtype},"
            tmp += f" min {np.min(tensor):.3f}, max {np.max(tensor):.3f}"
            if hasattr(tensor, "device"):
                tmp += f" device {tensor.device[-5:]}"
            self.logger.info(tmp)
        return signals, labels, sample_indices, window_start


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.getcwd())
    from src import utils
    from options.base_arguments import BaseArguments

    args, _ = BaseArguments().parse()

    # create the logger object
    logger = utils.get_logger(
        script_name=__name__,
        stream_handler=True,
        # log_file=f"output_logs_{args.data_mode}.log",
    )
    data = GradientData(args, logger)
    train_data, _, _ = data.get_data()
    print(train_data)
