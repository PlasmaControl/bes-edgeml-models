"""
Data class to package BES data for training using PyTorch without any 
modifications and transformations.
"""
from typing import Tuple

import numpy as np
import h5py

try:
    from .base_data import BaseData
    from ..options.train_arguments import TrainArguments
    from ..src import utils
except ImportError:
    from elm_prediction.data_preprocessing.base_data import BaseData
    from elm_prediction.options.train_arguments import TrainArguments
    from elm_prediction.src import utils


class UnprocessedData(BaseData):
    def _preprocess_data(
        self,
        elm_indices: np.ndarray = None,
        shuffle_sample_indices: bool = False,
        verbose = False,
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
        with h5py.File(self.datafile, 'r') as hf:
            for elm_index in elm_indices:
                elm_key = f"{elm_index:05d}"
                if verbose:
                    self.logger.info(f' ELM index {elm_index}')
                elm_event = hf[elm_key]
                _signals = np.array(elm_event["signals"], dtype=np.float32)
                # transposing so that the time dimension comes forward
                _signals = np.transpose(_signals, (1, 0)).reshape(-1, 8, 8)
                if self.args.automatic_labels:
                    _labels = np.array(elm_event["automatic_labels"], dtype=np.float32)
                else:
                    try:
                        _labels = np.array(elm_event["labels"], dtype=np.float32)
                    except KeyError:
                        _labels = np.array(elm_event["manual_labels"], dtype=np.float32)

                if self.args.normalize_data:
                    _signals = _signals.reshape(-1, 64)
                    _signals[:, :32] = _signals[:, :32] / np.max(_signals[:, :32])
                    _signals[:, 32:] = _signals[:, 32:] / np.max(_signals[:, 32:])
                    _signals = _signals.reshape(-1, 8, 8)

                if self.args.truncate_inputs:
                    active_elm_indices = np.where(_labels > 0)[0]
                    elm_end_index = active_elm_indices[-1] + self.args.truncate_buffer
                    _signals = _signals[:elm_end_index, ...]
                    _labels = _labels[:elm_end_index]

                # if len(_labels) < 2000:
                #     continue
                # get all the allowed indices till current time step
                result = self._get_valid_indices(
                    _signals=_signals,
                    _labels=_labels,
                    window_start_indices=window_start,
                    elm_start_indices=elm_start,
                    elm_stop_indices=elm_stop,
                    valid_t0=valid_t0,
                    labels=labels,
                    signals=signals,
                    verbose=verbose,
                )
                if result is None:
                    # insufficient pre-ELM period, continue
                    continue
                (
                    signals,
                    labels,
                    valid_t0,
                    window_start,
                    elm_start,
                    elm_stop,
                ) = result

        # valid indices for data sampling
        sample_indices = np.arange(valid_t0.size, dtype="int")
        sample_indices = sample_indices[valid_t0 == 1]

        if verbose:
            shifted_sample_indices = (
                sample_indices 
                + self.args.signal_window_size
                + self.args.label_look_ahead
                -1
                )
            sampled_labels = labels[ shifted_sample_indices ]
            n_active_elm = np.count_nonzero(sampled_labels)
            n_inactive_elm = np.count_nonzero(sampled_labels-1)
            self.logger.info(" Dataset summary")
            self.logger.info(f"  Count of non-ELM labels: {n_inactive_elm}")
            self.logger.info(f"  Count of ELM labels: {n_active_elm}")
            self.logger.info(f"  Ratio: {n_active_elm/n_inactive_elm:.3f}")

        if shuffle_sample_indices:
            np.random.shuffle(sample_indices)

        self.logger.info(
            "Data tensors -> signals, labels, sample_indices, window_start_indices:"
        )
        for tensor in [
            signals,
            labels,
            sample_indices,
            window_start,
        ]:
            tmp = f" shape {tensor.shape}, dtype {tensor.dtype},"
            tmp += f" min {np.min(tensor):.3f}, max {np.max(tensor):.3f}"
            if hasattr(tensor, "device"):
                tmp += f" device {tensor.device[-5:]}"
            self.logger.info(tmp)
        return signals, labels, sample_indices, window_start, elm_indices


if __name__=="__main__":
    arg_list = [
        '--use_all_data', 
        '--label_look_ahead', '400',
    ]
    args = TrainArguments().parse(verbose=True, arg_list=arg_list)
    LOGGER = utils.get_logger(script_name=__name__)
    data_cls = utils.create_data_class(args.data_preproc)
    data_obj = data_cls(args, LOGGER)
    elm_indices, all_data = data_obj.get_data(verbose=True)
