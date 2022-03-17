"""
Data class to package BES data for training using PyTorch
"""
import os
import logging
import argparse
from typing import Tuple, Union

import h5py
import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn import model_selection


class BaseData:
    def __init__(
        self,
        args: argparse.Namespace,
        logger: logging.getLogger,
        # datafile: str = None,
    ):
        """Helper class that takes care of all the data preparation steps: reading
        the HDF5 file, split all the ELM events into training, validation and test
        sets, upsample the data to reduce class imbalance and create a sample signal
        window.

        Args:
        -----
            datafile (str, optional): Path to the input datafile. Defaults to None.

        """
        self.args = args
        self.datafile = self.args.input_data_file
        assert(os.path.exists(self.datafile))

        self.logger = logger
        self.df = pd.DataFrame()

        self.logger.info(f"-------->  Data file: {self.datafile}")

        # get ELM indices from datafile
        with h5py.File(self.datafile, "r") as hf:
            self.elm_indices = np.array(
                [ int(key) for key in hf ], 
                dtype=np.int32,
                )
            count = sum( [ hf[key]['labels'].shape[0] for key in hf ] )

        self.logger.info(f"  Number of ELM events: {len(self.elm_indices)}")
        self.logger.info(f"  Total time frames: {count}")

        if self.args.data_preproc == "automatic_labels":
            try:
                csv_path = f"outputs/signal_window_{args.signal_window_size}/label_look_ahead_{args.label_look_ahead}/roc"
                self.labels_df = pd.read_csv(
                    os.path.join(
                        csv_path,
                        f"automatic_labels_df_sws_{args.signal_window_size}_{args.label_look_ahead}.csv",
                    )
                )
                self.labels_df["elm_event"] = self.labels_df["elm_event"].apply(
                    lambda x: f"{x:05d}"
                )
                print(self.labels_df.info())
            except FileNotFoundError:
                print("CSV file containing the automatic labels not found.")

    def get_data(
        self,
    ) -> Union[
        Tuple[ndarray, Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]],
        Tuple[
            Tuple[ndarray, ndarray, ndarray, ndarray, ndarray],
            Tuple[ndarray, ndarray, ndarray, ndarray, ndarray],
            Tuple[ndarray, ndarray, ndarray, ndarray, ndarray],
        ],
    ]:
        """Method to create data for training, validation and testing.

        Args:
        -----
            fold (int, optional): Fold index for K-fold cross-validation between
                [0-num_folds). Must be passed when `kfold=True`. Defaults to None.
            shuffle_sample_indices (bool, optional): Whether to shuffle the sample
                indices. Defaults to False.

        Returns:
        --------
            Tuple: Tuple containing data for training, validation and test sets.
        """
        train_data = None
        validation_data = None
        test_data = None
        all_data = None
        self.logger.info("  Reading ELM events and creating datasets")
        if self.args.use_all_data:
            all_data = self._preprocess_data(
                self.elm_indices,
                shuffle_sample_indices=True,
            )
            output = (self.elm_indices, all_data)
        else:
            training_elms, validation_elms, test_elms = self._partition_elms()
            self.logger.info("-------> Creating training data")
            train_data = self._preprocess_data(
                training_elms,
                shuffle_sample_indices=True,
            )
            self.logger.info("-------> Creating validation data")
            validation_data = self._preprocess_data(
                validation_elms,
                shuffle_sample_indices=False,
            )
            self.logger.info("--------> Creating test data")
            test_data = self._preprocess_data(
                test_elms,
                shuffle_sample_indices=False,
            )

            if self.args.balance_data and (self.args.data_preproc != 'regression'):
                train_data, validation_data, test_data = self._balance_data(train_data,
                                                                            validation_data,
                                                                            test_data)

            output = (train_data, validation_data, test_data)

        return output

    def _partition_elms(
        self, 
        # max_elms: int = None,
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """Partition all the ELM events into training, validation and test indices.
        Training and validation sets are created based on simple splitting with
        validation set being `fraction_validate` of the training set or by K-fold
        cross-validation.

        Args:
        -----
            max_elms (int, optional): Maximum number of ELM events to be used.
                Defaults to None (Take the entire data).
            fold (int, optional): Fold index for K-fold cross-validation. Defaults
                to None.

        Returns:
        --------
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing training,
                validation and test ELM indices.
        """
        # limit the data according to the max number of events passed
        if self.args.max_elms is not None and self.args.max_elms != -1:
            self.logger.info(f"  Limiting data read to {self.args.max_elms} ELM events.")
            n_elms = self.args.max_elms
        else:
            n_elms = len(self.elm_indices)

        # split the data into train, validation and test sets
        training_elms, test_validate_elms = model_selection.train_test_split(
            self.elm_indices[:n_elms],
            test_size=self.args.fraction_test + self.args.fraction_valid,
            shuffle=True,
            random_state=self.args.seed,
        )
        test_elms, validation_elms = model_selection.train_test_split(
            test_validate_elms,
            test_size=self.args.fraction_valid / (self.args.fraction_test + self.args.fraction_valid),
            shuffle=True,
            random_state=self.args.seed,
        )
        self.logger.info(f"  Training ELM events: {training_elms.size}")
        self.logger.info(f"  Validation ELM events: {validation_elms.size}")
        self.logger.info(f"  Test ELM events: {test_elms.size}")

        return training_elms, validation_elms, test_elms

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
        active_elm_indices = np.nonzero(_labels >= 0.5)[0]

        # `t0` is first index (or earliest time, or trailing time point) for signal window
        # `_valid_t0` denotes valid `t0` time points for signal window
        # initialize to zeros
        _valid_t0 = np.zeros(_labels.shape, dtype=np.int32)
        # largest `t0` index with signal window in pre-ELM period
        largest_t0 = active_elm_indices[0] - self.args.signal_window_size
        if largest_t0 < 0:
            return None
        # `t0` time points up to `largest_t0` are valid
        _valid_t0[:largest_t0+1] = 1

        # # `t0` within sws+la of end are invalid
        # _valid_t0 = np.ones(_labels.shape, dtype=np.int32)
        # sws_plus_la = self.args.signal_window_size + self.args.label_look_ahead
        # _valid_t0[ -sws_plus_la + 1 : ] = 0

        if signals is None:
            # lists for elm event start, active elm start, active elm stop
            window_start_indices = np.array([0])
            elm_start_indices = active_elm_indices[0]
            elm_stop_indices = active_elm_indices[-1]
            # concat valid_t0, signals, and labels
            valid_t0 = _valid_t0
            signals = _signals
            labels = _labels
        else:
            # append lists for elm event start, active elm start, active elm stop
            last_index = len(labels) - 1
            window_start_indices = np.append(
                window_start_indices, 
                last_index + 1
            )
            elm_start_indices = np.append(
                elm_start_indices, 
                active_elm_indices[0] + last_index + 1
            )
            elm_stop_indices = np.append(
                elm_stop_indices, 
                active_elm_indices[-1] + last_index + 1
            )
            # concat on axis 0 (time dimension)
            valid_t0 = np.concatenate([valid_t0, _valid_t0])
            signals = np.concatenate([signals, _signals], axis=0)
            labels = np.concatenate([labels, _labels], axis=0)

        return (
            signals,
            labels,
            valid_t0,
            window_start_indices,
            elm_start_indices,
            elm_stop_indices,
        )

    def _balance_data(self, train, valid, test):

        clip_type = self.args.balance_data

        self.logger.info("-" * 40)
        self.logger.info(f"  Balancing data - {clip_type}")
        self.logger.info("-" * 40)

        old = [train, valid, test]
        new = [None, None, None]

        for x, old_set in enumerate(old):
            signals, labels, idx, start_idx = old_set
            clipped = [[], [], []]
            for j, i in enumerate(start_idx, start=1):

                j = start_idx[j] if j != len(start_idx) else None

                elm_event_signals = signals[i:j]
                elm_event_labels = labels[i:j]

                n_elms = len(elm_event_labels[elm_event_labels == 1])
                n_pelms = len(elm_event_labels[elm_event_labels == 0])
                diff = n_pelms - n_elms

                if clip_type == 'clip_outside':
                    _signals = elm_event_signals[diff:] if diff >= 0 else elm_event_signals[:diff]
                    _labels = elm_event_labels[diff:] if diff >= 0 else elm_event_labels[:diff]
                elif clip_type == 'clip_inside':
                    slc = np.s_[n_elms:-n_elms] if diff >= 0 else np.s_[n_pelms:-n_pelms]
                    _signals = np.delete(elm_event_signals, slc, axis=0)
                    _labels = np.delete(elm_event_labels, slc, axis=0)
                elif clip_type == 'clip_even':
                    (f, t) = (n_pelms // n_elms, n_pelms % n_elms) if diff >= 0 else (
                    n_elms // n_pelms, n_elms % n_pelms)
                    _signals = np.concatenate((elm_event_signals[t:n_pelms:f], elm_event_signals[n_pelms:]))
                    _labels = np.concatenate((elm_event_labels[t:n_pelms:f], elm_event_labels[n_pelms:]))

                clipped[0].extend(_signals)
                clipped[1].extend(_labels)
                clipped[2].extend(range(len(_labels) - (self.args.signal_window_size - 1)))

            new_start_idxs = np.pad((np.diff(np.array([0] + clipped[1])) == -1).nonzero()[0], (1, 0))

            new[x] = (np.array(clipped[0]), np.array(clipped[1]), np.array(clipped[2]), new_start_idxs)

        return new

    def _preprocess_data(
        self,
        elm_indices: np.ndarray = None,
        shuffle_sample_indices: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError
