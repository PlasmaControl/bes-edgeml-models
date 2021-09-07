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
        datafile: str = None,
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
        self.datafile = datafile
        if self.datafile is None:
            self.datafile = os.path.join("data", self.args.input_file)
        self.logger = logger

        self.df = pd.DataFrame()
        self.elm_indices, self.hf = self._read_file()
        self.logger.info(
            f"Total frames in the whole data: {self._get_total_frames()}"
        )
        if self.args.data_preproc == "automatic_labels":
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
        # self.transition = np.linspace(
        #     0, 1, 2 * self.args.transition_halfwidth + 3
        # )

    def get_data(
        self, shuffle_sample_indices: bool = False, fold: int = None
    ) -> Union[Tuple[ndarray, Tuple[ndarray, ndarray, ndarray, ndarray]], Tuple[
        Tuple[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray, ndarray, ndarray], Tuple[
            ndarray, ndarray, ndarray, ndarray]]]:
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
        global train_data, validation_data, test_data, all_elms, all_data
        training_elms, validation_elms, test_elms = self._partition_elms(
            max_elms=self.args.max_elms, fold=fold
        )
        self.logger.info("Reading ELM events and creating datasets")
        self.logger.info("-" * 30)
        self.logger.info("  Creating training data")
        self.logger.info("-" * 30)
        if self.args.use_all_data:
            all_elms = np.concatenate(
                [training_elms, validation_elms, test_elms]
            )
            all_data = self._preprocess_data(
                all_elms,
                shuffle_sample_indices=shuffle_sample_indices,
                is_test_data=False,
            )
        else:
            train_data = self._preprocess_data(
                training_elms,
                shuffle_sample_indices=shuffle_sample_indices,
                is_test_data=False,
            )
            self.logger.info("-" * 30)
            self.logger.info("  Creating validation data")
            self.logger.info("-" * 30)
            validation_data = self._preprocess_data(
                validation_elms,
                shuffle_sample_indices=shuffle_sample_indices,
                is_test_data=False,
            )
            self.logger.info("-" * 30)
            self.logger.info("  Creating test data")
            self.logger.info("-" * 30)
            test_data = self._preprocess_data(
                test_elms,
                shuffle_sample_indices=shuffle_sample_indices,
                is_test_data=True,
            )

        self.hf.close()
        if self.hf:
            self.logger.info("File is open.")
        else:
            self.logger.info("File is closed.")

        return (
            (all_elms, all_data)
            if self.args.use_all_data
            else (
                train_data,
                validation_data,
                test_data,
            )
        )

    def _get_total_frames(self):
        count = 0
        for elm_index in self.elm_indices:
            elm_key = f"{elm_index:05d}"
            elm_event = self.hf[elm_key]
            count += np.array(elm_event["labels"]).shape[0]
        return count

    def _partition_elms(
        self, max_elms: int = None, fold: int = None
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

        Raises:
        -------
            Exception:  Throws error when `kfold` is set to True but fold index
                is not passed.

        Returns:
        --------
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing training,
                validation and test ELM indices.
        """
        # limit the data according to the max number of events passed
        if max_elms is not None and max_elms != -1:
            self.logger.info(f"Limiting data read to {max_elms} events.")
            n_elms = max_elms
        else:
            n_elms = len(self.elm_indices)

        # split the data into train, validation and test sets
        training_elms, test_elms = model_selection.train_test_split(
            self.elm_indices[:n_elms],
            test_size=self.args.fraction_test,
            shuffle=True,
            random_state=self.args.seed,
        )

        # kfold cross validation
        if self.args.kfold and fold is None:
            raise ValueError(
                f"K-fold cross validation is passed but fold index in range [0, {self.args.folds}) is not specified."
            )

        if self.args.kfold:
            self.logger.info("Using K-fold cross validation")
            self._kfold_cross_val(training_elms)
            training_elms = self.df[self.df["fold"] != fold]["elm_events"]
            validation_elms = self.df[self.df["fold"] == fold]["elm_events"]
        else:
            self.logger.info(
                "Creating training and validation datasets by simple splitting"
            )
            training_elms, validation_elms = model_selection.train_test_split(
                training_elms,
                test_size=self.args.fraction_valid,
                shuffle=True,
                random_state=self.args.seed,
            )
        self.logger.info(f"Number of training ELM events: {training_elms.size}")
        self.logger.info(
            f"Number of validation ELM events: {validation_elms.size}"
        )
        self.logger.info(f"Number of test ELM events: {test_elms.size}")

        return training_elms, validation_elms, test_elms

    def _read_file(self) -> Tuple[ndarray, h5py.File]:
        """Helper function to read a HDF5 file.

        Returns:
        --------
            Tuple[ndarray, h5py.File]: Tuple containing ELM indices and file object.
        """
        assert os.path.exists(self.datafile)
        self.logger.info(f"Found datafile: {self.datafile}")

        # get ELM indices from datafile
        hf = h5py.File(self.datafile, "r")
        self.logger.info(f"Number of ELM events in the datafile: {len(hf)}")
        elm_index = np.array([int(key) for key in hf], dtype=np.int32)
        return elm_index, hf

    def _kfold_cross_val(self, training_elms: np.ndarray) -> None:
        """Helper function to perform K-fold cross-validation.

        Args:
        -----
            training_elms (np.ndarray): Indices for training ELM events.
        """
        kf = model_selection.KFold(
            n_splits=self.args.folds, shuffle=True, random_state=self.args.seed
        )
        self.df["elm_events"] = training_elms
        self.df["fold"] = -1
        for f_, (_, valid_idx) in enumerate(kf.split(X=training_elms)):
            self.df.loc[valid_idx, "fold"] = f_

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
        # allowed indices; time data points which can be used for creating the data chunks
        _valid_t0 = np.ones(_labels.shape, dtype=np.int32)
        _valid_t0[
            -(self.args.signal_window_size + self.args.label_look_ahead) + 1 :
        ] = 0

        # indices for active elm events in each elm event
        active_elm_events = np.nonzero(_labels >= 0.5)[0]

        if signals is None:
            # initialize arrays
            window_start_indices = np.array([0])
            elm_start_indices = active_elm_events[0]
            elm_stop_indices = active_elm_events[-1]
            valid_t0 = _valid_t0
            signals = _signals
            labels = _labels
        else:
            # concat on axis 0 (time dimension)
            last_index = len(labels) - 1
            window_start_indices = np.append(
                window_start_indices, last_index + 1
            )
            elm_start_indices = np.append(
                elm_start_indices, active_elm_events[0] + last_index + 1
            )
            elm_stop_indices = np.append(
                elm_stop_indices, active_elm_events[-1] + last_index + 1
            )
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

    def _preprocess_data(
        self,
        elm_indices: np.ndarray = None,
        shuffle_sample_indices: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError
