"""
Data class to package BES data for training using PyTorch
"""
import os
import utils
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
from sklearn import model_selection
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


# create the logger object
LOGGER = utils.get_logger(
    script_name=__name__,
    stream_handler=False,
    log_file=f"output_logs_{config.data_mode}.log",
)


class Data:
    def __init__(
        self,
        datafile: str = None,
        fraction_validate: float = 0.2,
        fraction_test: float = 0.1,
        signal_dtype: str = "float32",
        kfold: bool = False,
        smoothen_transition: bool = False,
        balance_classes: bool = False,
    ):
        """Helper class that takes care of all the data preparation steps: reading
        the HDF5 file, split all the ELM events into training, validation and test
        sets, upsample the data to reduce class imbalance and create a sample signal
        window.

        Args:
        -----
            datafile (str, optional): Path to the input datafile. Defaults to None.
            fraction_validate (float, optional): Fraction of the total data to
                be used as a validation set. Ignored when using K-fold cross-
                validation. Defaults to 0.2.
            fraction_test (float, optional): Fraction of the total data to be
                used as test set. Defaults to 0.1.
            signal_dtype (str, optional): Datatype of the signals. Defaults to "float32".
            kfold (bool, optional): Boolean showing whether to use K-fold cross-
                validation or not. Defaults to False.
            smoothen_transition (bool, optional): Boolean showing whether to smooth
                the labels so that there is a gradual transition of the labels from
                0 to 1 with respect to the input time series. Defaults to False.
            balance_classes (bool, optional): Boolean representing whether or not
                to oversample the data to balance the label classes. Defaults to
                False.
        """
        self.datafile = datafile
        if self.datafile is None:
            self.datafile = os.path.join(config.data_dir, config.file_name)
        self.fraction_validate = fraction_validate
        self.fraction_test = fraction_test
        self.signal_dtype = signal_dtype
        self.kfold = kfold
        self.smoothen_transition = smoothen_transition
        self.balance_classes = balance_classes
        self.max_elms = config.max_elms

        self.df = pd.DataFrame()
        self.transition = np.linspace(0, 1, 2 * config.transition_halfwidth + 3)

    def get_data(
        self, shuffle_sample_indices: bool = False, fold: int = None
    ) -> Tuple:
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
        training_elms, validation_elms, test_elms = self._partition_elms(
            max_elms=config.max_elms, fold=fold
        )
        LOGGER.info("Reading ELM events and creating datasets")
        LOGGER.info("-" * 30)
        LOGGER.info("  Creating training data")
        LOGGER.info("-" * 30)
        train_data = self._preprocess_data(
            training_elms, shuffle_sample_indices=shuffle_sample_indices
        )
        LOGGER.info("-" * 30)
        LOGGER.info("  Creating validation data")
        LOGGER.info("-" * 30)
        validation_data = self._preprocess_data(
            validation_elms, shuffle_sample_indices=shuffle_sample_indices
        )
        LOGGER.info("-" * 30)
        LOGGER.info("  Creating test dataset")
        LOGGER.info("-" * 30)
        test_data = self._preprocess_data(
            test_elms, shuffle_sample_indices=shuffle_sample_indices
        )

        return train_data, validation_data, test_data

    def _preprocess_data(
        self,
        elm_indices: np.ndarray = None,
        shuffle_sample_indices: bool = False,
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
        hf = None
        valid_t0 = []
        labels = []

        # get ELM indices from the data file if not provided
        if elm_indices is None:
            elm_indices, hf = self._read_file()
        else:
            _, hf = self._read_file()

        # iterate through all the ELM indices
        for elm_index in elm_indices:
            elm_key = f"{elm_index:05d}"
            elm_event = hf[elm_key]
            _signals = np.array(elm_event["signals"], dtype=self.signal_dtype)
            # transposing so that the time dimension comes forward
            _signals = np.transpose(_signals, (1, 0)).reshape(-1, 8, 8)
            _labels = np.array(elm_event["labels"], dtype=self.signal_dtype)

            # TODO: add label smoothening

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
        sample_indices = self._oversample_data(
            _labels, valid_indices, elm_start, elm_stop
        )

        if shuffle_sample_indices:
            np.random.shuffle(sample_indices)

        LOGGER.info(
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
            LOGGER.info(tmp)

        hf.close()
        if hf:
            LOGGER.info("File is open.")
        else:
            LOGGER.info("File is closed.")
        return signals, labels, sample_indices, window_start

    def _partition_elms(
        self, max_elms: int = None, fold: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        # get ELM indices from datafile
        elm_index, _ = self._read_file()

        # limit the data according to the max number of events passed
        if max_elms is not None and max_elms != -1:
            LOGGER.info(f"Limiting data read to {max_elms} events.")
            n_elms = max_elms
        else:
            n_elms = len(elm_index)

        # split the data into train, validation and test sets
        training_elms, test_elms = model_selection.train_test_split(
            elm_index[:n_elms],
            test_size=self.fraction_test,
            shuffle=True,
            random_state=config.seed,
        )

        # kfold cross validation
        if self.kfold and fold is None:
            raise Exception(
                f"K-fold cross validation is passed but fold index in range [0, {config.folds}) is not specified."
            )

        if self.kfold:
            LOGGER.info("Using K-fold cross validation")
            self._kfold_cross_val(training_elms)
            training_elms = self.df[self.df["fold"] != fold]["elm_events"]
            validation_elms = self.df[self.df["fold"] == fold]["elm_events"]
        else:
            LOGGER.info(
                "Creating training and validation datasets by simple splitting"
            )
            training_elms, validation_elms = model_selection.train_test_split(
                training_elms, test_size=self.fraction_validate
            )
        LOGGER.info(f"Number of training ELM events: {training_elms.size}")
        LOGGER.info(f"Number of validation ELM events: {validation_elms.size}")
        LOGGER.info(f"Number of test ELM events: {test_elms.size}")

        return training_elms, validation_elms, test_elms

    def _kfold_cross_val(self, training_elms: np.ndarray) -> None:
        """Helper function to perform K-fold cross-validation.

        Args:
        -----
            training_elms (np.ndarray): Indices for training ELM events.
        """
        kf = model_selection.KFold(
            n_splits=config.folds, shuffle=True, random_state=config.seed
        )
        self.df["elm_events"] = training_elms
        self.df["fold"] = -1
        for f_, (_, valid_idx) in enumerate(kf.split(X=training_elms)):
            self.df.loc[valid_idx, "fold"] = f_

    def _read_file(self) -> Tuple[np.ndarray, h5py.File]:
        """Helper function to read a HDF5 file.

        Returns:
        --------
            Tuple[np.ndarray, h5py.File]: Tuple containing ELM indices and file object.
        """
        assert os.path.exists(self.datafile)
        LOGGER.info(f"Found datafile: {self.datafile}")

        # get ELM indices from datafile
        hf = h5py.File(self.datafile, "r")
        LOGGER.info(f"Number of ELM events in the datafile: {len(hf)}")
        elm_index = np.array([int(key) for key in hf], dtype=np.int32)
        return elm_index, hf

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
            labels (tf.Tensor, optional): Tensor containing the labels of the ELM
                events till (t-1)th time data point. Defaults to None.
            signals (tf.Tensor, optional): Tensor containing the input signals of
                the ELM events till (t-1)th time data point. Defaults to None.

        Returns:
        --------
            Tuple[ tf.Tensor, tf.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray ]: Tuple containing
                signals, labels, valid_t0, start and stop indices appended with current
                time data point.
        """
        # allowed indices; time data points which can be used for creating the data chunks
        _valid_t0 = np.ones(_labels.shape, dtype=np.int32)
        _valid_t0[
            -(config.signal_window_size + config.label_look_ahead) + 1 :
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

    def _oversample_data(
        self,
        _labels: np.ndarray,
        valid_indices: np.ndarray,
        elm_start: np.ndarray,
        elm_stop: np.ndarray,
        index_buffer: int = 20,
    ) -> np.ndarray:
        """Helper function to reduce the class imbalance by upsampling the data
        points with active ELMS.

        Args:
        -----
            labels_np (np.ndarray): NumPy array containing the labels.
            valid_indices (np.ndarray): Array containing the allowed indices.
            elm_start (np.ndarray): Array containing the indices for the start of
                active ELM events.
            elm_stop (np.ndarray): Array containing the indices for the end of
                active ELM events.
            index_buffer (int, optional): Number of buffer indices to use when
                doing upsampling. Defaults to 20.

        Returns:
        --------
            np.ndarray: Array containing all the indices which can be used to
                create the data chunk.
        """
        # indices for sampling data
        sample_indices = valid_indices

        # oversample active ELM periods to reduce class imbalance
        fraction_elm = np.count_nonzero(_labels >= 0.5) / _labels.shape[0]
        LOGGER.info(f"Active ELM fraction (raw data): {fraction_elm:.3f}")
        oversample_count = int((1 - fraction_elm) / fraction_elm) - 1
        LOGGER.info(
            f"Active ELM oversampling for balanced data: {oversample_count}"
        )
        if self.balance_classes:
            for i_start, i_stop in zip(elm_start, elm_stop):
                assert np.all(_labels[i_start : i_stop + 1] >= 0.5)
                active_elm_window = np.arange(
                    i_start - index_buffer, i_stop + index_buffer, dtype="int"
                )
                active_elm_window = np.tile(
                    active_elm_window, [oversample_count]
                )
                sample_indices = np.concatenate(
                    [sample_indices, active_elm_window]
                )
            fraction_elm = (
                np.count_nonzero(_labels[sample_indices] >= 0.5)
                / sample_indices.size
            )
            LOGGER.info(
                f"Active ELM fraction (balanced data): {fraction_elm:.3f}"
            )
        return sample_indices


class ELMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        sample_indices: np.ndarray,
        window_start: np.ndarray,
        signal_window_size: int,
        label_look_ahead: int,
        stack_elm_events: bool = False,
        transform=None,
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
        self.signals = signals
        self.labels = labels
        self.sample_indices = sample_indices
        self.window_start = window_start
        self.signal_window_size = signal_window_size
        self.label_look_ahead = label_look_ahead
        self.stack_elm_events = stack_elm_events
        self.transform = transform
        LOGGER.info("-" * 15)
        LOGGER.info(" Dataset class")
        LOGGER.info("-" * 15)
        LOGGER.info(f"Signals shape: {signals.shape}")
        LOGGER.info(f"Labels shape: {labels.shape}")
        LOGGER.info(f"Sample indices shape: {sample_indices.shape}")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx: int):
        elm_idx = self.sample_indices[idx]
        signal_window = self.signals[
            elm_idx : elm_idx + self.signal_window_size
        ]
        label = self.labels[
            elm_idx + self.signal_window_size + self.label_look_ahead - 1
        ].astype("int")
        if self.stack_elm_events:
            if config.signal_window_size == 8:
                signal_window = np.hsplit(
                    np.concatenate(signal_window, axis=-1), 2
                )
                signal_window = np.concatenate(signal_window)
            elif config.signal_window_size == 16:
                signal_window = np.hsplit(
                    np.concatenate(signal_window, axis=-1), 4
                )
                signal_window = np.concatenate(signal_window)
            else:
                raise Exception(
                    f"Expected signal window size is 8 or 16 but got {config.signal_window_size}."
                )
            if self.transform:
                signal_window = self.transform(image=signal_window)["image"]

        signal_window = torch.as_tensor(signal_window, dtype=torch.float32)
        signal_window.unsqueeze_(0)
        label = torch.as_tensor(label, dtype=torch.long)
        return signal_window, label


def get_transforms():
    return A.Compose([A.Resize(config.size, config.size)])


if __name__ == "__main__":
    fold = 1
    data = Data(kfold=True, balance_classes=config.balance_classes)
    LOGGER.info("-" * 10)
    LOGGER.info(f" Fold: {fold}")
    LOGGER.info("-" * 10)
    train_data, _, _ = data.get_data(shuffle_sample_indices=True, fold=fold)
    _, _, sample_indices, window_start = train_data

    LOGGER.info(f"Sample indices: {sample_indices[:10]}")
    values, counts = np.unique(sample_indices, return_counts=True)
    LOGGER.info(f"Values: {values[counts > 1]}")
    LOGGER.info(f"Counts: {counts[counts > 1]}")
    LOGGER.info(f"Number of non-unique values: {len(values[counts > 1])}")
    LOGGER.info(
        f"Window start indices - shape: {window_start.shape}, first 10: {window_start[:10]}"
    )
    transforms = get_transforms()

    train_dataset = ELMDataset(
        *train_data,
        config.signal_window_size,
        config.label_look_ahead,
        stack_elm_events=True,
        transform=transforms,
    )
    sample = train_dataset.__getitem__(0)
    print(sample[1])
    print(sample[0].shape)
    print(sample[0])
