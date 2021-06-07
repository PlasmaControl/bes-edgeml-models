import os
import logging
import h5py
from typing import Tuple, Callable, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import model_selection

try:
    from . import utilities, config

    print("Package-level relative import.")
except ImportError:
    import utilities
    import config

    print("Loading datafile from direct import.")


# log the model and data-preprocessing outputs
def get_logger(stream_handler=True):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create handlers
    f_handler = logging.FileHandler("output_logs.log")

    # create formatters and add it to handlers
    f_format = logging.Formatter(
        "%(asctime)s:%(name)s: %(levelname)s:%(message)s"
    )

    f_handler.setFormatter(f_format)

    # add handlers to the logger
    logger.addHandler(f_handler)

    # display the logs in console
    if stream_handler:
        s_handler = logging.StreamHandler()
        s_format = logging.Formatter("%(name)s:%(levelname)s:%(message)s")
        s_handler.setFormatter(s_format)
        logger.addHandler(s_handler)

    return logger


# create the logger object
LOGGER = get_logger()


class Data:
    """ """

    def __init__(
        self,
        datafile: str = None,
        fraction_validate: float = 0.2,
        fraction_test: float = 0.1,
        signal_dtype: str = "float32",
        kfold: bool = False,
        smoothen_transition: bool = False,
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
        """
        self.datafile = datafile
        if self.datafile is None:
            self.datafile = os.path.join(utilities.data_dir, config.file_name)
        self.fraction_validate = fraction_validate
        self.fraction_test = fraction_test
        self.signal_dtype = signal_dtype
        self.kfold = kfold
        self.smoothen_transition = smoothen_transition
        self.max_elms = config.max_elms

        self.transition = np.linspace(0, 1, 2 * config.transition_halfwidth + 3)

    def get_datasets(self, fold: int = None) -> Tuple[List, List]:
        """Method to create tensorflow datasets for training, validation and
        testing.

        Args:
        -----
            fold (int, optional): Fold index for K-fold cross-validation between
                [0-num_folds). Must be passed when `kfold=True`. Defaults to None.

        Returns:
        --------
            Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Tuple containing
                tensorflow datasets for training, validation and test sets.
        """
        training_elms, validation_elms, test_elms = self._partition_elms(
            max_elms=config.max_elms, fold=fold
        )
        LOGGER.info("Reading ELM events and creating datasets")
        LOGGER.info("-" * 30)
        LOGGER.info("  Creating training dataset")
        LOGGER.info("-" * 30)
        train_data, train_dataset = self._get_data(training_elms)
        LOGGER.info("-" * 30)
        LOGGER.info("  Creating validation dataset")
        LOGGER.info("-" * 30)
        validation_data, validation_dataset = self._get_data(
            validation_elms, shuffle_sample_indices=False
        )
        LOGGER.info("-" * 30)
        LOGGER.info("  Creating test dataset")
        LOGGER.info("-" * 30)
        test_data, test_dataset = self._get_data(
            test_elms, shuffle_sample_indices=False
        )

        return [train_data, validation_data, test_data], [
            train_dataset,
            validation_dataset,
            test_dataset,
        ]

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

        # shuffle indices
        np.random.shuffle(elm_index)
        if max_elms is not None and max_elms != -1:
            LOGGER.info(f"Limiting data read to {max_elms} events.")
            n_elms = max_elms
        else:
            n_elms = len(elm_index)

        # splitting the data into train, validation and test sets
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
            LOGGER.info("Using K-fold cross-validation")
            self._kfold_cross_val(training_elms)
            training_elms = self.df[self.df["fold"] != fold]["elm_events"]
            validation_elms = self.df[self.df["fold"] == fold]["elm_events"]
        else:
            LOGGER.info(
                "Creating training and validation datasets by simple splitting"
            )
            (
                training_elms,
                validation_elms,
            ) = model_selection.train_test_split(
                training_elms, test_size=self.fraction_validate
            )
        LOGGER.info(f"Number of training ELM events: {training_elms.size}")
        LOGGER.info(f"Number of validation ELM events: {validation_elms.size}")
        LOGGER.info(f"Number of test ELM events: {test_elms.size}")

        return training_elms, validation_elms, test_elms

    def _kfold_cross_val(self, training_elms) -> None:
        """Helper function to perform K-fold cross-validation.

        Args:
        -----
            training_elms (np.ndarray): Indices for training ELM events.
        """
        self.df = pd.DataFrame()
        kf = model_selection.KFold(
            n_splits=config.folds, shuffle=True, random_state=config.seed
        )
        self.df["elm_events"] = training_elms
        self.df["fold"] = -1
        for f_, (_, valid_idx) in enumerate(kf.split(X=training_elms)):
            self.df.loc[valid_idx, "fold"] = f_

    def _get_data(
        self, elms: np.ndarray, shuffle_sample_indices: bool = True
    ) -> Tuple[
        Tuple[tf.Tensor, tf.Tensor, np.ndarray, np.ndarray], tf.data.Dataset
    ]:
        """Creates tensorflow dataset after preprocessing steps like label smoothening,
        calculating allowed indices and upsampling.

        Args:
        -----
            elms (np.ndarray): ELM event indices for the corresponding mode.
            shuffle_sample_indices (bool, optional): Whether or not to shuffle
            the indices. Should be turned off during validation and test modes.
            Defaults to True.

        Returns:
        --------
            Tuple[Tuple[tf.Tensor, tf.Tensor, np.ndarray, np.ndarray], tf.data.Dataset]:
                Tuple containing data and tensorflow dataset for the given mode.
        """
        data = self._read_data(
            elms, shuffle_sample_indices=shuffle_sample_indices
        )
        dataset = self._make_dataset(
            signals=data[0],
            labels=data[1],
            sample_indices=data[2],
            batch_size=config.batch_size,
        )

        return data, dataset

    def _read_data(
        self,
        elm_indices: np.ndarray = None,
        shuffle_sample_indices: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor, np.ndarray, np.ndarray]:
        """Helper function to preprocess the data: reshape the input signal, use
        allowed indices to upsample the class minority labels [active ELM events].

        Args:
        -----
            elm_indices (np.ndarray, optional): ELM event indices for the corresponding
                mode. Defaults to None.
            shuffle_sample_indices (bool, optional): Whether to shuffle the indices
                once upsampled. Defaults to False.

        Returns:
        --------
            Tuple[tf.Tensor, tf.Tensor, np.ndarray, np.ndarray]: Tuple containing
                original signals, correponding labels, sample indices obtained
                after  upsampling and start index for each ELM event.
        """
        signals = None
        window_start = None
        elm_start = None
        elm_stop = None
        hf = None
        valid_t0 = []
        labels = []
        # get ELM indices from datafile if not provided
        if elm_indices is None:
            elm_indices, hf = self._read_file()
        _, hf = self._read_file()
        for elm_index in elm_indices:
            elm_key = f"{elm_index:05d}"
            elm_event = hf[elm_key]
            signals_np = np.array(elm_event["signals"], dtype=self.signal_dtype)
            signals_np = np.transpose(signals_np, (1, 0)).reshape(-1, 8, 8)
            labels_np = np.array(elm_event["labels"], dtype="float")

            # add transitions to make ELM labels 0 and 1 smoother
            if self.smoothen_transition:
                labels_np = self._label_transition(labels_np)

            indices_data = self._get_valid_indices(
                signals_np=signals_np,
                labels_np=labels_np,
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

        labels_np = np.array(labels)

        # valid indices for data sampling
        valid_indices = np.arange(valid_t0.size, dtype="int")
        valid_indices = valid_indices[valid_t0 == 1]

        sample_indices = self._oversample_data(
            labels_np, valid_indices, elm_start, elm_stop
        )

        sample_indices = tf.constant(sample_indices)

        # shuffle sample indices?
        if shuffle_sample_indices:
            print("Shuffling sample indices")
            sample_indices = tf.random.shuffle(sample_indices)

        LOGGER.info(
            "Data tensors: signals, labels, valid_indices, sample_indices, window_start_indices"
        )
        for tensor in [
            signals,
            labels,
            valid_indices,
            sample_indices,
            window_start,
        ]:
            tmp = f"  shape {tensor.shape} dtype {tensor.dtype}"
            tmp += f" min {np.min(tensor):.3f} max {np.max(tensor):.3f}"
            if hasattr(tensor, "device"):
                tmp += f" device {tensor.device[-5:]}"
            LOGGER.info(tmp)

        hf.close()
        if hf:
            LOGGER.info("File is open")
        else:
            LOGGER.info("File is closed")
        return signals, labels, sample_indices, window_start

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
        elm_index = np.array([int(key) for key in hf], dtype=np.int)
        return elm_index, hf

    def _label_transition(self, labels_np: np.ndarray) -> np.ndarray:
        """Helper function to smoothen the label transition from 0 to 1.

        Args:
        -----
            labels_np (np.ndarray): NumPy array containing the labels.

        Returns:
        --------
            np.ndarray: Smoothened labels.
        """
        non_zero_elm = np.nonzero(labels_np)[0]
        for direction, idx in zip([1, -1], [non_zero_elm[0], non_zero_elm[-1]]):
            idx_start = idx - config.transition_halfwidth - 1
            idx_end = idx + config.transition_halfwidth + 2
            labels_np[idx_start:idx_end] = self.transition[::direction]

        return labels_np

    def _get_valid_indices(
        self,
        signals_np: np.ndarray,
        labels_np: np.ndarray,
        window_start_indices: np.ndarray = None,
        elm_start_indices: np.ndarray = None,
        elm_stop_indices: np.ndarray = None,
        valid_t0: np.ndarray = None,
        labels: tf.Tensor = None,
        signals: tf.Tensor = None,
    ) -> Tuple[
        tf.Tensor, tf.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Helper function to concatenate the signals and labels for the ELM events
        for a given mode. It also creates allowed indices to sample from with respect
        to signal window size and label look ahead. See the README to know more
        about it.

        Args:
        -----
            signals_np (np.ndarray): NumPy array containing the inputs.
            labels_np (np.ndarray): NumPy array containing the labels.
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
        # allowed indices as the basis of the data windows
        valid_t0_np = np.ones(labels_np.shape, dtype=np.int8)
        valid_t0_np[
            -(config.signal_window_size + config.label_look_ahead) + 1 :
        ] = 0

        # indices for active ELM
        active_elm_indices = np.nonzero(labels_np >= 0.5)[0]

        if signals is None:
            # initialize tensors
            window_start_indices = np.array([0])
            elm_start_indices = np.array(active_elm_indices[0], dtype="int")
            elm_stop_indices = np.array(active_elm_indices[-1], dtype="int")
            valid_t0 = valid_t0_np
            signals = tf.convert_to_tensor(signals_np)
            labels = tf.convert_to_tensor(labels_np)
        else:
            # concat on axis=0 (time dimension)
            last_index = labels.shape[0] - 1
            window_start_indices = np.append(
                window_start_indices, last_index + 1
            )
            elm_start_indices = np.append(
                elm_start_indices,
                last_index + active_elm_indices[0] + 1,
            )
            elm_stop_indices = np.append(
                elm_stop_indices,
                last_index + active_elm_indices[-1] + 1,
            )
            valid_t0 = np.concatenate([valid_t0, valid_t0_np])
            signals = tf.concat(
                [signals, tf.convert_to_tensor(signals_np)], axis=0
            )
            labels = tf.concat(
                [labels, tf.convert_to_tensor(labels_np)], axis=0
            )

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
        labels_np: np.ndarray,
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
            np.ndarray: [description]
        """
        # indices for sampling data
        sample_indices = valid_indices

        # oversample active ELM periods to reduce class imbalance
        fraction_elm = np.count_nonzero(labels_np >= 0.5) / labels_np.shape[0]
        LOGGER.info(f"Active ELM fraction (raw data): {fraction_elm:.3f}")
        oversample_count = int((1 - fraction_elm) / fraction_elm) - 1
        LOGGER.info(
            f"Active ELM oversampling for balanced data: {oversample_count}"
        )
        for i_start, i_stop in zip(elm_start, elm_stop):
            assert np.all(labels_np[i_start : i_stop + 1] >= 0.5)
            active_elm_window = np.arange(
                i_start - index_buffer, i_stop + index_buffer, dtype="int"
            )
            active_elm_window = np.tile(active_elm_window, [oversample_count])
            sample_indices = np.concatenate([sample_indices, active_elm_window])
        fraction_elm = (
            np.count_nonzero(labels_np[sample_indices] >= 0.5)
            / sample_indices.size
        )
        LOGGER.info(f"Active ELM fraction (balanced data): {fraction_elm:.3f}")
        return sample_indices

    def _make_dataset(
        self,
        signals: tf.Tensor = None,
        labels: tf.Tensor = None,
        sample_indices: np.ndarray = None,
        batch_size: int = config.batch_size,
    ) -> tf.data.Dataset:
        """Create dataset batches from `tensorflow.data.Dataset.from_generator`.

        Args:
        -----
            signals (tf.Tensor, optional): Input data. Defaults to None.
            labels (tf.Tensor, optional): Targets. Defaults to None.
            sample_indices (np.ndarray, optional): Indices to sample the dataset
                from the generator. Defaults to None.
            batch_size (int, optional): Batch size for each dataset. Defaults to
                config.batch_size.

        Returns:
        --------
            tf.data.Dataset: Dataset batches.
        """
        LOGGER.info("Making dataset from generator")
        generator = self._make_generator(signals, labels, sample_indices)
        dtypes = (signals.dtype, labels.dtype)
        shapes = (
            tf.TensorShape([config.signal_window_size, 8, 8, 1]),
            tf.TensorShape([1]),
        )
        dataset = tf.data.Dataset.from_generator(
            generator, dtypes, output_shapes=shapes
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def _make_generator(
        self,
        signals_in: tf.Tensor,
        labels_in: tf.Tensor,
        sample_indices_in: np.ndarray,
    ) -> Callable:
        """Create dataset generator using the allowed indices.

        Args:
        -----
            signals_in (tf.Tensor): Input tensors.
            labels_in (tf.Tensor): Labels.
            sample_indices_in (np.ndarray): Array containing the upsampled indices
                obtained from allowed indices.

        Returns:
        --------
            Callable: Generator function yielding signal windows and labels
                corresponding to the signal windows.

        Yields:
        -------
            Tuple[tf.Tensor, tf.Tensor]: Signal window and labels.
        """
        signal_window_size = config.signal_window_size
        label_look_ahead = config.label_look_ahead

        def generator():
            i = 0
            while i < sample_indices_in.shape[0]:
                i_t0 = sample_indices_in[i]
                assert (
                    i_t0 + signal_window_size + label_look_ahead - 1
                ) < labels_in.shape[0]
                signal_window = signals_in[
                    i_t0 : i_t0 + signal_window_size, ...
                ]
                signal_window = tf.reshape(
                    signal_window, [signal_window_size, 8, 8, 1]
                )
                label = labels_in[
                    i_t0 + signal_window_size + label_look_ahead - 1
                ]
                label = tf.reshape(label, [1])
                i += 1
                yield signal_window, label

        return generator


if __name__ == "__main__":
    # turn off GPU visibility for tensorflow
    tf.config.set_visible_devices([], "GPU")
    print(f"Tensorflow version: {tf.__version__}")
    print("Visible devices:")
    for device in tf.config.get_visible_devices():
        print(f"{device.device_type} {device.name}")
    ds = Data(kfold=True)
    train, valid, test = ds.get_datasets(fold=0)
