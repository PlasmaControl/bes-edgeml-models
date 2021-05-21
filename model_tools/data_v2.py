import os
import h5py

import numpy as np
from sklearn import model_selection

# import tensorflow as tf

try:
    from . import utilities

    print("Package-level relative import.")
except ImportError:
    import utilities

    print("Loading datafile from direct import.")

# TODO: Add config file to make fixed variables globally available
class Data(object):
    # TODO: Add type hints
    def __init__(
        self,
        datafile: str = None,
        signal_window_size: int = 8,
        label_look_ahead: int = 0,
        max_elms=None,
        fraction_validate: float = 0.2,
        fraction_test: float = 0.1,
        training_batch_size: int = 4,
        transition_halfwidth: int = 3,
        signal_dtype: str = "float32",
        kfold: bool = False,
        folds: int = 6,
    ):
        # TODO: Add docstring
        # TODO: Add universal seed for reproducibility
        self.datafile = datafile
        if self.datafile is None:
            self.datafile = os.path.join(
                utilities.data_dir, "labeled-elm-events.hdf5"
            )
        self.signal_window_size = signal_window_size
        self.label_look_ahead = label_look_ahead
        self.max_elms = max_elms
        self.fraction_validate = fraction_validate
        self.fraction_test = fraction_test
        self.training_batch_size = training_batch_size
        self.transition_halfwidth = transition_halfwidth
        self.signal_dtype = signal_dtype
        self.kfold = kfold
        self.folds = folds

        self.training_elms = None
        self.validation_elms = None
        self.test_elms = None

        self.transition = np.linspace(0, 1, 2 * self.transition_halfwidth + 3)
        self.partition_elms()

        # TODO: Create a helper function for creating datasets to avoid repetition
        # print("Reading training ELMs and make dataset")
        # self.training_data = self.read_data(
        #     self.training_elms, shuffle_sample_indices=True
        # )
        # self.train_dataset = self.make_dataset(
        #     signals=self.training_data[0],
        #     labels=self.training_data[1],
        #     sample_indices=self.training_data[2],
        #     batch_size=self.training_batch_size,
        # )
        # self.train_dataset = self.train_dataset.repeat(count=20)

        # print("Reading validation ELMs and make dataset")
        # self.validation_data = self.read_data(
        #     self.validation_elms, shuffle_sample_indices=True
        # )
        # self.validation_dataset = self.make_dataset(
        #     signals=self.validation_data[0],
        #     labels=self.validation_data[1],
        #     sample_indices=self.validation_data[2],
        #     # batch_size=self.training_batch_size,
        # )

        # print("Reading test ELMs and make dataset")
        # self.test_data = self.read_data(
        #     self.test_elms, shuffle_sample_indices=False
        # )
        # self.validation_dataset = self.make_dataset(
        #     signals=self.test_data[0],
        #     labels=self.test_data[1],
        #     sample_indices=self.test_data[2],
        #     # batch_size=self.training_batch_size,
        # )
        # print(f"Training batch size: {self.training_batch_size}")
        # self.n_training_batches = (
        #     self.training_data[2].shape[0] // self.training_batch_size
        # )
        # print(f"Number of training batches: {self.n_training_batches}")

    # TODO: Add a function to read the HDF5 file
    def partition_elms(self):
        print(f"Datafile: {self.datafile}")
        assert os.path.exists(self.datafile)

        # get ELM indices from datafile
        with h5py.File(self.datafile, "r") as hf:
            print(f"    Number of ELM events in the datafile: {len(hf)}")
            elm_index = np.array([int(key) for key in hf], dtype=np.int)

        # shuffle indices
        np.random.shuffle(elm_index)
        if self.max_elms:
            print(f"Limiting data read to {self.max_elms} events.")
            n_elms = self.max_elms
        else:
            n_elms = len(elm_index)

        # splitting the data into train, validation and test sets
        self.training_elms, self.test_elms = model_selection.train_test_split(
            elm_index[:n_elms],
            test_size=self.fraction_test,
            shuffle=True,
            random_state=23,
        )

        # kfold cross validation
        if self.kfold:
            kf = model_selection.KFold(
                n_splits=6, shuffle=True, random_state=23
            )
            for f_, (train, valid) in enumerate(kf.split(X=self.training_elms)):
                print(f"Fold: {f_}")
                print(f"Train group: {self.training_elms[train]}")
                print(f"Valid group: {self.training_elms[valid]}")
        else:
            (
                self.training_elms,
                self.validation_elms,
            ) = model_selection.train_test_split(
                self.training_elms, test_size=self.fraction_validate
            )
            print(
                f"    Number of training ELM events: {self.training_elms.size}"
            )
            print(
                f"    Number of validation ELM events: {self.validation_elms.size}"
            )
        print(f"    Number of test ELM events: {self.test_elms.size}")

    def read_data(self, elm_indices=None, shuffle_sample_indices=False):
        signal = None
        with h5py.File(self.datafile, "r") as hf:
            elm_indices = np.array([int(key) for key in hf], dtype=np.int)
            for elm_index in elm_indices:
                elm_key = f"{elm_index:05d}"
                elm_event = hf[elm_key]
                signals = np.array(
                    elm_event["signals"], dtype=self.signal_dtype
                )
                signals = np.transpose(signals, (1, 0)).reshape(-1, 8, 8)
                print(signals.shape)
                #! rescale the signal to max=1
                labels = np.array(elm_event["labels"], dtype=np.int8)
                print(labels.shape)
                print(len(np.nonzero(labels)[0]))
                print(np.nonzero(labels)[0])
                # add transitions to make ELM turning ON and OFF smoother
                non_zero_elm = np.nonzero(labels)[0]
                #! Possibly a problem with using the direction?
                for direction, idx in zip(
                    [-1, 1], [non_zero_elm[0], non_zero_elm[-1]]
                ):
                    idx_start = idx - self.transition_halfwidth - 1
                    idx_end = idx + self.transition_halfwidth + 2
                    labels[idx_start:idx_end] = self.transition[::direction]
                    print(
                        f"Direction: {direction}, non zero labels:\n{np.nonzero(labels)[0]}"
                    )
                break


if __name__ == "__main__":
    ds = Data(max_elms=20, kfold=True)
    ds.read_data()
