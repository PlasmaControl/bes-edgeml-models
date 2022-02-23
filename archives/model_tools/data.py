"""
Data class to package BES data for training
"""

import numpy as np
import h5py
import tensorflow as tf

try:
    from . import utilities

    print("Package-level relative import")
except ImportError:
    import utilities

    print("Direct import")


class Data(object):
    def __init__(
        self,
        datafile=None,
        signal_window_size=8,
        label_look_ahead=0,
        # super_window_size=250,
        max_elms=None,
        fraction_validate=0.15,  # validation data fraction
        fraction_test=0.15,  # test data fraction
        training_batch_size=4,
        # super_window_shuffle_seed=None,
        transition_halfwidth=3,
        signal_dtype="float32",
        # elming_oversample=6,
    ):

        self.datafile = datafile
        if self.datafile is None:
            self.datafile = utilities.data_dir / "labeled-elm-events.hdf5"
        self.signal_window_size = signal_window_size
        self.label_look_ahead = label_look_ahead
        self.max_elms = max_elms
        self.fraction_validate = fraction_validate
        self.fraction_test = fraction_test
        self.training_batch_size = training_batch_size
        self.transition_halfwidth = transition_halfwidth
        self.signal_dtype = signal_dtype

        self.training_elms = None
        self.validation_elms = None
        self.test_elms = None

        self.transition = np.linspace(0, 1, 2 * self.transition_halfwidth + 3)

        self.partition_elms()

        print("Reading training ELMs and make dataset")
        self.training_data = self.read_data(
            self.training_elms,
            shuffle_sample_indices=True,
        )
        self.train_dataset = self.make_dataset(
            signals=self.training_data[0],
            labels=self.training_data[1],
            sample_indices=self.training_data[2],
            batch_size=self.training_batch_size,
        )
        self.train_dataset = self.train_dataset.repeat(count=20)

        print("Reading validation ELMs and make dataset")
        self.validation_data = self.read_data(
            self.validation_elms,
            shuffle_sample_indices=True,
        )
        self.validation_dataset = self.make_dataset(
            signals=self.validation_data[0],
            labels=self.validation_data[1],
            sample_indices=self.validation_data[2],
            # batch_size=16,
        )

        print("Reading test ELMs and make dataset")
        self.test_data = self.read_data(
            self.test_elms,
            shuffle_sample_indices=False,
        )
        self.test_dataset = self.make_dataset(
            signals=self.test_data[0],
            labels=self.test_data[1],
            sample_indices=self.test_data[2],
            # batch_size=16,
        )

        print(f"Training batch size: {self.training_batch_size}")
        self.n_training_batches = (
            self.training_data[2].shape[0] // self.training_batch_size
        )
        print(f"Number of training batches: {self.n_training_batches}")

    def partition_elms(self):
        print(f"Datafile: {self.datafile.as_posix()}")
        assert self.datafile.exists()
        # get ELM indices from datafile
        with h5py.File(self.datafile, "r") as hf:
            print(f"  ELM events in datafile: {len(hf)}")
            elm_index = np.array([int(key) for key in hf], dtype=np.int)
        # shuffle indices
        np.random.shuffle(elm_index)
        if self.max_elms:
            print(f"  Limiting data read to {self.max_elms} ELM events")
            n_elms = self.max_elms
        else:
            n_elms = elm_index.size
        # data partition sizes
        n_validation_elms = np.int(n_elms * self.fraction_validate)
        n_test_elms = np.int(n_elms * self.fraction_test)
        n_training_elms = n_elms - n_validation_elms - n_test_elms
        # partition ELMs
        elm_index = elm_index[:n_elms]
        self.training_elms = elm_index[:n_training_elms]
        elm_index = elm_index[n_training_elms:]
        self.validation_elms = elm_index[:n_validation_elms]
        elm_index = elm_index[n_validation_elms:]
        self.test_elms = elm_index[:]
        print(f"Training ELM events: {self.training_elms.size}")
        print(f"Validation ELM events: {self.validation_elms.size}")
        print(f"Test ELM events: {self.test_elms.size}")

    def read_data(self, elm_indices=(), shuffle_sample_indices=False):
        signals = None
        with h5py.File(self.datafile, "r") as hf:
            for elm_index in elm_indices:
                elm_key = f"{elm_index:05d}"
                elm_event = hf[elm_key]  # h5py group for ELM event
                signals_np = np.array(
                    elm_event["signals"], dtype=self.signal_dtype
                )
                signals_np = signals_np.transpose().reshape(
                    [-1, 8, 8]
                )  # reshape for 8x8 spatial BES data
                signals_np = signals_np / 10  # rescale to max=1
                labels_np = np.array(
                    elm_event["labels"], dtype=self.signal_dtype
                )
                # apply transition as ELM turns on and off
                i_ones = np.nonzero(labels_np)[0]  # indices of labels_np == 1
                for direction, idx in zip([1, -1], [i_ones[0], i_ones[-1]]):
                    idx0 = idx - self.transition_halfwidth - 1
                    idx1 = idx + self.transition_halfwidth + 2
                    labels_np[idx0:idx1] = self.transition[::direction]
                # valid indices as basis for data windows
                valid_t0_np = np.ones(labels_np.shape, dtype=np.int8)
                valid_t0_np[
                    -(self.signal_window_size + self.label_look_ahead) + 1 :
                ] = 0  # 0 for invalid t0
                assert np.count_nonzero(valid_t0_np == 0) == (
                    self.signal_window_size + self.label_look_ahead - 1
                )
                # indices for active ELM
                active_elm_indices = np.nonzero(labels_np >= 0.5)[0]
                if signals is None:
                    # initialize tensors
                    window_start_indices = np.array([0], dtype=np.int64)
                    elm_start_indices = np.array(
                        active_elm_indices[0], dtype=np.int64
                    )
                    elm_stop_indices = np.array(
                        active_elm_indices[-1], dtype=np.int64
                    )
                    valid_t0 = valid_t0_np
                    signals = tf.convert_to_tensor(signals_np)
                    labels = tf.convert_to_tensor(labels_np)
                else:
                    # concat on dim 0 (time dimension)
                    final_index = labels.shape[0] - 1
                    window_start_indices = np.append(
                        window_start_indices, final_index + 1
                    )
                    elm_start_indices = np.append(
                        elm_start_indices,
                        final_index + active_elm_indices[0] + 1,
                    )
                    elm_stop_indices = np.append(
                        elm_stop_indices,
                        final_index + active_elm_indices[-1] + 1,
                    )
                    valid_t0 = np.concatenate([valid_t0, valid_t0_np])
                    signals = tf.concat(
                        [signals, tf.convert_to_tensor(signals_np)], 0
                    )
                    labels = tf.concat(
                        [labels, tf.convert_to_tensor(labels_np)], 0
                    )

        labels_np = np.array(labels)

        assert np.all(labels_np[elm_start_indices] >= 0.5)
        assert np.all(labels_np[elm_start_indices - 1] < 0.5)
        assert np.all(labels_np[elm_stop_indices] >= 0.5)
        assert np.all(labels_np[elm_stop_indices + 1] < 0.5)

        # find valid indices for data sampling
        valid_indices = np.arange(valid_t0.size, dtype=np.int64)
        valid_indices = valid_indices[valid_t0 == 1]

        # indices for sampling data
        sample_indices = valid_indices

        # oversample active ELM periods to balance data
        fraction_elm = np.count_nonzero(labels_np >= 0.5) / labels.shape[0]
        print(f"  Active ELM fraction (raw data): {fraction_elm:.3f}")
        oversample_count = int((1 - fraction_elm) / fraction_elm) - 1
        print(
            f"  Active ELM oversampling for balanced data: {oversample_count}"
        )
        for i_start, i_stop in zip(elm_start_indices, elm_stop_indices):
            assert np.all(labels[i_start : i_stop + 1] >= 0.5)
            active_elm_window = np.arange(
                i_start - 20, i_stop + 20, dtype=np.int64
            )
            assert labels[active_elm_window[0]] < 0.5
            assert labels[active_elm_window[-1]] < 0.5
            assert active_elm_window[-1] < labels.shape[0]
            active_elm_window = np.tile(active_elm_window, [oversample_count])
            sample_indices = np.concatenate([sample_indices, active_elm_window])
        fraction_elm = (
            np.count_nonzero(labels_np[sample_indices] >= 0.5)
            / sample_indices.size
        )
        print(f"  Active ELM fraction (balanced data): {fraction_elm:.3f}")

        sample_indices = tf.constant(sample_indices)

        # shuffle sample indices?
        if shuffle_sample_indices:
            print("Shuffling sample indices")
            sample_indices = tf.random.shuffle(sample_indices)

        print(
            "Data tensors: signals, labels, valid_indices, sample_indices, window_start_indices"
        )
        for tensor in [
            signals,
            labels,
            valid_indices,
            sample_indices,
            window_start_indices,
        ]:
            tmp = f"  shape {tensor.shape} dtype {tensor.dtype}"
            tmp += f" min {np.min(tensor):.3f} max {np.max(tensor):.3f}"
            if hasattr(tensor, "device"):
                tmp += f" device {tensor.device[-5:]}"
            print(tmp)

        return signals, labels, sample_indices, window_start_indices

    def _make_generator(self, signals_in, labels_in, sample_indices_in):
        signal_window_size = self.signal_window_size
        label_look_ahead = self.label_look_ahead

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

    def make_dataset(
        self, signals=None, labels=None, sample_indices=None, batch_size=4
    ):
        print("  Make dataset from generator")
        generator = self._make_generator(signals, labels, sample_indices)
        dtypes = (signals.dtype, labels.dtype)
        shapes = (
            tf.TensorShape([self.signal_window_size, 8, 8, 1]),
            tf.TensorShape([1]),
        )
        dataset = tf.data.Dataset.from_generator(
            generator, dtypes, output_shapes=shapes
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


if __name__ == "__main__":

    # turn off GPU visibility
    tf.config.set_visible_devices([], "GPU")

    print("TF version:", tf.__version__)
    print("Visible devices:")
    for device in tf.config.get_visible_devices():
        print(f"  {device.device_type} {device.name}")

    data = Data(max_elms=100)
