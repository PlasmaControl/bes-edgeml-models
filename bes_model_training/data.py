from pathlib import Path
import numpy as np
import h5py
import tensorflow as tf


class Data(object):

    def __init__(self,
                 datafile=Path('data/labeled-elm-events-smithdr.hdf5'),
                 super_window_size=250,
                 signal_window_size=8,
                 label_look_ahead=0,
                 max_elms=None,
                 fraction_validate = 0.15,  # validation data for post-epoch evaluation
                 fraction_test = 0.15,  # test data for post-training evaluation
                 training_batch_size=4,
                 super_window_shuffle_seed=None,
                 make_logits=False,
                 transition_halfwidth=3,
                 signal_dtype='float32',
                 ):

        self.datafile = datafile
        self.super_window_size = super_window_size
        self.signal_window_size = signal_window_size
        self.label_look_ahead = label_look_ahead
        self.max_elms = max_elms
        self.fraction_validate = fraction_validate
        self.fraction_test = fraction_test
        self.training_batch_size = training_batch_size
        self.super_window_shuffle_seed = super_window_shuffle_seed
        self.make_logits = make_logits
        self.transition_halfwidth = transition_halfwidth
        self.signal_dtype = signal_dtype

        self.signals = None
        self.labels = None
        self.valid_t0 = None
        self.valid_indices = None
        self.label_fractions = None
        self.n_super_windows = None
        self.n_train = None
        self.n_validate = None
        self.n_test = None
        self.n_training_batches = None
        self.ds_train = None
        self.ds_validate = None
        self.ds_test = None
        self.test_signals_superwindows = None
        self.test_labels_superwindows = None

        self.read_data_into_tensors()
        self.normalize_signals()
        self.shuffle_super_windows()
        self.partition_and_make_datasets()

    def read_data_into_tensors(self):
        print(f'Datafile: {self.datafile.as_posix()}')
        total_signal_window = self.signal_window_size + self.label_look_ahead
        assert(self.datafile.exists())
        transition = np.linspace(0, 1, 2 * self.transition_halfwidth + 3)
        with h5py.File(self.datafile, 'r') as hf:
            print(f'Number of ELM events in datafile: {len(hf)}')
            print('Reading ELM event data into `super windows`')
            # loop over ELM events
            for ielm, elm_event in enumerate(hf.values()):
                if self.max_elms and ielm >= self.max_elms:
                    print(f'Limiting data read to {self.max_elms} ELM events')
                    break
                # partition data into `super windows` to enable shuffling of time-series windows
                n_super_windows = elm_event['labels'].size // self.super_window_size
                # load BES signals and reshape into super windows
                signals_np = np.array(elm_event['signals'][..., 0:n_super_windows * self.super_window_size],
                                      dtype=self.signal_dtype)
                signals_np = signals_np.T.reshape(n_super_windows, self.super_window_size, 8, 8)
                # load ELM labels and reshape into super windows
                labels_np = np.array(elm_event['labels'][0:n_super_windows * self.super_window_size],
                                     dtype=self.signal_dtype)
                # apply transition as ELM turns on and off
                i_ones = np.nonzero(labels_np)[0]
                for direction, idx in zip([1,-1], [i_ones[0], i_ones[-1]]):
                    idx0 = idx - self.transition_halfwidth - 1
                    idx1 = idx + self.transition_halfwidth + 2
                    labels_np[idx0:idx1] = transition[::direction]
                labels_np = labels_np.reshape(n_super_windows, self.super_window_size)
                labels_np[labels_np == 0] = 1e-3
                labels_np[labels_np == 1] = 1.0 - 1e-3
                # construct valid_t0 mask;
                valid_t0_np = np.ones(labels_np.shape, dtype=np.int8)
                valid_t0_np[:, (self.super_window_size-total_signal_window):self.super_window_size] = 0
                # convert to tensors
                signals_tmp = tf.convert_to_tensor(signals_np)
                labels_tmp = tf.convert_to_tensor(labels_np)
                valid_t0_tmp = tf.convert_to_tensor(valid_t0_np)
                if ielm == 0:
                    # initialize tensors
                    signals = signals_tmp
                    labels = labels_tmp
                    valid_t0 = valid_t0_tmp
                else:
                    # concat on dim 0 (super windows)
                    signals = tf.concat([signals, signals_tmp], 0)
                    labels = tf.concat([labels, labels_tmp], 0)
                    valid_t0 = tf.concat([valid_t0, valid_t0_tmp], 0)

        # active/inactive ELM summary
        n_times = np.prod(labels.shape)
        n_elm_times = np.count_nonzero(np.array(labels) >= 0.5)
        print(f'Total time points: {n_times}')
        self.label_fractions = {0: (n_times - n_elm_times) / n_times,
                                1: n_elm_times / n_times,
                                }
        for key, value in self.label_fractions.items():
            print(f'Label {key} fraction: {value:.3f}')

        self.signals = signals
        self.labels = labels
        self.valid_t0 = valid_t0
        self.n_super_windows = self.labels.shape[0]

        print('Data tensors: signals, labels, valid_t0')
        for tensor in [self.signals, self.labels, self.valid_t0]:
            tmp = f'  shape {tensor.shape} dtype {tensor.dtype} device {tensor.device[-5:]} '
            tmp += f'min {np.min(tensor):.3f} max {np.max(tensor):.3f}'
            print(tmp)

    def normalize_signals(self):
        print('Normalizing signals to max=1')
        self.signals = self.signals / np.max(np.abs(self.signals))

    def shuffle_super_windows(self):
        # shuffle super windows (dim 0 in signals, labels, and valid_t0)

        if self.super_window_shuffle_seed:
            print(f'Setting shuffle seed: {self.super_window_shuffle_seed}')
            tf.random.set_seed(self.super_window_shuffle_seed)

        shuffled_indices = tf.random.shuffle(tf.range(self.n_super_windows))

        if self.super_window_shuffle_seed:
            print('Beginning shuffled indices:')
            print(shuffled_indices[:6])

        def apply_shuffle(tensor_in):
            return tf.gather(tensor_in, shuffled_indices, axis=0)

        print('Shuffling super windows')
        self.signals = apply_shuffle(self.signals)
        self.labels = apply_shuffle(self.labels)
        self.valid_t0 = apply_shuffle(self.valid_t0)

        for tensor in [self.signals, self.labels, self.valid_t0]:
            tmp = f'  shape {tensor.shape} dtype {tensor.dtype} device {tensor.device[-5:]} '
            tmp += f'min {np.min(tensor):.3f} max {np.max(tensor):.3f}'
            print(tmp)

    def _apply_partition_flatten_and_make_valid_indices(self, ind0, ind1,
                                                        shuffle_valid_indices=False,
                                                        save_test_superwindows=False):
        # partition
        signals = self.signals[ind0:ind1, ...]
        labels = self.labels[ind0:ind1, ...]
        valid_t0 = self.valid_t0[ind0:ind1, ...]

        if save_test_superwindows:
            self.test_signals_superwindows = np.array(signals)
            self.test_labels_superwindows = np.array(labels)

        # flatten
        signals = tf.reshape(signals, [-1,8,8])
        labels = tf.reshape(labels, [-1])
        valid_t0 = tf.reshape(valid_t0, [-1])

        assert(signals.ndim == 3 and labels.ndim == 1 and valid_t0.ndim == 1)
        assert(signals.shape[0] == labels.shape[0] and
               signals.shape[0] == valid_t0.shape[0])

        # find valid indices
        # make np array of indices
        valid_indices= np.arange(valid_t0.shape[0], dtype=np.int64)
        # change invalid indices to -1
        valid_indices[valid_t0 == 0] = -1
        # remove invalid indices
        valid_indices = valid_indices[valid_indices != -1]
        # convert to tensor
        valid_indices = tf.convert_to_tensor(valid_indices)
        # check for consistency
        assert (np.all(valid_indices >= 0))
        max_index = np.max(valid_indices)+self.signal_window_size+self.label_look_ahead
        assert(max_index == signals.shape[0]-1)

        if shuffle_valid_indices:
            print('  Shuffling valid indices')
            valid_indices = tf.random.shuffle(valid_indices)

        print('  signals, labels, valid_t0, valid_indices:')
        for tensor in [signals, labels, valid_t0, valid_indices]:
            tmp = f'  shape {tensor.shape} dtype {tensor.dtype} device {tensor.device[-5:]} '
            tmp += f'min {np.min(tensor):.3f} max {np.max(tensor):.3f}'
            print(tmp)

        # Note: `valid_t0` arrays are no longer needed after this point

        return signals, labels, valid_indices

    def _make_generator(self, signals, labels, valid_indices):
        signal_window_size = self.signal_window_size
        label_look_ahead = self.label_look_ahead
        def generator():
            i = 0
            while i < valid_indices.shape[0]:
                i_t0 = valid_indices[i]
                assert((i_t0 + signal_window_size) < signals.shape[0])
                signal_window = signals[i_t0:i_t0 + signal_window_size, ...]
                signal_window = tf.reshape(signal_window, [signal_window_size, 8, 8, 1])
                assert((i_t0 + signal_window_size + label_look_ahead) < labels.shape[0])
                label = labels[i_t0 + signal_window_size + label_look_ahead]
                label = tf.reshape(label, [1])
                i += 1
                yield signal_window, label
        return generator

    def partition_and_make_datasets(self):
        print('Partitioning data into train, validate, and test')
        self.n_validate = np.int(self.fraction_validate * self.n_super_windows)
        self.n_test = np.int(self.fraction_test * self.n_super_windows)
        self.n_train = self.n_super_windows - self.n_test - self.n_validate
        print(f'Super window partition:')
        print(f'  n_train {self.n_train}   n_validate {self.n_validate}   n_test {self.n_test}')

        print('Training tensors')
        train_tensors = self._apply_partition_flatten_and_make_valid_indices(
            0,
            self.n_train,
            shuffle_valid_indices=True,
            )

        print('Validation tensors')
        validation_tensors = self._apply_partition_flatten_and_make_valid_indices(
            self.n_train,
            self.n_train + self.n_validate,
            )

        print('Testing tensors')
        test_tensors = self._apply_partition_flatten_and_make_valid_indices(
            self.n_train + self.n_validate,
            self.n_super_windows,
            save_test_superwindows=True,
            )

        print('Make generators')
        generator_train = self._make_generator(*train_tensors)
        generator_validate = self._make_generator(*validation_tensors)
        generator_test = self._make_generator(*test_tensors)

        # create datasets
        print('Make datasets')
        dtypes = (self.signals.dtype, self.labels.dtype)
        shapes = (tf.TensorShape([self.signal_window_size, 8, 8, 1]), tf.TensorShape([1]))
        self.ds_train = tf.data.Dataset.from_generator(generator_train, dtypes, output_shapes=shapes)
        self.ds_validate = tf.data.Dataset.from_generator(generator_validate, dtypes, output_shapes=shapes)
        self.ds_test = tf.data.Dataset.from_generator(generator_test, dtypes, output_shapes=shapes)

        print(f'Training batch size: {self.training_batch_size}')
        self.n_training_batches = train_tensors[2].shape[0] // self.training_batch_size
        print(f'Number of training batches: {self.n_training_batches}')

        print('Batching and prefetching')
        self.ds_train = self.ds_train.\
            batch(self.training_batch_size).\
            prefetch(tf.data.AUTOTUNE)
        # shuffle(2000, reshuffle_each_iteration=True).\
        self.ds_validate = self.ds_validate.\
            batch(16).\
            prefetch(tf.data.AUTOTUNE)
        self.ds_test = self.ds_test.\
            batch(16).\
            prefetch(tf.data.AUTOTUNE)


if __name__ == '__main__':

    # make last GPU visible and limit GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[-1], 'GPU')

    print('TF version:', tf.__version__)

    print('Visible devices:')
    for device in tf.config.get_visible_devices():
        print(f'  {device.device_type} {device.name}')

    data = Data(max_elms=10)