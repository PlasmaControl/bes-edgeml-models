"""
Data class to package BES data for training
"""

import numpy as np
import h5py
import tensorflow as tf

try:
    from . import utilities
    print('Package-level relative import')
except ImportError:
    import utilities
    print('Direct import')


class Data(object):

    def __init__(self,
                 datafile=None,
                 signal_window_size=8,
                 label_look_ahead=0,
                 super_window_size=250,
                 max_elms=None,
                 fraction_validate = 0.15,  # validation data fraction
                 fraction_test = 0.15,  # test data fraction
                 training_batch_size=4,
                 super_window_shuffle_seed=None,
                 transition_halfwidth=3,
                 signal_dtype='float32',
                 elming_oversample=6,
                 ):

        self.datafile = datafile
        if self.datafile is None:
            self.datafile = utilities.data_dir / 'labeled-elm-events.hdf5'
        self.super_window_size = super_window_size
        self.signal_window_size = signal_window_size
        self.label_look_ahead = label_look_ahead
        self.max_elms = max_elms
        self.fraction_validate = fraction_validate
        self.fraction_test = fraction_test
        self.training_batch_size = training_batch_size
        self.super_window_shuffle_seed = super_window_shuffle_seed
        self.transition_halfwidth = transition_halfwidth
        self.signal_dtype = signal_dtype
        self.elming_oversample = elming_oversample

        self.training_elms = None
        self.validation_elms = None
        self.test_elms = None

        self.transition = np.linspace(0, 1, 2 * self.transition_halfwidth + 3)

        self.partition_elms()

        print('Reading training ELMs and make dataset')
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

        print(f'Training batch size: {self.training_batch_size}')
        self.n_training_batches = self.training_data[1].shape[0] // self.training_batch_size
        print(f'Number of training batches: {self.n_training_batches}')

        print('Reading validation ELMs and make dataset')
        self.validation_data = self.read_data(
            self.validation_elms,
            shuffle_sample_indices=False,
            )
        self.validation_dataset = self.make_dataset(
            signals=self.validation_data[0],
            labels=self.validation_data[1],
            sample_indices=self.validation_data[2],
            batch_size=16,
            )

        print('Reading test ELMs and make dataset')
        self.test_data = self.read_data(
            self.test_elms,
            shuffle_sample_indices=False,
            )
        self.test_dataset = self.make_dataset(
            signals=self.test_data[0],
            labels=self.test_data[1],
            sample_indices=self.test_data[2],
            batch_size=16,
            )


        # self.signals = None
        # self.labels = None
        # self.valid_t0 = None
        # self.valid_indices = None
        # self.label_fractions = None
        # self.training_label_fractions = None
        # self.n_super_windows = None
        # self.n_train = None
        # self.n_validate = None
        # self.n_test = None
        # self.n_training_batches = None
        # self.ds_train = None
        # self.ds_validate = None
        # self.ds_test = None
        # self.test_signals_superwindows = None
        # self.test_labels_superwindows = None

        # self.read_data_into_tensors()
        # self.normalize_signals()
        # self.shuffle_super_windows()
        # self.partition_and_make_datasets()

    def partition_elms(self):
        print(f'Datafile: {self.datafile.as_posix()}')
        assert(self.datafile.exists())
        # get ELM indices from datafile
        with h5py.File(self.datafile, 'r') as hf:
            print(f'  ELM events in datafile: {len(hf)}')
            elm_index = np.array([int(key) for key in hf], dtype=np.int)
        # shuffle indices
        np.random.shuffle(elm_index)
        if self.max_elms:
            print(f'  Limiting data read to {self.max_elms} ELM events')
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
        print(f'Training ELM events: {self.training_elms.size}')
        print(f'Validation ELM events: {self.validation_elms.size}')
        print(f'Test ELM events: {self.test_elms.size}')

    def read_data(self,
                  elm_indices=(),
                  shuffle_sample_indices=False):
        signals = None
        with h5py.File(self.datafile, 'r') as hf:
            for elm_index in elm_indices:
                elm_key = f'{elm_index:05d}'
                elm_event = hf[elm_key]  # h5py group for ELM event
                signals_np = np.array(elm_event['signals'], dtype=self.signal_dtype)
                signals_np = signals_np.transpose().reshape([-1, 8, 8])  # reshape for 8x8 spatial BES data
                signals_np = signals_np / 10  # rescale to max=1
                labels_np = np.array(elm_event['labels'], dtype=self.signal_dtype)
                # apply transition as ELM turns on and off
                i_ones = np.nonzero(labels_np)[0]   # indices of labels_np == 1
                for direction, idx in zip([1,-1], [i_ones[0], i_ones[-1]]):
                    idx0 = idx - self.transition_halfwidth - 1
                    idx1 = idx + self.transition_halfwidth + 2
                    labels_np[idx0:idx1] = self.transition[::direction]
                # valid indices as basis for data windows
                valid_t0_np = np.ones(labels_np.shape, dtype=np.int8)
                valid_t0_np[-(self.signal_window_size+self.label_look_ahead)+1:] = 0   # 0 for invalid t0
                assert(np.count_nonzero(valid_t0_np==0) == (self.signal_window_size + self.label_look_ahead - 1))
                # indices for active ELM
                active_elm_indices = np.nonzero(labels_np >= 0.5)[0]
                if signals is None:
                    # initialize tensors
                    window_start_indices = np.array([0], dtype=np.int64)
                    elm_start_indices = np.array(active_elm_indices[0], dtype=np.int64)
                    elm_stop_indices = np.array(active_elm_indices[-1], dtype=np.int64)
                    valid_t0 = valid_t0_np
                    signals = tf.convert_to_tensor(signals_np)
                    labels = tf.convert_to_tensor(labels_np)
                else:
                    # concat on dim 0 (time dimension)
                    final_index = labels.shape[0]-1
                    window_start_indices = np.append(window_start_indices, final_index+1)
                    elm_start_indices = np.append(elm_start_indices, final_index+active_elm_indices[0]+1)
                    elm_stop_indices = np.append(elm_stop_indices, final_index+active_elm_indices[-1]+1)
                    valid_t0 = np.concatenate([valid_t0, valid_t0_np])
                    signals = tf.concat([signals, tf.convert_to_tensor(signals_np)], 0)
                    labels = tf.concat([labels, tf.convert_to_tensor(labels_np)], 0)

        labels_np = np.array(labels)

        assert(np.all(labels_np[elm_start_indices] >= 0.5))
        assert(np.all(labels_np[elm_start_indices-1] < 0.5))
        assert(np.all(labels_np[elm_stop_indices] >= 0.5))
        assert(np.all(labels_np[elm_stop_indices+1] < 0.5))

        # find valid indices for data sampling
        valid_indices= np.arange(valid_t0.size, dtype=np.int64)
        valid_indices = valid_indices[valid_t0 == 1]

        # indices for sampling data
        sample_indices = valid_indices

        # oversample active ELM periods to balance data
        fraction_elm = np.count_nonzero(labels_np >= 0.5) / labels.shape[0]
        print(f'  Active ELM fraction (raw data): {fraction_elm:.3f}')
        oversample_count = int((1-fraction_elm)/fraction_elm)-1
        print(f'  Active ELM oversampling for balanced data: {oversample_count}')
        for i_start, i_stop in zip(elm_start_indices, elm_stop_indices):
            assert(np.all(labels[i_start:i_stop+1]>=0.5))
            active_elm_window = np.arange(i_start-20, i_stop+20, dtype=np.int64)
            assert(labels[active_elm_window[0]]<0.5)
            assert(labels[active_elm_window[-1]]<0.5)
            active_elm_window = np.tile(active_elm_window, [oversample_count])
            sample_indices = np.concatenate([sample_indices, active_elm_window])
        fraction_elm = np.count_nonzero(labels_np[sample_indices] >= 0.5) / sample_indices.size
        print(f'  Active ELM fraction (balanced data): {fraction_elm:.3f}')

        sample_indices = tf.constant(sample_indices)

        # shuffle sample indices?
        if shuffle_sample_indices:
            print('Shuffling sample indices')
            sample_indices = tf.random.shuffle(sample_indices)

        print('Data tensors: signals, labels, valid_indices, sample_indices, window_start_indices')
        for tensor in [signals, labels, valid_indices, sample_indices, window_start_indices]:
            self._print_tensor_summary(tensor)

        return signals, labels, sample_indices, window_start_indices

    @staticmethod
    def _print_tensor_summary(tensor):
        tmp = f'  shape {tensor.shape} dtype {tensor.dtype}'
        tmp += f' min {np.min(tensor):.3f} max {np.max(tensor):.3f}'
        if hasattr(tensor, 'device'):
            tmp += f' device {tensor.device[-5:]}'
        print(tmp)

    def _make_generator(self, signals_in, labels_in, sample_indices_in):
        signal_window_size = self.signal_window_size
        label_look_ahead = self.label_look_ahead
        def generator():
            i = 0
            while i < sample_indices_in.shape[0]:
                i_t0 = sample_indices_in[i]
                assert((i_t0 + signal_window_size + label_look_ahead) < labels_in.shape[0])
                signal_window = signals_in[i_t0:i_t0 + signal_window_size, ...]
                signal_window = tf.reshape(signal_window, [signal_window_size, 8, 8, 1])
                label = labels_in[i_t0 + signal_window_size + label_look_ahead]
                label = tf.reshape(label, [1])
                i += 1
                yield signal_window, label
        return generator


    def make_dataset(self,
                     signals=None,
                     labels=None,
                     sample_indices=None,
                     batch_size=4):

        # create datasets
        print('  Make dataset from generator')
        generator = self._make_generator(signals, labels, sample_indices)
        dtypes = (signals.dtype, labels.dtype)
        shapes = (tf.TensorShape([self.signal_window_size, 8, 8, 1]), tf.TensorShape([1]))
        dataset = tf.data.Dataset.from_generator(
            generator,
            dtypes,
            output_shapes=shapes,
            )

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset

    # def read_data_into_tensors(self):
    #
    #     print('Applying transition: ', transition)
    #     total_signal_window = self.signal_window_size + self.label_look_ahead
    #     # loop over ELM events
    #         for ielm, elm_event in enumerate(hf.values()):
    #             if self.max_elms and ielm >= self.max_elms:
    #                 print(f'  Limiting data read to {self.max_elms} ELM events')
    #                 break
    #             # partition data into `super windows` to enable shuffling of time-series windows
    #             n_super_windows = elm_event['labels'].size // self.super_window_size
    #             # load BES signals and reshape into super windows
    #             signals_np = np.array(elm_event['signals'][..., 0:n_super_windows * self.super_window_size],
    #                                   dtype=self.signal_dtype)
    #             signals_np = signals_np.T.reshape(n_super_windows, self.super_window_size, 8, 8)
    #             # load ELM labels
    #             labels_np = np.array(elm_event['labels'][0:n_super_windows * self.super_window_size],
    #                                  dtype=self.signal_dtype)
    #             # apply transition as ELM turns on and off
    #             i_ones = np.nonzero(labels_np)[0]
    #             for direction, idx in zip([1,-1], [i_ones[0], i_ones[-1]]):
    #                 idx0 = idx - self.transition_halfwidth - 1
    #                 idx1 = idx + self.transition_halfwidth + 2
    #                 labels_np[idx0:idx1] = transition[::direction]
    #             # reshape labels into super windows
    #             labels_np = labels_np.reshape(n_super_windows, self.super_window_size)
    #             # construct valid_t0 mask;
    #             valid_t0_np = np.ones(labels_np.shape, dtype=np.int8)
    #             valid_t0_np[:, (self.super_window_size-total_signal_window):self.super_window_size] = 0
    #             # convert to tensors
    #             signals_tmp = tf.convert_to_tensor(signals_np)
    #             labels_tmp = tf.convert_to_tensor(labels_np)
    #             valid_t0_tmp = tf.convert_to_tensor(valid_t0_np)
    #             if ielm == 0:
    #                 # initialize tensors
    #                 signals = signals_tmp
    #                 labels = labels_tmp
    #                 valid_t0 = valid_t0_tmp
    #             else:
    #                 # concat on dim 0 (super windows)
    #                 signals = tf.concat([signals, signals_tmp], 0)
    #                 labels = tf.concat([labels, labels_tmp], 0)
    #                 valid_t0 = tf.concat([valid_t0, valid_t0_tmp], 0)
    #     print('Finished reading data files')

        # # active/inactive ELM summary
        # n_times = np.prod(labels.shape)
        # n_elm_times = np.count_nonzero(np.array(labels) >= 0.5)
        # print(f'Total time points: {n_times}')
        # self.label_fractions = {
        #     0: (n_times - n_elm_times) / n_times,
        #     1: n_elm_times / n_times,
        #     }
        # for key, value in self.label_fractions.items():
        #     print(f'Label {key} fraction: {value:.3f}')
        #
        # self.n_super_windows = labels.shape[0]

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
            self._print_tensor_summary(tensor)

    def _apply_partition_flatten_and_make_valid_indices(self, ind0, ind1,
                                                        is_training_data=False,
                                                        save_test_superwindows=False):
        # partition
        signals = self.signals[ind0:ind1, ...]
        labels = self.labels[ind0:ind1, ...]
        valid_t0 = self.valid_t0[ind0:ind1, ...]

        if save_test_superwindows:
            self.test_signals_superwindows = np.array(signals)
            self.test_labels_superwindows = np.array(labels)

        # oversample superwindows with ELMs
        if is_training_data and self.elming_oversample:
            print(f'Oversampling factor for training data with ELMs: {self.elming_oversample}')
            n_super_windows = labels.shape[0]
            print(signals.shape, labels.shape)
            for i in np.arange(n_super_windows):
                if np.any(labels[i, :] >= 0.1):
                    tmp = tf.tile(signals[i:i+1, :, :, :], [self.elming_oversample, 1, 1, 1])
                    signals = tf.concat([signals, tmp], axis=0)
                    tmp = tf.tile(labels[i:i+1, :], [self.elming_oversample, 1])
                    labels = tf.concat([labels, tmp], axis=0)
                    tmp = tf.tile(valid_t0[i:i+1, :], [self.elming_oversample, 1])
                    valid_t0 = tf.concat([valid_t0, tmp], axis=0)

            n_times = np.prod(labels.shape)
            n_elm_times = np.count_nonzero(np.array(labels) >= 0.5)
            print(f'Total time points: {n_times}')
            self.training_label_fractions = {0: (n_times - n_elm_times) / n_times,
                                             1: n_elm_times / n_times,
                                             }
            for key, value in self.training_label_fractions.items():
                print(f'Training label {key} fraction: {value:.3f}')

            print('  signals, labels, valid_t0:')
            for tensor in [signals, labels, valid_t0]:
                self._print_tensor_summary(tensor)

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

        if is_training_data:
            print('  Shuffling valid indices')
            valid_indices = tf.random.shuffle(valid_indices)

        print('Flat tensors: signals, labels, valid_t0, valid_indices:')
        for tensor in [signals, labels, valid_t0, valid_indices]:
            self._print_tensor_summary(tensor)

        # Note: `valid_t0` arrays are no longer needed after this point

        return signals, labels, valid_indices

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
            is_training_data=True,
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
        self.ds_validate = self.ds_validate.\
            batch(32).\
            prefetch(tf.data.AUTOTUNE)
        self.ds_test = self.ds_test.\
            batch(32).\
            prefetch(tf.data.AUTOTUNE)


if __name__ == '__main__':

    # turn off GPU visibility
    tf.config.set_visible_devices([], 'GPU')

    print('TF version:', tf.__version__)
    print('Visible devices:')
    for device in tf.config.get_visible_devices():
        print(f'  {device.device_type} {device.name}')

    data = Data(max_elms=None)