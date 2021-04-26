import sys
from pathlib import Path
import inspect
import pickle
import tensorflow as tf
import datetime
from tensorflow import keras
import model
import data


class Exp_Learning_Rate_Schedule(keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self,
                 initial_learning_rate=1e-3,
                 minimum_learning_rate_factor=10,
                 epochs_per_halving=1,
                 batches_per_epoch=10000,
                 ):
        super(Exp_Learning_Rate_Schedule, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.minimum_learning_rate_factor = minimum_learning_rate_factor
        self.epochs_per_halving = epochs_per_halving
        self.batches_per_epoch = batches_per_epoch

        self.minimum_learning_rate = self.initial_learning_rate / self.minimum_learning_rate_factor
        self.batches_per_halving = self.epochs_per_halving * self.batches_per_epoch
        print(f'Learning rate halves every {self.epochs_per_halving} epochs ({self.batches_per_epoch} batches per epoch)')

    def __call__(self, step):
        decay_factor = tf.math.pow(0.5, step / self.batches_per_halving)
        learning_rate = self.initial_learning_rate * decay_factor
        output_rate = tf.math.maximum(learning_rate, self.minimum_learning_rate)
        return output_rate

    def get_config(self):
        config = {'initial_learning_rate': self.initial_learning_rate,
                  'minimum_learning_rate_factor': self.minimum_learning_rate_factor,
                  'epochs_per_halving': self.epochs_per_halving,
                  'batches_per_epoch': self.batches_per_epoch,
                  }
        return config


def train_model(
        # data kwargs
        max_elms=None,
        super_window_shuffle_seed=None,
        fraction_validate=0.15,  # validation data for post-epoch evaluation
        fraction_test=0.15,  # test data for post-training evaluation
        # model kwargs
        n_filters_1=16,
        n_filters_2=24,
        n_dense_1=80,
        n_dense_2=30,
        dropout_rate=0.2,
        l2_factor=2e-3,
        relu_negative_slope=0.002,
        # optimization kwargs
        epochs_per_halving=3,
        initial_learning_rate=3e-5,
        minimum_learning_rate_factor=30,
        momentum=0.2,
        # training kwargs
        epochs=5,
        callbacks=None,
        # testing kwargs
        skip_testing=False,
        # save/checkpoint
        prefix='trained_model',
        # early stopping
        min_delta=1e-3,
        patience=5,
        # save stdout/stderr to file?
        save_std_to_file=False,
        # gpu
        i_gpu=-1,
        ):

    folder = prefix + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")

    model_dir = Path('models') / folder
    model_dir.mkdir(parents=True)

    if save_std_to_file:
        stdout_file = model_dir / 'stdout.txt'
        stderr_file = model_dir / 'stderr.txt'
        sys.stdout = open(stdout_file, 'w')
        sys.stderr = open(stderr_file, 'w')



    # TF environment
    print('TF version:', tf.__version__)
    for gpu_device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu_device, True)
    print('Available devices:')
    for device in tf.config.list_physical_devices():
        print(f'  {device.device_type}, {device.name}')

    # set GPU visibility
    gpus = tf.config.list_physical_devices('GPU')
    if i_gpu == -1:
        i_gpu = len(gpus)-1
    gpu_device = gpus[i_gpu]
    gpu_tag = f'/GPU:{i_gpu}'
    tf.config.set_visible_devices(gpu_device, 'GPU')

    print('Visible devices:')
    for device in tf.config.get_visible_devices():
        print(f'  {device.device_type}, {device.name}')



    print('Inputs and defaults')
    signature = inspect.signature(train_model)
    local_vars = locals()
    input_vars = {}
    for key, param in signature.parameters.items():
        input_vars[key] = local_vars[key]
        print(f'  {key}  input {input_vars[key]}  (default {param.default})')
    pickle_file = model_dir / 'inputs.pickle'
    with pickle_file.open('wb') as f:
        pickle.dump(input_vars, f)


    # get data
    d = data.Data(
        max_elms=max_elms,
        super_window_shuffle_seed=super_window_shuffle_seed,
        fraction_validate=fraction_validate,
        fraction_test=fraction_test,
        )

    test_file = model_dir / 'test_data.pickle'
    with test_file.open('wb') as f:
        pickle.dump({'signals': d.test_signals_superwindows,
                     'labels': d.test_labels_superwindows},
                    f)


    # define model
    m = model.cnn_model(
        n_filters_1=n_filters_1,
        n_filters_2=n_filters_2,
        n_dense_1=n_dense_1,
        n_dense_2=n_dense_2,
        dropout_rate=dropout_rate,
        l2_factor=l2_factor,
        relu_negative_slope=relu_negative_slope,
        )


    # optimizer
    optimizer = keras.optimizers.SGD(
        learning_rate=Exp_Learning_Rate_Schedule(
            initial_learning_rate=initial_learning_rate,
            minimum_learning_rate_factor=minimum_learning_rate_factor,
            batches_per_epoch=d.n_training_batches,
            epochs_per_halving=epochs_per_halving,
            ),
        momentum=momentum,
        nesterov=True,
        )


    metrics = [
        keras.metrics.BinaryCrossentropy(),
        keras.metrics.BinaryAccuracy(),
        # keras.metrics.FalseNegatives(),
        # keras.metrics.FalsePositives(),
        # keras.metrics.TrueNegatives(),
        # keras.metrics.TruePositives(),
        ]

    # weighted_metrics = [
    #     keras.metrics.BinaryCrossentropy(),
    #     keras.metrics.BinaryAccuracy(),
    #     ]
    weighted_metrics = None

    m.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics,
        weighted_metrics=weighted_metrics,
        )


    sample_output = m.evaluate(
        x=d.ds_test,
        steps=1,
        verbose=0,
        return_dict=True,
        )
    print('Sample evaluation:')
    for key, value in sample_output.items():
        print(f'  {key}, {value:.4f}')


    if not callbacks:

        # Tensorboard logs
        log_dir = model_dir / 'tensorboard-logs'
        log_dir.mkdir(parents=True)
        print(f'Tensorboard log dir: {log_dir.as_posix()}')

        checkpoint_dir = model_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True)
        print(f'Checkpoint dir: {checkpoint_dir.as_posix()}')

        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir.as_posix(),
                histogram_freq=1,
                update_freq=5000,
                ),
            tf.keras.callbacks.EarlyStopping(
                min_delta=min_delta,
                patience=patience,
                verbose=1,
                ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_dir,
                monitor='val_loss',
                ),
            ]

    history = m.fit(
        x=d.ds_train,
        verbose=2,
        epochs=epochs,
        validation_data=d.ds_validate,
        workers=2,
        use_multiprocessing=True,
        # class_weight=d.class_weights,
        callbacks=callbacks,
        )


    print('Final validation metrics')
    for key, value in history.history.items():
        print(f'  {key}, {value[-1]:.4f}')


    if skip_testing:
        result = None
    else:
        result = m.evaluate(
            x=d.ds_test,
            verbose=0,
            use_multiprocessing=True,
            workers=2,
            return_dict=True,
            )

        print('Test metrics')
        for key, value in result.items():
            print(f'  {key}, {value:.4f}')

    save_file = model_dir / 'saved_model.tf'
    m.save(save_file)

    if save_std_to_file:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    return history, result


if __name__ == '__main__':

    # # make only last GPU visible
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     # limit GPU visibility to last GPU
    #     tf.config.set_visible_devices(gpus[-1], 'GPU')
    #     # set memory growth
    #     tf.config.experimental.set_memory_growth(gpus[-1], True)
    #
    # print('Visible devices:')
    # for dev in tf.config.get_visible_devices():
    #     print(f'  {dev.device_type}, {dev.name}')

    try:
        hist, res = train_model(max_elms=5, epochs=1, save_std_to_file=True, skip_testing=False)
    finally:
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__