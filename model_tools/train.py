import sys
import inspect
import pickle
import datetime
import tensorflow as tf
from tensorflow import keras

try:
    from . import model
    from . import data
    from . import paths
    from . import utilities
    print('Package-level relative import')
except ImportError:
    import model
    import data
    import paths
    import utilities
    print('Direct import')


def train_model(
        # data kwargs
        label_look_ahead=0,
        signal_window_size=8,
        super_window_size=250,
        max_elms_per_datafile=None,
        super_window_shuffle_seed=None,
        fraction_validate=0.15,  # validation data for post-epoch evaluation
        fraction_test=0.15,  # test data for post-training evaluation
        transition_halfwidth=3,
        training_batch_size=4,
        # model kwargs
        conv_size=3,
        cnn_layers=(4, 8),
        dense_layers=(40, 20),
        dropout_rate=0.2,
        l2_factor=2e-3,
        relu_negative_slope=0.002,
        use_sigmoid=False,
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
        i_gpu=None,
        ):

    folder = prefix + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")

    model_dir = paths.model_dir / folder
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
    if i_gpu is None:
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
        signal_window_size=signal_window_size,
        label_look_ahead=label_look_ahead,
        super_window_size=super_window_size,
        max_elms_per_datafile=max_elms_per_datafile,
        super_window_shuffle_seed=super_window_shuffle_seed,
        fraction_validate=fraction_validate,
        fraction_test=fraction_test,
        training_batch_size=training_batch_size,
        transition_halfwidth=transition_halfwidth,
        )

    test_file = model_dir / 'test_data.pickle'
    with test_file.open('wb') as f:
        pickle.dump({'signals': d.test_signals_superwindows,
                     'labels': d.test_labels_superwindows},
                    f)


    # define model
    m = model.cnn_model(
        signal_window_size=signal_window_size,
        conv_size=conv_size,
        cnn_layers=cnn_layers,
        dense_layers=dense_layers,
        dropout_rate=dropout_rate,
        l2_factor=l2_factor,
        relu_negative_slope=relu_negative_slope,
        use_sigmoid=use_sigmoid,
        )


    # optimizer
    optimizer = keras.optimizers.SGD(
        learning_rate=utilities.Exp_Learning_Rate_Schedule(
            initial_learning_rate=initial_learning_rate,
            minimum_learning_rate_factor=minimum_learning_rate_factor,
            batches_per_epoch=d.n_training_batches,
            epochs_per_halving=epochs_per_halving,
            ),
        momentum=momentum,
        nesterov=True,
        )


    m.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
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
    print(f'Saving model: {save_file.as_posix()}')
    m.save(save_file)

    if save_std_to_file:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    return history, result


if __name__ == '__main__':

    try:
        hist, res = train_model(max_elms_per_datafile=None,
                                signal_window_size=16,
                                super_window_size=500,
                                epochs=24,
                                initial_learning_rate=3e-5,
                                epochs_per_halving=8,
                                patience=8,
                                min_delta=1e-4,
                                conv_size=3,
                                dropout_rate=0.05,
                                cnn_layers=(12, 20),
                                dense_layers=(100, 40),
                                save_std_to_file=True,
                                i_gpu=0,
                                )
    finally:
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__