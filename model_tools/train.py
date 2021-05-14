import sys
import inspect
import pickle
import datetime
import numpy as np
import tensorflow as tf

try:
    from . import model, data, utilities
    print('Package-level relative import')
except ImportError:
    import model, data, utilities
    print('Direct import')


def train_model(
        # data kwargs
        label_look_ahead=0,
        signal_window_size=8,
        max_elms=None,
        fraction_validate=0.15,  # validation data for post-epoch evaluation
        fraction_test=0.15,  # test data for post-training evaluation
        transition_halfwidth=3,
        training_batch_size=4,
        # specify model
        model_type='cnn',  # 'cnn' or 'features'
        # kwargs for all models
        dense_layers=(40, 20),
        dropout_rate=0.1,
        l2_factor=2e-3,
        relu_negative_slope=0.002,
        # kwargs for cnn models
        conv_size=3,
        cnn_layers=(4, 8),
        # kwargs for feature models
        maxpool_size=2,  # 0 to skip maxpool
        filters=16,
        # optimization kwargs
        epochs_per_halving=4,
        initial_learning_rate=3e-5,
        minimum_learning_rate_factor=30,
        momentum=0.2,
        # training kwargs
        epochs=4,
        steps_per_epoch=50000,  # batches per epoch
        fit_verbose=2,
        callbacks=None,
        # early stopping
        min_delta=5e-4,  # minimum relative improvement in validation metric to continue training
        patience=5,  # epochs allowed to satisfy min_delta improvement
        # save/checkpoint
        prefix='trained_model',
        # save stdout/stderr to file?
        save_std_to_file=True,
        ):

    assert(model_type in ['cnn', 'features'])

    folder = prefix + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")

    model_dir = utilities.model_dir / folder
    model_dir.mkdir(parents=True)

    if save_std_to_file:
        # capture stdout/stderr and send to files
        stdout_file = model_dir / 'stdout.txt'
        stderr_file = model_dir / 'stderr.txt'
        sys.stdout = open(stdout_file, 'w')
        sys.stderr = open(stderr_file, 'w')


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
    elm_data = data.Data(
        signal_window_size=signal_window_size,
        label_look_ahead=label_look_ahead,
        max_elms=max_elms,
        fraction_validate=fraction_validate,
        fraction_test=fraction_test,
        training_batch_size=training_batch_size,
        transition_halfwidth=transition_halfwidth,
        )

    test_file = model_dir / 'test_data.pickle'
    with test_file.open('wb') as f:
        pickle.dump({'signals': np.array(elm_data.test_data[0]),
                     'labels': np.array(elm_data.test_data[1]),
                     'window_start_indices': elm_data.test_data[3],
                     'signal_window_size': elm_data.signal_window_size,
                     'label_look_ahead': elm_data.label_look_ahead},
                    f)


    # kwargs for all models
    model_kwargs = dict(
        signal_window_size=signal_window_size,
        dense_layers=dense_layers,
        dropout_rate=dropout_rate,
        l2_factor=l2_factor,
        relu_negative_slope=relu_negative_slope,
        )

    # define model
    if model_type == 'cnn':
        m = model.cnn_model(
            # kwargs for all models
            **model_kwargs,
            # kwargs for CNN models
            conv_size=conv_size,
            cnn_layers=cnn_layers,
            )
    elif model_type == 'features':
        m = model.feature_model(
            # kwargs for all models
            **model_kwargs,
            # kwargs for feature models
            maxpool_size=maxpool_size,
            filters=filters,
            )
    else:
        raise ValueError


    # optimizer
    steps_per_halving = steps_per_epoch * epochs_per_halving
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=utilities.Exp_Learning_Rate_Schedule(
            initial_learning_rate=initial_learning_rate,
            minimum_learning_rate_factor=minimum_learning_rate_factor,
            steps_per_halving=steps_per_halving,
            ),
        momentum=momentum,
        nesterov=True,
        )


    m.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        )


    sample_output = m.evaluate(
        x=elm_data.test_dataset,
        steps=1,
        verbose=0,
        return_dict=True,
        )
    print('Sample evaluation:')
    for key, value in sample_output.items():
        print(f'  {key}, {value:.4f}')


    if not callbacks:

        # default callbacks when not specified by input

        # Tensorboard logs
        # log_dir = model_dir / 'tensorboard-logs'
        # log_dir.mkdir(parents=True)
        # print(f'Tensorboard log dir: {log_dir.as_posix()}')

        checkpoint_dir = model_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True)
        print(f'Checkpoint dir: {checkpoint_dir.as_posix()}')

        callbacks = [
            # tf.keras.callbacks.TensorBoard(
            #     log_dir=log_dir.as_posix(),
            #     histogram_freq=1,
            #     update_freq=5000,
            #     ),
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
        x=elm_data.train_dataset,
        verbose=fit_verbose,
        epochs=epochs,
        validation_data=elm_data.validation_dataset,
        workers=2,
        use_multiprocessing=True,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,  # batches per epoch
        validation_steps=steps_per_epoch//4,
        )


    print('Final validation metrics')
    for key, value in history.history.items():
        print(f'  {key}, {value[-1]:.4f}')


    # evaluate validation and test datasets
    for ds_name, dataset in zip(['Validation', 'Test'],
                                [elm_data.validation_dataset, elm_data.test_dataset]):
        print(f'{ds_name} metrics')
        result = m.evaluate(
            x=dataset,
            verbose=0,
            use_multiprocessing=True,
            workers=2,
            return_dict=True,
            )
        for key, value in result.items():
            print(f'  {key}, {value:.4f}')


    save_file = model_dir / 'saved_model.tf'
    print(f'Saving model: {save_file.as_posix()}')
    m.save(save_file)

    if save_std_to_file:
        # release stdout/stderr catpure
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    return history, result


if __name__ == '__main__':

    # TF environment
    print('TF version:', tf.__version__)
    for gpu_device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu_device, True)
    print('Available devices:')
    for device in tf.config.list_physical_devices():
        print(f'  {device.device_type}, {device.name}')

    # set GPU visibility
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if len(sys.argv)>=2:
            # command line option to specify GPU index
            i_gpu = int(sys.argv[1])
        else:
            # default to last GPU
            i_gpu = len(gpus)-1
        tf.config.set_visible_devices(gpus[i_gpu], 'GPU')

    print('Visible devices:')
    for device in tf.config.get_visible_devices():
        print(f'  {device.device_type}, {device.name}')


    hist, res = train_model(max_elms=None,
                            epochs=8,
                            steps_per_epoch=50000,  # batches per epoch
                            training_batch_size=4,
                            signal_window_size=16,
                            initial_learning_rate=3e-5,
                            epochs_per_halving=4,  # epochs per LR halving
                            momentum=0.2,
                            dropout_rate=0.05,
                            relu_negative_slope=5e-3,
                            l2_factor=1e-3,
                            model_type='features',  # 'cnn' or 'features'
                            maxpool_size=2,  # spatial down-sample
                            filters=20,  # feature kernels (should rename kw)
                            dense_layers=(80, 40),
                            patience=4,  # epochs to wait to allow satisfaction of min_delta improvement
                            min_delta=1e-4,  # minimum relative improvement (validation loss) to avoid early stopping
                            fraction_test=0.15,
                            fraction_validate=0.05,
                            save_std_to_file=False,  # False for regular stdout, True to redirect to file in model dir.
                            fit_verbose=1,  # 0 for no message, 1 for progress bar, 2 for epoch summary
                            )

    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__