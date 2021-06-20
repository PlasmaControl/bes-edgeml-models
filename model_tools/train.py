import sys
import inspect
import pickle
import datetime
import numpy as np
import tensorflow as tf

try:
    from . import model, data_v2, utilities, config

    print("Package-level relative import")
except ImportError:
    import model, data_v2, utilities, config

    print("Direct import")


def train_model(
    # specify model
    model_type="cnn",  # 'cnn' or 'features'
    # training kwargs
    epochs=4,
    steps_per_epoch=50000,  # batches per epoch
    fit_verbose=2,
    callbacks=None,
    # early stopping
    min_delta=5e-4,  # minimum relative improvement in validation metric to continue training
    patience=5,  # epochs allowed to satisfy min_delta improvement
    # save/checkpoint
    prefix="trained_model",
    # save stdout/stderr to file?
    save_std_to_file=True,
):

    assert model_type in ["cnn", "features"]

    folder = prefix + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")

    model_dir = utilities.model_dir / folder
    model_dir.mkdir(parents=True)

    if save_std_to_file:
        # capture stdout/stderr and send to files
        stdout_file = model_dir / "stdout.txt"
        stderr_file = model_dir / "stderr.txt"
        sys.stdout = open(stdout_file, "w")
        sys.stderr = open(stderr_file, "w")

    print("Inputs and defaults")
    signature = inspect.signature(train_model)
    local_vars = locals()
    input_vars = {}
    for key, param in signature.parameters.items():
        input_vars[key] = local_vars[key]
        print(f"  {key}  input {input_vars[key]}  (default {param.default})")
    pickle_file = model_dir / "inputs.pickle"
    with pickle_file.open("wb") as f:
        pickle.dump(input_vars, f)

    # get data
    elm_data = data_v2.Data(
        fraction_validate=config.fraction_validate,
        fraction_test=config.fraction_test,
        kfold=False,
        smoothen_transition=False,
    )
    all_data, datasets = elm_data.get_datasets()
    _, _, test_data = all_data
    train_dataset, validation_dataset, test_dataset = datasets

    test_file = model_dir / "test_data.pickle"
    with test_file.open("wb") as f:
        pickle.dump(
            {
                "signals": np.array(test_data[0]),
                "labels": np.array(test_data[1]),
                "sample_indices": test_data[2],
                "window_start_indices": test_data[3],
                "signal_window_size": config.signal_window_size,
                "label_look_ahead": config.label_look_ahead,
            },
            f,
        )

    # kwargs for all models
    model_kwargs = dict(
        signal_window_size=config.signal_window_size,
        dense_layers=config.dense_layers,
        dropout_rate=config.dropout_rate,
        l2_factor=config.l2_factor,
        relu_negative_slope=config.relu_negative_slope,
    )

    # define model
    if model_type == "cnn":
        m = model.cnn_model(
            # kwargs for all models
            **model_kwargs,
            # kwargs for CNN models
            conv_size=config.conv_size,
            cnn_layers=config.cnn_layers,
        )
    elif model_type == "features":
        m = model.feature_model(
            # kwargs for all models
            **model_kwargs,
            # kwargs for feature models
            maxpool_size=config.maxpool_size,
            filters=config.filters,
        )
    else:
        raise ValueError(
            'Unknown model type is passed. Use either "cnn" or "features".'
        )

    # optimizer
    steps_per_halving = steps_per_epoch * config.epochs_per_halving
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=utilities.Exp_Learning_Rate_Schedule(
            initial_learning_rate=config.initial_learning_rate,
            minimum_learning_rate_factor=config.minimum_learning_rate_factor,
            steps_per_halving=steps_per_halving,
        ),
        momentum=config.momentum,
        nesterov=True,
    )

    m.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    )

    sample_output = m.evaluate(
        x=test_dataset,
        steps=1,
        verbose=0,
        return_dict=True,
    )
    print("Sample evaluation:")
    for key, value in sample_output.items():
        print(f"  {key}, {value:.4f}")

    if not callbacks:

        # default callbacks when not specified by input

        # Tensorboard logs
        # log_dir = model_dir / 'tensorboard-logs'
        # log_dir.mkdir(parents=True)
        # print(f'Tensorboard log dir: {log_dir.as_posix()}')

        checkpoint_dir = model_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        print(f"Checkpoint dir: {checkpoint_dir.as_posix()}")

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
                monitor="val_loss",
            ),
        ]

    history = m.fit(
        x=train_dataset,
        verbose=fit_verbose,
        epochs=epochs,
        validation_data=validation_dataset,
        workers=2,
        use_multiprocessing=True,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,  # batches per epoch
        validation_steps=steps_per_epoch // 4,
    )

    print("Final validation metrics")
    for key, value in history.history.items():
        print(f"  {key}, {value[-1]:.4f}")

    # evaluate validation and test datasets
    for ds_name, dataset in zip(
        ["Validation", "Test"],
        [validation_dataset, test_dataset],
    ):
        print(f"{ds_name} metrics")
        result = m.evaluate(
            x=dataset,
            verbose=0,
            use_multiprocessing=True,
            workers=2,
            return_dict=True,
        )
        for key, value in result.items():
            print(f"  {key}, {value:.4f}")

    save_file = model_dir / "saved_model.tf"
    print(f"Saving model: {save_file.as_posix()}")
    m.save(save_file)

    if save_std_to_file:
        # release stdout/stderr catpure
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    return history, result


if __name__ == "__main__":
    # TF environment
    print("TF version:", tf.__version__)
    for gpu_device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu_device, True)
    print("Available devices:")
    for device in tf.config.list_physical_devices():
        print(f"  {device.device_type}, {device.name}")

    # set GPU visibility
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        if len(sys.argv) >= 2:
            # command line option to specify GPU index
            i_gpu = int(sys.argv[1])
        else:
            # default to last GPU
            i_gpu = len(gpus) - 1
        tf.config.set_visible_devices(gpus[i_gpu], "GPU")

    print("Visible devices:")
    for device in tf.config.get_visible_devices():
        print(f"  {device.device_type}, {device.name}")

    hist, res = train_model(
        epochs=8,
        steps_per_epoch=50000,  # batches per epoch
        model_type="features",  # 'cnn' or 'features'
        patience=4,  # epochs to wait to allow satisfaction of min_delta improvement
        min_delta=1e-4,  # minimum relative improvement (validation loss) to avoid early stopping
        save_std_to_file=False,  # False for regular stdout, True to redirect to file in model dir.
        fit_verbose=1,  # 0 for no message, 1 for progress bar, 2 for epoch summary
    )

    if hasattr(sys.stdout, "close"):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
