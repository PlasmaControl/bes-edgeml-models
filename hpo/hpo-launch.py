import sys
import tensorflow as tf
import optuna
from  model_tools.train import train_model


def optuna_objective_wrapper(trial):

    print('\n')
    print(f'****   Trial number: {trial.number}')
    print('Parameters:')
    for key, value in trial.params.items():
        print(f'  {key}: {value}')

    # monitor validation loss
    monitor = 'val_loss'

    callbacks = [
            tf.keras.callbacks.EarlyStopping(
                min_delta=5e-4,  # minimum validation loss decrease for early stopping
                patience=4,  # epochs to wait to satisfy min_delta decrease before early stop
                verbose=1,
                ),
            optuna.integration.tfkeras.TFKerasPruningCallback(
                trial,
                monitor,
                ),
        ]


    model_type = 'features'  # 'cnn' or 'features'
    prefix = f'hpo_features_01_trial_{trial.number:03d}'
    epochs = 12  # max epochs per trial

    assert(model_type in ['cnn', 'features'])

    model_kwargs = dict(
        model_type = model_type,
        dense_layers = (
            trial.suggest_int('dense_1', 60, 120, step=20),
            trial.suggest_int('dense_2', 20, 50, step=10)
            ),
        dropout_rate = trial.suggest_float('dropout_rate', 0.02, 0.3, step=0.4),
        l2_factor = trial.suggest_float('l2_factor', 3e-4, 3e-2, log=True),
        relu_negative_slope = trial.suggest_float('relu_negative_slope', 3e-4, 1e-1, log=True),
        )

    if model_type == 'cnn':
        # kwargs for cnn model
        model_kwargs.update(
            conv_size=3,
            cnn_layers=(
                trial.suggest_int('cnn_1', 8, 20, step=4),
                trial.suggest_int('cnn_2', 20, 28, step=4),
                ),
            )
    elif model_type == 'features':
        # kwargs for features model
        model_kwargs.update(
            maxpool_size=trial.suggest_int('maxpool_size', 0, 4, step=2),  # 0 to skip maxpool
            filters=trial.suggest_int('filters', 12, 32, step=4),
            )
    else:
        raise ValueError


    history, _ = train_model(
        # data kwargs
        label_look_ahead=0,
        signal_window_size=16,
        max_elms=None,
        fraction_validate=0.15,  # validation data for post-epoch evaluation
        fraction_test=0.15,  # test data for post-training evaluation
        # transition_halfwidth=trial.suggest_int('transition_halfwidth', 2, 4),
        transition_halfwidth=4,
        training_batch_size=4,
        # model kwargs
        **model_kwargs,
        # optimization kwargs
        epochs_per_halving=trial.suggest_int('epochs_per_halving', 2, 8, step=2),
        initial_learning_rate=trial.suggest_float('initial_learning_rate', 1e-5, 3e-3, log=True),
        minimum_learning_rate_factor=trial.suggest_int('minimum_learning_rate_factor', 10, 100, log=True),
        momentum=trial.suggest_float('momentum', 0.0, 0.8, step=0.1),
        # training kwargs
        epochs=epochs,
        fit_verbose=2,
        callbacks=callbacks,
        # save/checkpoint
        prefix=prefix,
        )

    print(f'Trial number: {trial.number}')
    print('Parameters:')
    for key, value in trial.params.items():
        print(f'  {key}: {value}')

    # output is final validation loss
    output = history.history[monitor][-1]

    return output


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise ValueError

    db_url = sys.argv[1]
    study_name = sys.argv[2]
    i_gpu = int(sys.argv[3])

    n_startup_trials = 80

    print('TF version:', tf.__version__)

    print('Worker configuration')
    print(f'  Database URL: {db_url}')
    print(f'  Study name: {study_name}')
    print(f'  GPU index: {i_gpu}')
    print(f'  n_startup_trials: {n_startup_trials}')

    gpus = tf.config.list_physical_devices('GPU')

    target_gpu = gpus[i_gpu]

    print(f'Running on device {target_gpu.name}')
    tf.config.set_visible_devices(target_gpu, 'GPU')
    tf.config.experimental.set_memory_growth(target_gpu, True)

    print('Visible devices:')
    for device in tf.config.get_visible_devices():
        print(f'  {device.device_type}, {device.name}')

    print('Load study')
    study = optuna.load_study(
        storage=db_url,
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=n_startup_trials,  # random trials before activating sampler
            ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials//2,  # minimum trials before pruning allowed
            n_warmup_steps=6,  # minimum epochs before pruning allowed
            ),
        )

    print('Run study.optimize()')
    study.optimize(
        optuna_objective_wrapper,
        timeout=14*60*60,  # stop *initiating* trials after timeout (seconds); end time is unknown
        )

    print('End study.optimize()')
