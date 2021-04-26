import sys
import tensorflow as tf
import optuna
import train


def optuna_objective_wrapper(trial):

    print('\n')
    print(f'****   Trial number: {trial.number}')
    print('Parameters:')
    for key, value in trial.params.items():
        print(f'  {key}: {value}')

    monitor = 'val_weighted_binary_crossentropy'

    callbacks = [
            tf.keras.callbacks.EarlyStopping(
                min_delta=1e-3,
                patience=10,
                verbose=1,
            ),
            optuna.integration.tfkeras.TFKerasPruningCallback(trial, monitor),
        ]

    training_kwargs = {
        'max_elms': None,
        'epochs': 30,
        'super_window_shuffle_seed': None,
        'fraction_validate': 0.2,
        'fraction_test': 0.02,
        'skip_testing': True,
        'callbacks': callbacks,
        'n_filters_1': trial.suggest_int('n_filters_1', 8, 24, step=4),
        'n_filters_2': trial.suggest_int('n_filters_2', 8, 24, step=4),
        'n_dense_1': trial.suggest_int('n_dense_1', 60, 120, step=20),
        'n_dense_2': trial.suggest_int('n_dense_2', 20, 50, step=10),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.05, 0.5, step=0.05),
        'l2_factor': trial.suggest_float('l2_factor', 1e-3, 1e-1, log=True),
        'relu_negative_slope': trial.suggest_float('relu_negative_slope', 0.0, 22e-3, step=2e-3),
        'initial_learning_rate': trial.suggest_float('initial_learning_rate', 1e-6, 3e-3, log=True),
        'momentum': trial.suggest_float('momentum', 0.0, 1.0, step=0.1),
        'epochs_per_halving': trial.suggest_int('epochs_per_halving', 1, 5, step=1),
        }

    history, _ = train.train_model(**training_kwargs)

    result = history.history[monitor][-1]

    print(f'Trial number: {trial.number}')
    print('Parameters:')
    for key, value in trial.params.items():
        print(f'  {key}: {value}')
    print(f'Trial result: {result:.4f}')

    return result


if __name__ == '__main__':
    if len(sys.argv) == 5:
        db_url = sys.argv[1]
        study_name = sys.argv[2]
        i_gpu = int(sys.argv[3])
        n_startup_trials = int(sys.argv[4])

        print('TF version:', tf.__version__)

        print('Worker configuration')
        print(f'  db url: {db_url}')
        print(f'  study name: {study_name}')
        print(f'  GPU index: {i_gpu}')
        print(f'  n_startup_trials: {n_startup_trials}')

        gpus = tf.config.list_physical_devices('GPU')

        target_gpu = gpus[i_gpu]

        print(f'Running on device {target_gpu.name}')
        tf.config.set_visible_devices(target_gpu, 'GPU')
        tf.config.experimental.set_memory_growth(target_gpu, True)

        visible_gpus = tf.config.get_visible_devices('GPU')
        for gpu in visible_gpus:
            print(f'Visible GPU: {gpu.name}')

        print('Load study')
        study = optuna.load_study(
            storage=db_url,
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=n_startup_trials,
                ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=n_startup_trials,
                n_warmup_steps=12,
                ),
            )

        print('Start study.optimize()')
        study.optimize(
            optuna_objective_wrapper,
            timeout=18*60*60,
            )

        print('End study.optimize()')
