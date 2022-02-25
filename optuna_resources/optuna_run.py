import sys
from pathlib import Path
import concurrent.futures
import time
from typing import Callable, Union

import numpy as np
import pywt
import optuna

# add this file's directory to PYTHONPATH
sys.path.insert(0, Path(__file__).parent.as_posix())

from single_train import single_train


def objective_wrapper(
        trial: Union[optuna.trial.Trial, optuna.trial.FrozenTrial],
        db_dir: str,
        i_gpu: int,
        objective_func: Callable,
    ):

    db_dir = Path(db_dir)
    assert db_dir.exists()
    trial_dir = db_dir / f'trial_{trial.number:04d}'
    trial_dir.mkdir()

    trial_output = trial_dir / 'output.txt'
    f = trial_output.open('w')
    sys.stdout = f
    sys.stderr = f

    input_keywords = {
        'input_data_file':(Path.home()/'scratch/edgeml/data/labeled-elm-events.hdf5').as_posix(),
        'output_dir':trial_dir.as_posix(),
        'device':f'cuda:{i_gpu:d}',
    }

    input_keywords = objective_func(trial, input_keywords)

    print(f'Trial {trial.number}')
    for key, value in trial.params.items():
        print(f'  Optuna param: {key}, value: {value}')
    for key, value in input_keywords.items():
        print(f'  Model input: {key}, value: {value}')

    outputs = single_train(trial=trial, **input_keywords)

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    f.close()

    return outputs['f1_scores'][-1]


def worker(
        db_url: str,
        db_dir: str,
        n_trials: int,
        i_gpu: int,
        n_startup_trials: int,
        objective_func: Callable,
    ):

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=n_startup_trials,  # random trials before activating sampler
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=60,  # trials before enabling pruner
        n_warmup_steps=5,  # epochs before enabling pruner
        n_min_trials=15,  # min. trials at each epoch to enable pruner
    )

    # load study
    study = optuna.load_study(
        study_name='study',
        storage=db_url,
        sampler=sampler,
        pruner=pruner,
    )

    lambda_objective = lambda trial: objective_wrapper(trial, db_dir, i_gpu, objective_func)

    # run an optimization process
    study.optimize(
        lambda_objective,
        n_trials=n_trials,  # trials for this study.optimize() call
        gc_after_trial=True,
        catch=(AssertionError,)  # fail trials with assertion error and continue study
    )


def optuna_run(
        db_name: str,
        n_gpus: int,
        n_workers_per_gpu: int,
        n_trials_per_worker: int,
        n_startup_trials: int,
        objective_func: Callable,
        ):

    db_file = Path(db_name) / f'{db_name}.db'
    db_file.parent.mkdir(parents=True, exist_ok=True)
    db_url = f'sqlite:///{db_file.as_posix()}'
    if db_file.exists():
        print(f'Studies in storage: {db_url}')
        for study in optuna.get_all_study_summaries(db_url):
            print(f'  Study {study.study_name} with {study.n_trials} trials')

    storage = optuna.storages.RDBStorage(
        url=db_url,
    )

    study = optuna.create_study(
        study_name='study',
        storage=storage,
        load_if_exists=True,
        direction='maximize',
    )

    # FAIL any zombie trials that are stuck in `RUNNING` state
    trials = storage.get_all_trials(study._study_id,
                                    deepcopy=False,
                                    states=(optuna.trial.TrialState.RUNNING,))
    for trial in trials:
        print(f'Setting trial {trial.number} with state {trial.state} to FAIL')
        status = storage.set_trial_state(trial._trial_id,
                                         optuna.trial.TrialState.FAIL)
        print(f'Success?: {status}')

    # launch workers
    n_workers = n_gpus * n_workers_per_gpu
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
    # with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i_worker in range(n_workers):
            i_gpu = i_worker % n_gpus
            print(f'Launching worker {i_worker+1} (of {n_workers}) on gpu {i_gpu} and running {n_trials_per_worker} trials')
            future = executor.submit(
                worker,  # callable that calls study.optimize()
                db_url,
                db_file.parent.as_posix(),
                n_trials_per_worker,
                i_gpu,
                n_startup_trials,
                objective_func,
            )
            futures.append(future)
            time.sleep(0.1)
        concurrent.futures.wait(futures)
        for i_future, future in enumerate(futures):
            if future.exception() is None:
                print(f'Future {i_future} returned {future.result()}')
            else:
                print(f'Future {i_future} exception:')
                print(future.exception())


def optuna_test(
        trial: Union[optuna.trial.Trial, optuna.trial.FrozenTrial],
        input_keywords: dict,
    ) -> dict:

    input_keywords['signal_window_size'] = 128
    input_keywords['label_look_ahead'] = 50
    input_keywords['max_elms'] = 5
    input_keywords['n_epochs'] = 2

    raw_num_filters_factor_8 = trial.suggest_int('raw_num_filters_factor_8', 3, 5)
    raw_num_filters = 8 * raw_num_filters_factor_8
    input_keywords['raw_num_filters'] = raw_num_filters

    return input_keywords


if __name__=='__main__':
    func = optuna_test
    optuna_run(
        db_name=func.__name__,
        n_gpus=2,  # 2 for head node
        n_workers_per_gpu=2,  # 3 max
        n_trials_per_worker=2,
        n_startup_trials=4,
        objective_func=func,
    )
