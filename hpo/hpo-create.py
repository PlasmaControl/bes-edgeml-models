import time
from pathlib import Path
import sys
import concurrent.futures
import shlex
import subprocess
import tensorflow as tf
import optuna


if len(sys.argv) > 1:
    db_file = sys.argv[1]
else:
    db_file = 'optuna.db'

db_url = f'sqlite:///{db_file}'
study_name = 'study'

print(f'Database url: {db_url}')

if Path(db_file).exists():
    n_startup_trials = 0
else:
    n_startup_trials = 100

print(f'Startup trials: {n_startup_trials}')

optuna.create_study(storage=db_url,
    study_name=study_name,
    direction='minimize',
    load_if_exists=True,
    )

gpus = tf.config.list_physical_devices('GPU')
n_gpus = len(gpus)
n_workers_per_gpu = 3
n_workers = n_gpus * n_workers_per_gpu

print('Starting pool')
with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
    futures = []
    for i_worker in range(n_workers):
        i_gpu = i_worker % n_gpus
        print(f'  Submitting worker {i_worker+1}/{n_workers} on GPU {gpus[i_gpu].name}')
        command = f'python3 hpo-launch.py {db_url} {study_name} {i_gpu} {n_startup_trials}'
        print(f'  Subprocess command: {command}')
        future = executor.submit(
            subprocess.run,
            shlex.split(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            )
        futures.append(future)
        time.sleep(15)

    print('All workers submitted')
    t1 = time.time()
    while True:
        named_tuple = concurrent.futures.wait(futures, timeout=30*60)
        print(f'  Elapsed time = {(time.time()-t1)/60:.1f} min')
        for i_future, future in enumerate(futures):
            print(f'    Worker {i_future}/{n_workers} running {future.running()} done {future.done()}')
        if len(named_tuple.done) == n_workers:
            break

    print('All workered finished')
    for i_future, future in enumerate(futures):
        print('\n\n')
        print(f'****  Worker {i_future}/{n_workers}')
        if future.exception():
            print('  exception:')
            print(future.exception())
            continue
        result = future.result()
        print(f'  return code: {result.returncode}')
        print(f'  stderr:')
        print(result.stderr.decode('utf-8'))
        print(f'  stdout:')
        print(result.stdout.decode('utf-8'))

print('\n\n')
print('Pool finished')
