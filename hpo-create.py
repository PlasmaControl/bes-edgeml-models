import time
import sys
import concurrent.futures
import subprocess
import tensorflow as tf
import optuna


if len(sys.argv) == 4:
    db_file = sys.argv[1]
    study_name = sys.argv[2]
    n_startup_trials = int(sys.argv[3])
else:
    if len(sys.argv) > 1:
        raise ValueError
    db_file = 'optuna-test.db'
    study_name = 'study-02'
    n_startup_trials = 4

db_url = f'sqlite:///{db_file}'

print('\n\n')
print(f'DB url: {db_url}')
print(f'Study name: {study_name}')

optuna.create_study(storage=db_url,
    study_name=study_name,
    direction='minimize',
    load_if_exists=True,
    )

gpus = tf.config.list_physical_devices('GPU')
n_gpus = len(gpus)
n_workers_per_gpu = 3
n_workers = n_gpus * n_workers_per_gpu

print('\n\n')
print('Starting pool')
with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
    futures = []
    for i_worker in range(n_workers):
        i_gpu = i_worker % n_gpus
        print(f'Submitting worker {i_worker+1}/{n_workers} on GPU {gpus[i_gpu].name}')
        command = ['python3',
                   'optuna-launch.py',
                   db_url,
                   study_name,
                   f'{i_gpu}',
                   f'{n_startup_trials}',
                   ]
        futures.append(
            executor.submit(subprocess.run,
                            command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
        )
        time.sleep(1)

    print('\n\n')
    print('All workers submitted')
    t1 = time.time()
    while True:
        named_tuple = concurrent.futures.wait(futures, timeout=30*60)
        print(f'Elapsed time = {(time.time()-t1)/60:.1f} min')
        for i_future, future in enumerate(futures):
            print(f'  Worker {i_future}/{n_workers} running {future.running()} done {future.done()}')
        if len(named_tuple.done) == n_workers:
            break

    print('\n\n')
    print('All workered finished')
    for i_future, future in enumerate(futures):
        print('\n\n')
        print(f'****  Worker {i_future}/{n_workers}')
        exception = future.exception()
        if exception:
            print('exception:')
            print(exception)
            continue
        result = future.result()
        print(f'  return code: {result.returncode}')
        print(f'  stderr:')
        print(result.stderr.decode('utf-8'))
        print(f'  stdout:')
        print(result.stdout.decode('utf-8'))

print('\n\n')
print('Pool finished')
