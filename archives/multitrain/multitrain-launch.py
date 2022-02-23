import time
import concurrent.futures
import subprocess
import tensorflow as tf



# TF environment
print('TF version:', tf.__version__)
for gpu_device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu_device, True)
print('Available devices:')
for device in tf.config.list_physical_devices():
    print(f'  {device.device_type}, {device.name}')
print('Visible devices:')
for device in tf.config.get_visible_devices():
    print(f'  {device.device_type}, {device.name}')



gpus = tf.config.list_physical_devices('GPU')
n_gpus = len(gpus)
n_workers_per_gpu = 3
n_workers = n_gpus * n_workers_per_gpu


print('\n')
print('Starting pool')
with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
    futures = []
    for i_worker in range(n_workers):
        i_gpu = i_worker % n_gpus
        print(f'Submitting worker {i_worker+1}/{n_workers} on GPU {gpus[i_gpu].name}')
        command = ['python3',
                   'multitrain-wrapper.py',
                   f'{i_gpu}',
                   ]
        futures.append(
            executor.submit(subprocess.run,
                            command,
                            # stdout=subprocess.PIPE,
                            # stderr=subprocess.PIPE,
                            )
            )
        if i_worker != n_workers-1:
            time.sleep(20)

    print('\n')
    print('All workers submitted')
    t1 = time.time()
    while True:
        named_tuple = concurrent.futures.wait(futures, timeout=10*60)
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
        # print(f'  stderr:')
        # print(result.stderr.decode('utf-8'))
        # print(f'  stdout:')
        # print(result.stdout.decode('utf-8'))

print('\n\n')
print('Pool finished')
