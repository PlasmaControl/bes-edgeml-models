import time
import concurrent.futures
import subprocess


n_workers = 4
print("Starting pool...")
with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
    futures = []
    for i_worker in range(n_workers):
        command = [
            "python",
            "lstm_autoencoder.py",
            "--model_name",
            "lstm_ae",
            "--data_preproc",
            "rnn",
            "--signal_window_size",
            "16",
            "--label_look_ahead",
            "150",
            # --hidden_size 32 --device cuda --batch_size 256 --n_epochs 60 --max_elms -1 --normalize_data --truncate_inputs"
        ]
        futures.append(executor.submit(subprocess.run, command))
    print(f"Futures:\n{futures}")
    print("All workers submitted")
    t1 = time.time()
    while True:
        named_tuple = concurrent.futures.wait(futures, timeout=10 * 60)
        print(f"Elapsed time = {(time.time()-t1)/60:.1f} min")
        for i_future, future in enumerate(futures):
            print(
                f"  Worker {i_future}/{n_workers} running {future.running()} done {future.done()}"
            )
        if len(named_tuple.done) == n_workers:
            break

    print("\n\n")
    print("All workered finished")
    for i_future, future in enumerate(futures):
        print("\n\n")
        print(f"****  Worker {i_future}/{n_workers}")
        exception = future.exception()
        if exception:
            print("exception:")
            print(exception)
            continue
        result = future.result()
        print(f"  return code: {result.returncode}")
