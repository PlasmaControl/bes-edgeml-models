import sys
import train


if len(sys.argv) == 2:
    train.train_model(
        epochs = 20,
        save_std_to_file = True,
        prefix = 'multitrain-01',
        patience = 10,
        i_gpu = int(sys.argv[1]),
        )