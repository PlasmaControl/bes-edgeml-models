import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from options.test_arguments import TestArguments
from src import utils

sns.set_palette("deep")

plt.style.use("/home/lakshya/plt_custom.mplstyle")

if __name__ == "__main__":
    args, parser = TestArguments().parse(verbose=True)
    utils.test_args_compat(args, parser)
    LOGGER = utils.get_logger(script_name=__name__)
    data_cls = utils.create_data(args.data_preproc)
    data_obj = data_cls(args, LOGGER)
    train_data, _, _ = data_obj.get_data()
    signals, labels, valid_indices, window_start = train_data
    start = window_start[0]
    stop = window_start[1] - 1
    first_signal = signals[start:stop]
    signal_max = np.max(first_signal)
    signal_min = np.min(first_signal)
    print(f"Max, min values are respectively: {signal_max} and {signal_min}")
    # first_signal = (first_signal - signal_min) / (signal_max - signal_min)
    first_label = labels[start:stop]
    print(first_signal.shape)
    print(first_label.shape)
    # print(first_signal[::8])

    y1_4 = np.gradient(first_signal[::4], axis=0)

    # plt.plot(first_signal[::8, 0, 0], label='x')
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(18, 16))
    axs = axs.flatten()
    time_grad = y1_4.reshape(-1, 64)
    print(f"Time gradient shape: {time_grad.shape}")
    hop = 0
    for ax in axs:
        for i in range(4):
            ax.plot(time_grad[:, i + hop], label=f"ch: {i+hop+1}")
        ax.plot(first_label[::4], c="k", linestyle="--", label="ground truth")
        ax.legend(fontsize=7, frameon=False)
        hop += 4
    plt.suptitle(
        f"Time derivatives, hop-length: 4, signal window size: 128", fontsize=18
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not args.dry_run:
        plt.savefig(
            os.path.join(
                args.output_dir,
                f"time_gradients_all_channels_hop_4_sws_128.png",
            ),
        )
    plt.show()

    time_grad_diffs = np.diff(time_grad, axis=0, prepend=0)
    print(f"Time gradient diffs shape: {time_grad_diffs.shape}")
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(18, 16))
    axs = axs.flatten()
    hop = 0
    for ax in axs:
        for i in range(4):
            ax.plot(time_grad_diffs[:, i + hop], label=f"ch: {i+hop+1}")
        ax.plot(first_label[::4], c="k", linestyle="--", label="ground truth")
        ax.legend(fontsize=7, frameon=False)
        hop += 4
    plt.suptitle(
        f"Time derivatives' differences, hop-length: 4, signal window size: 128",
        fontsize=18,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not args.dry_run:
        plt.savefig(
            os.path.join(
                args.output_dir, f"time_gradients_diff_hop_4_sws_128.png"
            ),
        )
    plt.show()
