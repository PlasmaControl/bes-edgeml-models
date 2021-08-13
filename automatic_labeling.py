import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from options.test_arguments import TestArguments
from src import utils

sns.set_palette("deep")

# plt.style.use("/home/lakshya/plt_custom.mplstyle")

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
    first_label = labels[start:stop]
    print(first_signal.shape)
    print(first_label.shape)
    # print(first_signal[::8])
    y1_2 = np.gradient(first_signal[::2], axis=0)
    y2_2 = np.gradient(first_signal[::2], 2, axis=0)
    y4_2 = np.gradient(first_signal[::2], 4, axis=0)
    y8_2 = np.gradient(first_signal[::2], 8, axis=0)

    y1_4 = np.gradient(first_signal[::4], axis=0)
    y2_4 = np.gradient(first_signal[::4], 2, axis=0)
    y4_4 = np.gradient(first_signal[::4], 4, axis=0)
    y8_4 = np.gradient(first_signal[::4], 8, axis=0)

    y1_8 = np.gradient(first_signal[::8], axis=0)
    y2_8 = np.gradient(first_signal[::8], 2, axis=0)
    y4_8 = np.gradient(first_signal[::8], 4, axis=0)
    y8_8 = np.gradient(first_signal[::8], 8, axis=0)

    # plt.plot(first_signal[::8, 0, 0], label='x')
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(12, 16))
    ax = ax.flatten()
    ax[0].plot(y1_2[:, 0, 0], label="y1_00, hop=2")
    ax[0].plot(y2_2[:, 0, 0], label="y2_00, hop=2")
    ax[0].plot(y4_2[:, 0, 0], label="y4_00, hop=2")
    ax[0].plot(y8_2[:, 0, 0], label="y8_00, hop=2")
    ax[0].legend()

    ax[1].plot(y1_2[:, -1, -1], label="y1_77, hop=2")
    ax[1].plot(y2_2[:, -1, -1], label="y2_77, hop=2")
    ax[1].plot(y4_2[:, -1, -1], label="y4_77, hop=2")
    ax[1].plot(y8_2[:, -1, -1], label="y8_77, hop=2")
    ax[1].legend()

    ax[2].plot(y1_4[:, 0, 0], label="y1_00, hop=4")
    ax[2].plot(y2_4[:, 0, 0], label="y2_00, hop=4")
    ax[2].plot(y4_4[:, 0, 0], label="y4_00, hop=4")
    ax[2].plot(y8_4[:, 0, 0], label="y8_00, hop=4")
    ax[2].legend()

    ax[3].plot(y1_4[:, -1, -1], label="y1_77, hop=4")
    ax[3].plot(y2_4[:, -1, -1], label="y2_77, hop=4")
    ax[3].plot(y4_4[:, -1, -1], label="y4_77, hop=4")
    ax[3].plot(y8_4[:, -1, -1], label="y8_77, hop=4")
    ax[3].legend()

    ax[4].plot(y1_8[:, 0, 0], label="y1_00, hop=8")
    ax[4].plot(y2_8[:, 0, 0], label="y2_00, hop=8")
    ax[4].plot(y4_8[:, 0, 0], label="y4_00, hop=8")
    ax[4].plot(y8_8[:, 0, 0], label="y8_00, hop=8")
    ax[4].legend()

    ax[5].plot(y1_8[:, -1, -1], label="y1_77, hop=8")
    ax[5].plot(y2_8[:, -1, -1], label="y2_77, hop=8")
    ax[5].plot(y4_8[:, -1, -1], label="y4_77, hop=8")
    ax[5].plot(y8_8[:, -1, -1], label="y8_77, hop=8")
    ax[5].legend()

    ax[6].plot(first_signal[::2, 0, 0], label="signal_00, hop=2")
    ax[6].plot(first_signal[::2, -1, -1], label="signal_77, hop=2")
    ax[6].legend()

    ax[7].plot(first_signal[::4, 0, 0], label="signal_00, hop=4")
    ax[7].plot(first_signal[::4, -1, -1], label="signal_77, hop=4")
    ax[7].legend()
    plt.tight_layout()
    plt.show()
