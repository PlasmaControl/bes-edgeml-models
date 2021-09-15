import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

from src import utils
from options.test_arguments import TestArguments

plt.style.use("/home/lakshya/plt_custom.mplstyle")

if __name__ == "__main__":
    args, parser = TestArguments().parse(verbose=True)
    utils.test_args_compat(args, parser, infer_mode=True)

    LOGGER = utils.get_logger(script_name=__name__)
    data_cls = utils.create_data(args.data_preproc)
    data_obj = data_cls(args, LOGGER)

    all_elms, all_data = data_obj.get_data()
    signals, labels, valid_indices, window_start = all_data
    start = window_start[0]
    end = window_start[1] - 1
    signal = signals[start : end + 1]
    signal = signal / np.max(signal)
    label = labels[start : end + 1]
    print(signal.shape)
    print(label.shape)

    fig = plt.figure(figsize=(6, 4), dpi=100)
    plt.plot(signal[:, 2, 0], label="Ch. 17", lw=1.25)
    plt.plot(signal[:, 2, 6], label="Ch. 22", lw=1.25)
    plt.plot(label, label="Ground truth", ls="-", lw=1.5)
    plt.xlabel("Time (micro-s)", fontsize=10)
    plt.ylabel("Signal | label", fontsize=10)
    plt.tick_params(axis="x", labelsize=8)
    plt.tick_params(axis="y", labelsize=8)
    plt.ylim([None, 1.1])
    sns.despine(offset=10, trim=False)
    plt.legend(fontsize=10, frameon=False)
    plt.gca().spines["left"].set_color("lightgrey")
    plt.gca().spines["bottom"].set_color("lightgrey")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "plain_elm_event.png"), dpi=100)
    plt.show()
