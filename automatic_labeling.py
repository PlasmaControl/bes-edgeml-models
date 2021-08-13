import os

import numpy as np
import pandas as pd
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
    num_elms = len(window_start)
    dfs = []
    hop_length = 4
    for i_elm in range(num_elms):
        print(f"Processing elm event with start index: {window_start[i_elm]}")
        start = window_start[i_elm]
        if i_elm < num_elms - 1:
            stop = window_start[i_elm + 1] - 1
        else:
            stop = labels.size
        signal = signals[start:stop]
        signal_max = np.max(signal)
        signal_min = np.min(signal)
        # print(
        #     f"Max, min values are respectively: {signal_max} and {signal_min}"
        # )
        # signal = (signal - signal_min) / (signal_max - signal_min)
        label = labels[start:stop]
        # print(f"Signals shape: {signal.shape}")
        # print(f"Labels shape: {label.shape}")
        # print(signal[::8])

        y1_4 = np.gradient(signal[::hop_length], 2, axis=0)
        time_grad = y1_4.reshape(-1, 64)
        # print(f"Time gradient shape: {time_grad.shape}")

        time_grad_diffs = np.diff(time_grad, axis=0, prepend=0)
        # print(f"Time gradient diffs shape: {time_grad_diffs.shape}")

        if args.plot_data and i_elm == 0:
            # plt.plot(signal[::8, 0, 0], label='x')
            fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(18, 16))
            axs = axs.flatten()
            hop = 0
            for ax in axs:
                for i in range(4):
                    ax.plot(time_grad[:, i + hop], label=f"ch: {i+hop+1}")
                ax.plot(
                    label[::4],
                    c="k",
                    linestyle="--",
                    label="ground truth",
                )
                ax.legend(fontsize=7, frameon=False)
                hop += 4
            plt.suptitle(
                f"Time derivatives, hop-length: 4, signal window size: 128, step size: 2",
                fontsize=18,
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if not args.dry_run:
                plt.savefig(
                    os.path.join(
                        args.output_dir,
                        f"time_gradients_all_channels_hop_4_sws_128_step_2{args.filename_suffix}.png",
                    ),
                )
            plt.show()

            fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(18, 16))
            axs = axs.flatten()
            hop = 0
            for ax in axs:
                for i in range(4):
                    ax.plot(time_grad_diffs[:, i + hop], label=f"ch: {i+hop+1}")
                ax.plot(
                    label[::4],
                    c="k",
                    linestyle="--",
                    label="ground truth",
                )
                ax.legend(fontsize=7, frameon=False)
                hop += 4
            plt.suptitle(
                f"Time derivatives' differences, hop-length: 4, signal window size: 128, step size: 2",
                fontsize=18,
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if not args.dry_run:
                plt.savefig(
                    os.path.join(
                        args.output_dir,
                        f"time_gradients_diff_hop_4_sws_128_step_2{args.filename_suffix}.png",
                    ),
                )
            plt.show()

        # find labels channel by channel and store them in a pandas dataframe
        df = pd.DataFrame()
        for i in range(time_grad_diffs.shape[1]):
            auto_label = np.zeros((time_grad_diffs.shape[0],), dtype="int")
            positive_indicies = np.where(np.abs(time_grad_diffs[:, i]) > 0.3)[0]
            auto_label[positive_indicies] = 1
            auto_label = np.repeat(auto_label, repeats=hop_length)
            auto_label = auto_label[: label.shape[0]]
            df[f"ch_{i+1}"] = auto_label
        df["elm_event_index"] = i_elm + 1
        df["manual_label"] = label.astype(int)
        dfs.append(df)

    dfs = pd.concat(dfs, axis=0)
    print(dfs.head())
    print(dfs.info())
    channels = [
        col
        for col in dfs.columns
        if col not in ["elm_event_index", "manual_label"]
    ]
    dfs["automatic_label"] = (np.sum(dfs[channels], axis=1) > 5).astype(int)
    print(dfs["automatic_label"].value_counts())
    print(dfs["manual_label"].value_counts())

    if args.plot_data:
        fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(14, 16))
        axs = axs.flatten()
        for i, idx in enumerate(np.random.randint(1, num_elms + 1, size=12)):
            ax = axs[i]
            elm_evnt = dfs[dfs["elm_event_index"] == idx]
            elm_evnt["manual_label"].plot(ax=ax, label="manual_label")
            elm_evnt["automatic_label"].plot(ax=ax, label="automatic_label")
            ax.legend(fontsize=10, frameon=False)
        plt.suptitle(
            f"Manual vs automatic labeling, hop-length: 4, signal window size: 128, step size: 2",
            fontsize=18,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if not args.dry_run:
            plt.savefig(
                os.path.join(
                    args.output_dir,
                    f"manual_automatic_labeling_hop_4_sws_128_step_2{args.filename_suffix}.png",
                ),
            )
        plt.show()
