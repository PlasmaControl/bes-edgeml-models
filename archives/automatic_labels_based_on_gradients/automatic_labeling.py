import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.getcwd())

from options.test_arguments import TestArguments
import utils

sns.set_palette("deep")

if __name__ == "__main__":
    output_dir = "automatic_label_plots"
    args, parser = TestArguments().parse(verbose=True)
    LOGGER = utils.get_logger(script_name=__name__)
    data_cls = utils.create_data_class(args.data_preproc)
    paths = utils.create_output_paths(args, infer_mode=True)
    roc_dir = paths[-1]
    data_obj = data_cls(args, LOGGER)
    all_elms, all_data = data_obj.get_data()
    signals, labels, valid_indices, window_start = all_data
    print(signals.shape)
    print(labels.shape)
    num_elms = len(window_start)
    print(f"Total ELMS: {num_elms}")
    print(f"Total ELMS from elm index: {len(all_elms)}")
    dfs = []
    signals_list = []
    hop_length = 4
    total_length = 0
    for i_elm in range(num_elms):
        print(
            f"{i_elm}. Processing elm event with start index: {window_start[i_elm]}"
        )
        start = window_start[i_elm]
        if i_elm < num_elms - 1:
            stop = window_start[i_elm + 1] - 1
        else:
            stop = labels.size
        signal = signals[start : stop + 1]
        signals_list.append(signal)
        signal_max = np.max(signal)
        signal_min = np.min(signal)
        label = labels[start : stop + 1]
        print(f"Label length: {len(label)}")
        total_length += len(label)
        # print(signal[::8])

        y1_4 = np.gradient(signal[::hop_length], axis=0)
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
                for i, a in zip(range(4), [0.9, 0.8, 0.75, 0.7]):
                    ax.plot(
                        time_grad[:, i + hop], label=f"ch: {i+hop+1}", alpha=a
                    )
                ax.plot(
                    label[::4],
                    c="k",
                    linestyle="--",
                    label="ground truth",
                    alpha=0.7,
                )
                ax.legend(fontsize=7, frameon=False)
                ax.spines["left"].set_color("gray")
                ax.spines["bottom"].set_color("gray")
                hop += 4
            plt.suptitle(
                f"Time derivatives, hop-length: 4, signal window size: {args.signal_window_size}",
                fontsize=18,
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if not args.dry_run:
                plt.savefig(
                    os.path.join(
                        output_dir,
                        f"all_channels_time_gradients_hop_4_sws_{args.signal_window_size}{args.filename_suffix}.png",
                    ),
                    dpi=150,
                )
            plt.show()

            fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(18, 16))
            axs = axs.flatten()
            hop = 0
            for ax in axs:
                for i, a in zip(range(4), [0.9, 0.8, 0.75, 0.7]):
                    ax.plot(
                        time_grad_diffs[:, i + hop],
                        label=f"ch: {i+hop+1}",
                        alpha=a,
                    )
                ax.plot(
                    label[::4],
                    c="k",
                    linestyle="--",
                    label="ground truth",
                    alpha=0.7,
                )
                ax.legend(fontsize=7, frameon=False)
                ax.spines["left"].set_color("gray")
                ax.spines["bottom"].set_color("gray")
                hop += 4
            plt.suptitle(
                f"Time derivatives' differences, hop-length: 4, signal window size: {args.signal_window_size}",
                fontsize=18,
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if not args.dry_run:
                plt.savefig(
                    os.path.join(
                        output_dir,
                        f"all_channels_time_gradients_tdiff_hop_4_sws_{args.signal_window_size}{args.filename_suffix}.png",
                    ),
                    dpi=150,
                )
            plt.show()

        # find labels channel by channel and store them in a pandas dataframe
        df = pd.DataFrame()
        for i in range(time_grad.shape[1]):
            auto_label = np.zeros((time_grad.shape[0],), dtype="int")
            positive_indicies = np.where(np.abs(time_grad[:, i]) > 0.4)[0]
            auto_label[positive_indicies] = 1
            auto_label = np.repeat(auto_label, repeats=hop_length)
            auto_label = auto_label[: label.shape[0]]
            df[f"ch_{i+1}"] = auto_label
        df["elm_id"] = i_elm + 1
        df["manual_label"] = label.astype(int)
        channels = [
            col
            for col in df.columns
            if col not in ["elm_event_index", "manual_label"]
        ]
        df["automatic_label"] = (np.sum(df[channels], axis=1) > 12).astype(int)
        auto_label = df["automatic_label"].values
        manual_label = df["manual_label"].values
        for i in range(len(manual_label)):
            if (manual_label[i] or auto_label[i]) and (manual_label[i] == 0):
                auto_label[i] = 0
        x = list(np.where(auto_label == 1)[0])
        if not x:
            print(
                f"Found no active elm with serial no: {i_elm} and start index: {window_start[i_elm]} "
            )
            df["automatic_label"] = 1
        else:
            auto_label[x[0] : x[-1]] = 1
            df["automatic_label"] = auto_label
        dfs.append(df)

    dfs = pd.concat(dfs, axis=0)
    # print(dfs.head())
    # print(dfs.info())

    print(dfs["automatic_label"].value_counts())
    print(dfs["manual_label"].value_counts())
    elm_ids = dict(enumerate(all_elms, start=1))
    dfs["elm_event"] = dfs["elm_id"].map(elm_ids).apply(lambda x: f"{x:05d}")
    print(dfs)
    dfs.loc[:, ["elm_id", "elm_event", "automatic_label"]].to_csv(
        os.path.join(
            roc_dir,
            f"automatic_labels_df_sws_{args.signal_window_size}_{args.label_look_ahead}.csv",
        ),
        index=False,
    )

    if args.plot_data:
        elm_index_list = list(range(num_elms))
        num_pages = int(len(elm_index_list) / 12) + 1
        print(f"Total ELM events: {num_elms}")
        print(f"Total pages: {num_pages}")
        elm_event_list = [
            elm_index_list[i * 12 : (i + 1) * 12] for i in range(num_pages)
        ]
        for page_num, page in enumerate(elm_event_list):
            if len(page) == 12:
                fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(14, 16))
            else:
                remaining_elms = 12 * num_pages - num_elms
                rows = 4
                cols = remaining_elms // rows
                fig, axs = plt.subplots(nrows=rows, ncols=1, figsize=(10, 12))
            fig.subplots_adjust(hspace=1.5)
            axs = axs.flatten()
            for i in range(len(page)):
                print(f"Plotting elm event {i+1:02d} on page {page_num+1}")
                ax = axs[i]
                plt.setp(ax.get_xticklabels(), fontsize=9)
                plt.setp(ax.get_yticklabels(), fontsize=9)
                elm_evnt = dfs[dfs["elm_id"] == i + 1 + (12 * page_num)]
                elm_evnt["manual_label"].plot(ax=ax, label="manual_label")
                elm_evnt["automatic_label"].plot(ax=ax, label="automatic_label")
                ax.plot(
                    signals_list[i + (12 * page_num)][:, 0, 0] / 10.0,
                    lw=1.0,
                    label="ch: 1",
                    alpha=0.6,
                )
                ax.plot(
                    signals_list[i + (12 * page_num)][:, 2, 6] / 10.0,
                    lw=1.0,
                    label="ch: 22",
                    alpha=0.6,
                )
                ax.plot(
                    signals_list[i + (12 * page_num)][:, 7, 7] / 5.0,
                    lw=1.0,
                    label="ch: 64",
                    alpha=0.6,
                )
                ax.set_title(
                    f"ELM index: {all_elms[i + (12 * page_num)]:05d}",
                    fontsize=12,
                )
                ax.legend(fontsize=10, frameon=False)
                ax.spines["left"].set_color("gray")
                ax.spines["bottom"].set_color("gray")
            if page_num == 0:
                plt.suptitle(
                    f"Manual vs automatic labeling, hop-length: 4, signal window size: {args.signal_window_size}",
                    fontsize=18,
                )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=1.5, h_pad=1.5)
            if not args.dry_run:
                fname = os.path.join(
                    output_dir,
                    f"page_{page_num+1}_elm_{page[0]+1}_{page[-1]+1}.png",
                )
                print(f"Creating file: {fname}")
                plt.savefig(fname, dpi=150)
            # if page_num in [0, 1]:
            #     plt.show()
            # else:
            #     break
            plt.close()
    print(f"Total length: {total_length}")
