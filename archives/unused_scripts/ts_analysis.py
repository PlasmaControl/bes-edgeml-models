"""Classical time series analysis (autocorrelation, lag_plots) of the ELM events.
DEPRECATED!!
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
from pandas.plotting import lag_plot, autocorrelation_plot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FastICA, PCA

sys.path.append(os.getcwd())

from options.test_arguments import TestArguments
import utils

# sns.set_palette("Dark2")
colors = list(sns.color_palette("Dark2").as_hex())


def plot_autocorr(args: argparse.Namespace, column: str, df: pd.DataFrame):
    fig = plt.figure(figsize=(8, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)

    ax1 = fig.add_subplot(gs[0, :])
    autocorrelation_plot(df[column], c=colors[0], ax=ax1)
    ax1.spines["left"].set_color("gray")
    ax1.spines["bottom"].set_color("gray")
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)

    ax2 = fig.add_subplot(gs[1, 0])
    lag_plot(df[column], lag=100, c=colors[1], ax=ax2, ec="k")
    ax2.spines["left"].set_color("gray")
    ax2.spines["bottom"].set_color("gray")
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 1])
    lag_plot(df[column], lag=200, c=colors[2], ax=ax3, ec="k")
    ax3.spines["left"].set_color("gray")
    ax3.spines["bottom"].set_color("gray")
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)

    ax4 = fig.add_subplot(gs[2, 0])
    lag_plot(df[column], lag=400, c=colors[3], ax=ax4, ec="k")
    ax4.spines["left"].set_color("gray")
    ax4.spines["bottom"].set_color("gray")
    ax4.spines["right"].set_visible(False)
    ax4.spines["top"].set_visible(False)

    ax5 = fig.add_subplot(gs[2, 1])
    lag_plot(df[column], lag=500, c=colors[4], ax=ax5, ec="k")
    ax5.spines["left"].set_color("gray")
    ax5.spines["bottom"].set_color("gray")
    ax5.spines["right"].set_visible(False)
    ax5.spines["top"].set_visible(False)

    plt.suptitle(f"ELM ID:46, BES {column}", fontsize=18)
    plt.tight_layout()  # rect=[0, 0.03, 1, 0.95], pad=1.5, h_pad=1.5)
    if not args.dry_run:
        plt.savefig(
            os.path.join(
                args.output_dir,
                f"auto_correlation_plots_elm_id_46_{column}.png",
            ),
            dpi=150,
        )
    plt.show()


def decompose_signal(
        args: argparse.Namespace, signal_channel: str, df: pd.DataFrame
):
    signal = df[signal_channel].values
    norm_signal = (signal - np.mean(signal)) / np.std(signal)
    norm_signal = norm_signal.reshape(-1, 1)
    ica = FastICA(n_components=2)
    decomp_signal = ica.fit_transform(norm_signal)

    models = [signal, decomp_signal.T]
    names = [f"Original signal, {signal_channel}", "ICA recovered signal"]
    fig = plt.figure(figsize=(6, 10))
    for ii, (model, name) in enumerate(zip(models, names), 1):
        ax = fig.add_subplot(2, 1, ii)
        ax.set_title(name)
        for sig, color in zip(model, colors[:2]):
            ax.plot(sig, color=color)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args, parser = TestArguments().parse(verbose=True)
    LOGGER = utils.get_logger(script_name=__name__)
    data_cls = utils.create_data_class(args.data_preproc)
    data_obj = data_cls(args, LOGGER)
    train_data, _, _ = data_obj.get_data()
    signals, labels, valid_indices, window_start = train_data
    print(f"Signals shape: {signals.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Valid indices shape: {valid_indices.shape}")
    print(f"Window start shape: {window_start.shape}")

    start = window_start[45]
    end = window_start[46] - 1
    signal = signals[start: end + 1, ...]
    signal = signal.reshape(-1, 64)
    label = labels[start: end + 1]
    with open("single_elm_event.npy", "wb") as f:
        np.save(f, signal)
        np.save(f, label)

    signal_df = pd.DataFrame(
        signal.tolist(), columns=[f"ch:{i}" for i in range(1, 65)]
    )
    print(signal_df)
    plot_autocorr(args, column="ch:1", df=signal_df)
    plot_autocorr(args, column="ch:22", df=signal_df)
    plot_autocorr(args, column="ch:64", df=signal_df)

    # decompose_signal(args, signal_channel="ch:1", df=signal_df)
