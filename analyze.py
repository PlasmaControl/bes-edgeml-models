import os
import pickle
from typing import Tuple, List, Union
import argparse
import logging

# import matplotlib

# matplotlib.use("TkAgg")
import cv2
import matplotlib
import torch
import numpy as np
from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import plotly.express as px
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition as comp
from torch.functional import norm
from tqdm import tqdm

from data_preprocessing import *
from src import data, utils, dataset
from options.test_arguments import TestArguments
from matplotlib.backends.backend_pdf import PdfPages

# plt.style.use("/home/lakshya/plt_custom.mplstyle")
# plt.style.use("/home/lm9679/plt_custom.mplstyle")
colors = sns.color_palette("deep").as_hex()
sns.set_theme()


def get_test_dataset(
        args: argparse.Namespace,
        file_name: str,
        logger: logging.getLogger = None,
        transforms=None
) -> Tuple[tuple, data.ELMDataset]:
    """Read the pickle file containing the test data and return PyTorch dataset
    and data attributes such as signals, labels, sample_indices, and
    window_start_indices.

    Args:
    -----
        args (argparse.Namespace): Argparse namespace object containing all the
            base and test arguments.
        file_name (str): Name of the test data file.
        logger (logging.getLogger): Logger object that adds inference logs to
            a file. Defaults to None.
        transforms: Image transforms to perform data augmentation on the given
            input. Defaults to None.
    """
    with open(file_name, "rb") as f:
        test_data = pickle.load(f)

    signals = np.array(test_data["signals"])
    labels = np.array(test_data["labels"])
    sample_indices = np.array(test_data["sample_indices"])
    window_start = np.array(test_data["window_start"])
    data_attrs = (signals, labels, sample_indices, window_start)
    test_dataset = dataset.ELMDataset(
        args, *data_attrs, logger=logger, transform=transforms, phase="testing"
    )

    return data_attrs, test_dataset


def predict(
        args: argparse.Namespace,
        test_data: tuple,
        model: object,
        device: torch.device,
) -> dict:
    signals = test_data[0]
    print(f"Signals shape: {signals.shape}")
    labels = test_data[1]
    _ = test_data[2]  # sample_indices
    window_start = test_data[3]
    num_elms = len(window_start)
    elm_predictions = dict()
    for i_elm in range(num_elms):
        print(f"Processing elm event with start index: {window_start[i_elm]}")
        i_start = window_start[i_elm]
        if i_elm < num_elms - 1:
            i_stop = window_start[i_elm + 1] - 1
        else:
            i_stop = labels.size
        # gathering the indices for active elm events
        elm_signals = signals[i_start:i_stop, ...]
        elm_labels = labels[i_start:i_stop]
        active_elm = np.where(elm_labels > 0.0)[0]
        active_elm_start = active_elm[0]
        active_elm_end = active_elm[-1]
        # add a buffer of a given number of time frames
        # in both start and end
        active_elm_start_extended = (
            (active_elm_start - args.buffer_frames)
            if (active_elm_start - args.buffer_frames) > 0
            else 0
        )
        active_elm_end_extended = (
            (active_elm_end + args.buffer_frames)
            if (active_elm_end + args.buffer_frames) < len(elm_labels) - 1
            else len(elm_labels) - 1
        )
        predictions = np.zeros(
            elm_labels.size
            - args.signal_window_size
            - args.label_look_ahead
            + 1
        )
        for j in range(predictions.size):
            if args.data_preproc == "interpolate":
                signals_resized = []
                input_signals = np.array(
                    elm_signals[j: j + args.signal_window_size, :, :],
                    dtype=np.float32,
                )
                for signal in input_signals:
                    signal = cv2.resize(
                        signal,
                        dsize=(args.interpolate_size, args.interpolate_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    signals_resized.append(signal)
                signals_resized = np.array(signals_resized)
                input_signals = torch.as_tensor(
                    signals_resized.reshape(
                        [1, 1, args.signal_window_size, 8, 8]
                    ),
                    dtype=torch.float32,
                )
            else:
                input_signals = torch.as_tensor(
                    elm_signals[j: j + args.signal_window_size, :, :].reshape(
                        [1, 1, args.signal_window_size, 8, 8]
                    ),
                    dtype=torch.float32,
                )
            input_signals = input_signals.to(device)
            predictions[j] = model(input_signals)
        elm_signals = elm_signals[
                      : (-args.signal_window_size - args.label_look_ahead + 1), ...
                      ]
        elm_labels = elm_labels[
                     : (-args.signal_window_size - args.label_look_ahead + 1)
                     ]
        # convert logits to probability
        # calculate micro predictions for each time step
        micro_predictions = (
            torch.sigmoid(torch.as_tensor(predictions, dtype=torch.float32))
                .cpu()
                .numpy()
        )
        # filter signals, labels and micro-predictions for active elm regions
        # elm_signals_active_elms = elm_signals[
        #     active_elm_start_extended:active_elm_end_extended
        # ]
        elm_labels_active_elms = elm_labels[
                                 active_elm_start_extended:active_elm_end_extended
                                 ]
        micro_predictions_active_elms = micro_predictions[
                                        active_elm_start_extended:active_elm_end_extended
                                        ]

        # filter signals, labels and micro-predictions for non-active elm regions
        # elm_signals_pre_active_elms = elm_signals[:active_elm_start_extended]
        # elm_signals_post_active_elms = elm_signals[active_elm_end_extended:]
        elm_labels_pre_active_elms = elm_labels[:active_elm_start_extended]
        elm_labels_post_active_elms = elm_labels[active_elm_end_extended:]
        micro_predictions_pre_active_elms = micro_predictions[
                                            :active_elm_start_extended
                                            ]
        micro_predictions_post_active_elms = micro_predictions[
                                             active_elm_end_extended:
                                             ]
        # calculate macro predictions for each elm event
        active_elm_true_frames_count = active_elm_end - active_elm_start + 1
        active_elm_prediction_count = np.sum(
            micro_predictions_active_elms > 0.4
        )
        macro_predictions_active_elms = np.array(
            [
                (
                        active_elm_prediction_count > active_elm_true_frames_count
                ).astype(int)
            ]
        )
        macro_predictions_pre_active_elms = np.array(
            [(np.sum(micro_predictions_pre_active_elms > 0.4) > 0).astype(int)]
        )
        macro_predictions_post_active_elms = np.array(
            [(np.sum(micro_predictions_post_active_elms > 0.4) > 0).astype(int)]
        )
        macro_labels = np.array(
            [
                np.mean(elm_labels_pre_active_elms).astype(int),
                int(
                    np.sum(elm_labels_active_elms)
                    / (active_elm_end - active_elm_start + 1)
                ),
                np.mean(elm_labels_post_active_elms).astype(int),
            ]
        )
        macro_predictions = np.concatenate(
            [
                macro_predictions_pre_active_elms,
                macro_predictions_active_elms,
                macro_predictions_post_active_elms,
            ]
        )
        # elm_time = np.arange(
        #     active_elm_start_extended, active_elm_end_extended
        # )  [: (-args.signal_window_size - args.label_look_ahead + 1)]
        elm_time = np.arange(elm_labels.size) + window_start[i_elm]
        elm_predictions[window_start[i_elm]] = {
            "signals": elm_signals,
            "labels": elm_labels,
            "micro_predictions": micro_predictions,
            "macro_labels": macro_labels,
            "macro_predictions": macro_predictions,
            "elm_time": elm_time,
        }
    return elm_predictions


def predict_v2(
        args: argparse.Namespace,
        test_data: tuple,
        model: object,
        hook_layer: str,
        device: torch.device,
) -> dict:
    signals = test_data[0]
    print(f"Signals shape: {signals.shape}")
    labels = test_data[1]
    _ = test_data[2]  # sample_indices
    window_start = test_data[3]
    num_elms = len(window_start)
    elm_predictions = dict()

    ### ------------ Forward Hook ---------- ###
    activations = []

    def get_activation():
        def hook(model, input, output):
            o = output.detach()[0].numpy()
            activations.append(o)

        return hook

    (act_layer, weight_layer) = get_layer(model, hook_layer)

    act_layer.register_forward_hook(get_activation())
    weights = weight_layer.weight.detach().numpy()[0]
    ### ------------ /Forward Hook ---------- ###

    for i_elm in range(num_elms):
        # ^^^ FOR JEFF'S TESTING PURPOSES ^^^ #
        # for i_elm in range(1):
        print(f"Processing elm event with start index: {window_start[i_elm]}")
        i_start = window_start[i_elm]
        if i_elm < num_elms - 1:
            i_stop = window_start[i_elm + 1] - 1
        else:
            i_stop = labels.size
        # gathering the indices for active elm events
        elm_signals = signals[i_start:i_stop, ...]
        elm_labels = labels[i_start:i_stop]
        active_elm = np.where(elm_labels > 0.0)[0]
        active_elm_start = active_elm[0]
        active_elm_lower_buffer = active_elm_start - args.truncate_buffer
        active_elm_upper_buffer = active_elm_start + args.truncate_buffer
        predictions = np.zeros(
            elm_labels.size
            - args.signal_window_size
            - args.label_look_ahead
            + 1
        )
        activations = []
        for j in range(predictions.size):
            if args.use_gradients:
                input_signals = np.array(
                    elm_signals[j: j + args.signal_window_size, :, :].reshape(
                        [1, args.signal_window_size, 8, 8, 6]
                    ),
                    dtype=np.float32,
                )
                input_signals = np.transpose(
                    input_signals, axes=(0, 4, 1, 2, 3)
                )
            else:
                input_signals = np.array(
                    elm_signals[j: j + args.signal_window_size, :, :].reshape(
                        [1, 1, args.signal_window_size, 8, 8]
                    ),
                    dtype=np.float32,
                )
            input_signals = torch.as_tensor(input_signals, dtype=torch.float32)
            input_signals = input_signals.to(device)
            predictions[j] = model(input_signals)

        activations = np.array(activations)

        elm_signals = elm_signals[
                      : (-args.signal_window_size - args.label_look_ahead + 1), ...
                      ]
        elm_labels = elm_labels[
                     : (-args.signal_window_size - args.label_look_ahead + 1)
                     ]
        # convert logits to probability
        # calculate micro predictions for each time step
        micro_predictions = (
            torch.sigmoid(torch.as_tensor(predictions, dtype=torch.float32))
                .cpu()
                .numpy()
        )
        # filter labels and micro-predictions for active elm regions
        elm_labels_active_elms = elm_labels[
                                 active_elm_lower_buffer:active_elm_upper_buffer
                                 ]
        micro_predictions_active_elms = micro_predictions[
                                        active_elm_lower_buffer:active_elm_upper_buffer
                                        ]

        # filter labels and micro-predictions for non-active elm regions
        elm_labels_pre_active_elms = elm_labels[:active_elm_lower_buffer]
        micro_predictions_pre_active_elms = micro_predictions[
                                            :active_elm_lower_buffer
                                            ]

        # calculate macro predictions for each region
        macro_predictions_active_elms = np.array(
            [np.any(micro_predictions_active_elms).astype(int)]
        )

        macro_predictions_pre_active_elms = np.array(
            [np.any(micro_predictions_pre_active_elms > 0.4).astype(int)]
        )

        macro_labels = np.array([0, 1], dtype="int")
        macro_predictions = np.concatenate(
            [macro_predictions_pre_active_elms, macro_predictions_active_elms]
        )
        elm_time = np.arange(elm_labels.size) + window_start[i_elm]
        elm_predictions[window_start[i_elm]] = {
            "activations": activations,
            "weights": weights,
            "signals": elm_signals,
            "labels": elm_labels,
            "micro_predictions": micro_predictions,
            "macro_labels": macro_labels,
            "macro_predictions": macro_predictions,
            "elm_time": elm_time,
        }
    return elm_predictions


def make_arrays(dic: dict, key: str):
    '''
    Helper function to return value from predict_v2 elm_prediction dict key
    as numpy array along with start and stop indices of ELM.
    ----------------------------------------------------
    Returns dict:   {
                    'key': np.array,        (dimensions: N-nodes x len_ELM)
                    'elm_start': np.array,  (dimensions: 1 x len_ELM)
                    'elm_end': np.array     (dimensions: 1 x len_ELM)
                    }
    Jeff Zimmerman
    '''

    ends = np.cumsum([dic[elm_idx]['elm_time'].shape[0] for elm_idx in list(dic.keys())])
    starts = np.pad(ends, (1, 0), mode='constant')[:-1]

    arr = np.empty((ends[-1], *list(dic.values())[0][key].squeeze().shape[1:]))
    label_arr = np.empty(ends[-1])

    for i_start, i_end, elm_dic in zip(starts, ends, list(dic.values())):
        arr[i_start:i_end] = elm_dic[key].squeeze()
        label_arr[i_start:i_end] = elm_dic['labels'].squeeze()

    return {key: arr, 'labels': label_arr, 'elm_start_idx': starts, 'elm_end_idx': ends}


def plot_boxes(elm_predictions: dict,
               layer: str = None):
    elm_ids = list(elm_predictions.keys())
    elm_id = elm_ids[0]
    activations = elm_predictions[elm_id]['activations'].squeeze().T

    ncols = 4
    nrows = ceil(activations.shape[0] / (4 * ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    for i, ax in enumerate(axs.flat):
        vals = activations[4 * i: 4 * i + 4]
        labels = np.chararray(vals.shape, itemsize=7, unicode=True)
        for j in range(4):
            labels[j, :] = f'Node {4 * i + j}'
        activations_df = pd.DataFrame(data=list(zip(labels.flat, vals.flat)),
                                      columns=['Node', 'Activation']
                                      )
        sns.violinplot(x='Node', y='Activation', data=activations_df, ax=ax)

    fig.suptitle(f'Violin Plots of Node Activations in Layer {layer}')
    # plt.tight_layout()
    plt.show()


def perform_PCA(elm_predictions: dict,
                layer=None,
                plot: bool = True):
    '''
    Use scikit learn's pca analysis tools to reduce dimensionality of hidden layer
    output.
    Jeff Zimmerman
    '''

    elm_arrays = make_arrays(elm_predictions, 'activations')
    activations = elm_arrays['activations']
    start_end = zip(elm_arrays['elm_start_idx'], elm_arrays['elm_end_idx'])
    elm_labels = elm_arrays['labels']
    # weights = make_array(elm_predictions, 'weights')
    t_dim = np.arange(activations.shape[0])

    # weighted = []
    # for i, act in enumerate(activations):
    #     weighted.append(weights[i] * act)

    standard = StandardScaler().fit_transform(activations)
    # standard_t = np.append([t_dim], standard.T, axis=0).T

    n_components = 5
    pca = comp.PCA(n_components=n_components)
    pca.fit(standard)
    decomposed = pca.transform(standard)
    decomposed_t = np.append([t_dim], decomposed.T, axis=0).T
    print(f'Explained variance ratio of first {n_components} components')
    for i, x in enumerate(pca.explained_variance_ratio_[:n_components]):
        print(f'\tComponent {i}: {x * 100:0.3f}')
    for component in range(3):
        print(f'\nComponent {component}:\n{pca.components_[component]}')

    if plot:
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # for start, end in list(zip(elm_start, elm_end))[:1]:
        #     labels = elm_labels[start:end]
        #     fig = px.scatter_3d(x=decomposed_t[:, 0][start:end],
        #                         y=decomposed_t[:, 1][start:end],
        #                         z=decomposed_t[:, 2][start:end],
        #                         color=labels)
        #
        # # px.set_xlabel('Time $(\mu s)$')
        # # px.set_ylabel('PC 1')
        # # px.set_zlabel('PC 2')
        # fig.show()

        fig, axs = plt.subplots(nrows=n_components, ncols=n_components, figsize=(16, 9))
        for ax in axs.flat:
            ax.set_axis_off()
        for pc_col in range(0, n_components):
            for pc_row in range(0, pc_col + 1):
                ax = axs[pc_row][pc_col]
                ax.set_axis_on()
                pc_x = decomposed_t[:, pc_col - pc_row]
                pc_y = decomposed_t[:, pc_col + 1]
                for start, end in start_end[:1]:
                    ax.scatter(x=pc_x[start:end],
                               y=pc_y[start:end],
                               edgecolor='none',
                               c=matplotlib.cm.gist_rainbow(np.linspace(0, 1, end - start)),
                               s=4)
                ax.set_ylabel(f'PC_{pc_col + 1}')
                ax.set_xlabel('Time' if pc_row == pc_col else f'PC_{pc_row + 1}')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f'PCA of Layer {layer} in feature model')
        plt.show()

    return decomposed_t


def correlate_pca(pca: np.ndarray, elm_predictions: dict, layer: str, type: str):
    """
    Plot correlations of input channels with PCA.
    Specify type='box' for box plot of correlations along radial dimension
    Specify type='hist' for histogram of correlation all 64 channels
    """
    ixslice = pd.IndexSlice
    pca_t = pca.T[1:]
    elm_list = list(elm_predictions.keys())
    arrs = make_arrays(elm_predictions, 'signals')
    signals = arrs['signals']
    start_end = zip(arrs['elm_start_idx'], arrs['elm_end_idx'])
    reshaped = np.empty((64, signals.shape[0]))
    df_index_a = np.empty(signals.shape[0])
    df_index_b = np.empty(signals.shape[0])
    for i, (start_idx, end_idx) in enumerate(start_end):
        df_index_a[start_idx:end_idx] = elm_list[i]
        df_index_b[start_idx:end_idx] = np.arange(end_idx - start_idx)

    df_index = pd.MultiIndex.from_arrays([df_index_a, df_index_b], names=['ELM_ID', 'Time'])

    for channel_y in range(8):
        for channel_x in range(8):
            reshaped[8 * channel_x + channel_y] = signals[:, channel_y, channel_x]

    data_arr = np.append(reshaped, pca_t, axis=0).T
    data_labels = [f'Channel_{x + 1}' for x in range(64)]
    pc_labels = [f'PC_{x + 1}' for x in range(len(pca_t))]
    labels_all = np.append(data_labels, pc_labels)
    df_all = pd.DataFrame(data_arr, columns=labels_all, index=df_index)
    df_all = df_all.groupby(level='ELM_ID').corr().loc[ixslice[:, 'PC_1':'PC_5'], 'Channel_1':'Channel_64']

    if type.lower() == 'hist':
        pc1_corr_df = df_all.xs('PC_1', level=1)
        pc2_corr_df = df_all.xs('PC_2', level=1)
        for x in range(4):
            fig, axs = plt.subplots(4, 4)
            for i, ax in enumerate(axs.flat, start=1):
                i = x * 16 + i
                sns.histplot(data=pc1_corr_df, x=f'Channel_{i}', ax=ax, color=sns.color_palette()[1])
                sns.kdeplot(data=pc1_corr_df, x=f'Channel_{i}', ax=ax, color=sns.color_palette()[1])
                ax.set_title(f'Channel {i}', loc='left', y=0.8)
            fig.suptitle('Distribution of Correlations Between PC1 and Input Channels')
            plt.show()

    if type.lower() == 'box':
        df_box = df_all.loc[:, 'Channel_17':'Channel_24'].reset_index(level=1)
        # must make df categorical to work on seaborn box plot.
        df_box_index = df_box.index
        all_vals = []
        pc_vals = []
        channel_vals = []
        for col in df_box.loc[:, 'Channel_17':'Channel_24']:
            this_column_values = df_box[col].tolist()
            this_column_pc = df_box['level_1'].tolist()
            all_vals += this_column_values
            pc_vals += this_column_pc
            channel_vals += [col] * len(this_column_values)
        df_box = pd.DataFrame(data=np.transpose([pc_vals, channel_vals, all_vals]),
                              columns=['Component', 'Channel', 'Correlation'])
        df_box['Correlation'] = df_box['Correlation'].astype(float)
        fig, ax = plt.subplots(1, 1)
        sns.boxplot(data=df_box, x='Channel', y='Correlation', hue='Component', ax=ax)
        fig.suptitle(f'Distribution of Correlation Along Radial Axis')
        ax.set_title(f'Layer: {layer}')
        plt.show()


def plot(
        args: argparse.Namespace,
        elm_predictions: dict,
        plot_dir: str,
        elms: List[int],
        elm_range: str,
        n_rows: Union[int, None] = None,
        n_cols: Union[int, None] = None,
        figsize: tuple = (14, 12),
) -> None:
    state = np.random.RandomState(seed=args.seed)
    elm_id = list(elm_predictions.keys())
    i_elms = state.choice(elm_id, args.plot_num, replace=False)
    if args.save_pdf:
        pp = PdfPages(args.save_pdf)
    for elm_no, i_elm in enumerate(i_elms[:48]):
        fig, axs = plt.subplots(10, 4, figsize=(8, 8))
        plt.subplots_adjust(hspace=0)
        print(f"ELM {elm_no} of {len(i_elms)} with {len(elm_predictions[i_elm]['elm_time'])} time points")
        activations = elm_predictions[i_elm]["activations"]
        weights = elm_predictions[i_elm]["weights"]
        for i in range(4):
            signals = elm_predictions[i_elm]["signals"]
            labels = elm_predictions[i_elm]["labels"]
            elm_start = np.where(labels > 0)[0][0]
            predictions = elm_predictions[i_elm]["micro_predictions"]
            elm_time = elm_predictions[i_elm]["elm_time"]
            axs[9][i].plot(elm_time,
                           signals[:, 2, 6],
                           label="BES ch. 22" if i == 0 else '_nolegend_',
                           c=colors[0])
            axs[8][i].plot(
                elm_time - args.label_look_ahead,
                predictions,
                label="Prediction" if i == 0 else '_nolegend_',
                ls="-.",
                lw=1.25,
                c=colors[-2],
            )
            axs[9][i].set_xlabel("Time (micro-s)")
            for j in [8, 9]:
                axs[j][i].set_ylim([-1, 1])
                axs[j][i].grid(axis="y")
                axs[j][i].plot(
                    elm_time,
                    labels + 0.02,
                    label="Ground truth" if i + j == 8 else '_nolegend_',
                    ls="-.",
                    lw=1.25,
                    c=colors[1],
                )
                axs[j][i].axvline(
                    elm_start - 75,
                    ymin=0,
                    ymax=0.9,
                    c=colors[-1],
                    ls=":",
                    lw=1.5,
                )
                axs[j][i].axvline(
                    elm_start + 75,
                    ymin=0,
                    ymax=0.9,
                    c=colors[-1],
                    ls=":",
                    lw=1.5,
                )
                axs[j][i].plot(
                    elm_time,
                    labels + 0.02,
                    ls="-.",
                    lw=1.25,
                    c=colors[1],
                )
                axs[j][i].text(.5, .9, "Prediction" if j == 8 else "Signal",
                               horizontalalignment='center',
                               verticalalignment='top',
                               transform=axs[j][i].transAxes)
            for j in range(8):
                weighted = activations[i * 8 + j] * weights[0][i * 8 + j]
                axs[j][i].plot(weighted, label="Node output (weighted)" if i + j == 0 else '_nolegend_', )
                axs[j][i].set_xticks([])
                axs[j][i].text(.5, .9, f'Node {i * 8 + j}: weight {weights[0][i * 8 + j]:.3f}',
                               horizontalalignment='center',
                               verticalalignment='top',
                               fontsize=8,
                               transform=axs[j][i].transAxes)
                axs[j][i].set_ylim(-1, 1)
                axs[j][i].plot(
                    elm_time,
                    labels + 0.02,
                    ls="-.",
                    lw=1.25,
                    c=colors[1],
                )
                axs[j][i].axvline(
                    elm_start - 75,
                    ymin=0,
                    ymax=0.9,
                    c=colors[-1],
                    ls=":",
                    lw=1.5,
                    label="Buffer limits" if i + j == 0 else '_nolegend_',
                )
                axs[j][i].axvline(
                    elm_start + 75,
                    ymin=0,
                    ymax=0.9,
                    c=colors[-1],
                    ls=":",
                    lw=1.5,
                )
                axs[j][i].grid(axis="y")
                axs[j][i].axvline(
                    elm_start - 75,
                    ymin=0,
                    ymax=0.9,
                    c=colors[-1],
                    ls=":",
                    lw=1.5,
                )
                axs[j][i].axvline(
                    elm_start + 75,
                    ymin=0,
                    ymax=0.9,
                    c=colors[-1],
                    ls=":",
                    lw=1.5,
                )

        fig.legend(fontsize=8, frameon=False)
        plt.suptitle(f"Model output on elm {i_elm}", fontsize=20)
        if args.save_pdf:
            pp.savefig(fig)
        else:
            plt.show()
    if args.save_pdf:
        pp.close()
    return
    plt.suptitle(f"Model output on {args.data_mode} classes", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not args.dry_run:
        fig.savefig(
            os.path.join(
                plot_dir,
                f"{args.model_name}_{args.data_mode}_lookahead_{args.label_look_ahead}_time_series{args.filename_suffix}_{elm_range}.png",
            ),
            dpi=100,
        )
    plt.show()


def plot_all(
        args: argparse.Namespace,
        elm_predictions: dict,
        plot_dir: str,
) -> None:
    state = np.random.RandomState(seed=args.seed)
    elm_id = list(elm_predictions.keys())
    # i_elms = state.choice(elm_id, args.plot_num, replace=False)
    i_elms_1_12 = elm_id[:12]
    i_elms_12_24 = elm_id[12:24]
    i_elms_24_36 = elm_id[24:36]
    i_elms_36_42 = elm_id[36:]

    # plot 1-12
    plot(args, elm_predictions, plot_dir, i_elms_1_12, elm_range="1-12")
    # plot 12-24
    plot(args, elm_predictions, plot_dir, i_elms_12_24, elm_range="12-24")
    # plot 24-36
    plot(args, elm_predictions, plot_dir, i_elms_24_36, elm_range="24-36")
    # plot 36-42
    plot(
        args,
        elm_predictions,
        plot_dir,
        i_elms_36_42,
        elm_range="36-42",
        n_rows=2,
        n_cols=3,
        figsize=(14, 6),
    )


def show_details(test_data: tuple) -> None:
    print("Test data information")
    signals = test_data[0]
    labels = test_data[1]
    sample_indices = test_data[2]
    window_start = test_data[3]
    print(f"Signals shape: {signals.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample indices shape: {sample_indices.shape}")
    print(f"Window start indices: {window_start}")


def show_metrics(
        args: argparse.Namespace,
        y_true: np.ndarray,
        y_probas: np.ndarray,
        report_dir: str,
        roc_dir: str,
        plot_dir: str,
        pred_mode: str,
):
    if pred_mode == "micro":
        if np.array_equal(y_probas, y_probas.astype(bool)):
            raise ValueError(
                "Metrics for micro mode require micro_predictions but macro_predictions are passed."
            )
        y_preds = (y_probas > args.threshold).astype(int)

        # creating a classification report
        cm = metrics.confusion_matrix(y_true, y_preds)

        # calculate the log of the confusion matrix scaled by the
        # total error (false positives + false negatives)
        # cm_log = np.log(cm)
        x, y = np.where(~np.eye(cm.shape[0], dtype=bool))
        coords = tuple(zip(x, y))
        total_error = np.sum(cm[coords])
        cm = cm / total_error

        cr = metrics.classification_report(y_true, y_preds, output_dict=True)
        df = pd.DataFrame(cr).transpose()
        print(f"Classification report:\n{df}")

        # ROC details
        fpr, tpr, thresh = metrics.roc_curve(y_true, y_probas)
        roc_details = pd.DataFrame()
        roc_details["fpr"] = fpr
        roc_details["tpr"] = tpr
        roc_details["threshold"] = thresh

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot()
        sns.heatmap(
            cm,
            annot=True,
            ax=ax,
            annot_kws={"size": 14},
            fmt=".3f",
            norm=LogNorm(),
        )
        plt.setp(ax.get_yticklabels(), rotation=0)
        ax.set_xlabel("Predicted Label", fontsize=14)
        ax.set_ylabel("True Label", fontsize=14)
        ax.text(
            x=0.5,
            y=1.05,
            s="Micro predictions",
            fontsize=18,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )
        ax.text(
            x=0.5,
            y=1.01,
            s=f"Signal window: {args.signal_window_size}, Label look ahead: {args.label_look_ahead}",
            fontsize=12,
            alpha=0.75,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )
        if not args.dry_run:
            df.to_csv(
                os.path.join(
                    report_dir,
                    f"{args.model_name}_classification_report_micro_{args.data_mode}_lookahead_{args.label_look_ahead}{args.filename_suffix}.csv",
                ),
                index=True,
            )
            roc_details.to_csv(
                os.path.join(
                    roc_dir,
                    f"{args.model_name}_roc_details_micro_{args.data_mode}_lookahead_{args.label_look_ahead}{args.filename_suffix}.csv",
                ),
                index=False,
            )
            fig.savefig(
                os.path.join(
                    plot_dir,
                    f"{args.model_name}_confusion_matrix_micro_{args.data_mode}_lookahead_{args.label_look_ahead}{args.filename_suffix}.png",
                ),
                dpi=100,
            )
        plt.show()
    elif pred_mode == "macro":
        if not np.array_equal(y_probas, y_probas.astype(bool)):
            raise ValueError(
                "Metrics for macro mode require macro_predictions but micro_predictions are passed."
            )

        # creating a classification report
        cm = metrics.confusion_matrix(y_true, y_probas)
        cr = metrics.classification_report(y_true, y_probas, output_dict=True)
        df = pd.DataFrame(cr).transpose()
        print(f"Classification report:\n{df}")

        # ROC details
        fpr, tpr, thresh = metrics.roc_curve(y_true, y_probas)
        roc_details = pd.DataFrame()
        roc_details["fpr"] = fpr
        roc_details["tpr"] = tpr
        roc_details["threshold"] = thresh

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot()
        sns.heatmap(cm, annot=True, ax=ax, annot_kws={"size": 14}, fmt="d")
        plt.setp(ax.get_yticklabels(), rotation=0)
        ax.set_xlabel("Predicted Label", fontsize=14)
        ax.set_ylabel("True Label", fontsize=14)
        ax.text(
            x=0.5,
            y=1.05,
            s="Macro predictions",
            fontsize=18,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )
        ax.text(
            x=0.5,
            y=1.01,
            s=f"Signal window: {args.signal_window_size}, Label look ahead: {args.label_look_ahead}",
            fontsize=12,
            alpha=0.75,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )
        if not args.dry_run:
            df.to_csv(
                os.path.join(
                    report_dir,
                    f"{args.model_name}_classification_report_macro_{args.data_mode}_lookahead_{args.label_look_ahead}{args.filename_suffix}.csv",
                ),
                index=True,
            )
            roc_details.to_csv(
                os.path.join(
                    roc_dir,
                    f"{args.model_name}_roc_details_macro_{args.data_mode}_lookahead_{args.label_look_ahead}{args.filename_suffix}.csv",
                ),
                index=False,
            )
            fig.savefig(
                os.path.join(
                    plot_dir,
                    f"{args.model_name}_confusion_matrix_macro_{args.data_mode}_lookahead_{args.label_look_ahead}{args.filename_suffix}.png",
                ),
                dpi=100,
            )
        plt.show()
    else:
        raise ValueError(
            f"Expected pred_mode to be either `micro` or`macro` but {pred_mode} is passed."
        )


def model_predict(
        model,
        device: torch.device,
        data_loader: torch.utils.data.DataLoader,
) -> None:
    # put the elm_model to eval mode
    model.eval()
    predictions = []
    targets = []
    for images, labels in tqdm(data_loader):
        images = images.to(device)

        with torch.no_grad():
            preds = model(images)
        preds = preds.view(-1)
        predictions.append(torch.sigmoid(preds).cpu().numpy())
        targets.append(labels.cpu().numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    print(predictions[:10], targets[:10])
    print(
        f"ROC score on test data: {metrics.roc_auc_score(targets, predictions)}"
    )


def get_dict_values(pred_dict: dict, mode: str):
    targets = []
    predictions = []
    if mode == "micro":
        for vals in pred_dict.values():
            for k, v in vals.items():
                if k == "labels":
                    targets.append(v)
                if k == "micro_predictions":
                    predictions.append(v)
        return np.concatenate(targets), np.concatenate(predictions)
    elif mode == "macro":
        for vals in pred_dict.values():
            for k, v in vals.items():
                if k == "macro_labels":
                    targets.append(v)
                if k == "macro_predictions":
                    predictions.append(v)
        return np.concatenate(targets), np.concatenate(predictions)


def get_layer(model: object, layer: str):
    layer_dict = model.layers

    layer_idx = list(layer_dict.keys()).index(layer)
    weight_idx = layer_idx + 1

    act_layer = layer_dict[layer]
    weight_layer = list(layer_dict.items())[weight_idx][1]

    return (act_layer, weight_layer)


def main(
        args: argparse.Namespace,
) -> None:
    logger = utils.get_logger(
        script_name=__name__,
        stream_handler=True,
    )
    # instantiate the elm_model and load the checkpoint
    model_cls = utils.create_model(args.model_name)
    model = model_cls(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # load the model checkpoint and other paths
    output_paths = utils.create_output_paths(args, infer_mode=True)
    (
        test_data_dir,
        model_ckpt_dir,
        clf_report_dir,
        plot_dir,
        roc_dir,
    ) = output_paths
    model_ckpt_path = os.path.join(
        model_ckpt_dir,
        f"{args.model_name}_{args.data_mode}_lookahead_{args.label_look_ahead}{args.filename_suffix}.pth",
    )
    print(f"Using elm_model checkpoint: {model_ckpt_path}")
    model.load_state_dict(
        torch.load(
            model_ckpt_path,
            map_location=device,
        )["model"]
    )

    # get the test data and dataloader
    test_fname = os.path.join(
        test_data_dir,
        f"test_data_{args.data_mode}_lookahead_{args.label_look_ahead}{args.filename_suffix}.pkl",
    )

    print(f"Using test data file: {test_fname}")
    test_data, test_dataset = get_test_dataset(
        args, file_name=test_fname, logger=logger
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )
    inputs, _ = next(iter(test_loader))
    print(f"Input size: {inputs.shape}")

    if args.test_data_info:
        show_details(test_data)

    model_predict(model, device, test_loader)

    # get prediction dictionary containing truncated signals, labels,
    # micro-/macro-predictions and elm_time
    pca_layer = 'fc1'
    pred_dict = predict_v2(args=args, test_data=test_data, model=model, device=device, hook_layer=pca_layer)
    # plot_boxes(pred_dict, layer=pca_layer)
    pca = perform_PCA(pred_dict, layer=pca_layer, plot=False)
    correlate_pca(pca, pred_dict, layer=pca_layer, type='box')
    return

    if args.plot_data:
        plot_all(args, pred_dict, plot_dir)

    if args.show_metrics:
        # show metrics for micro predictions
        targets_micro, predictions_micro = get_dict_values(
            pred_dict, mode="micro"
        )
        show_metrics(
            args,
            targets_micro,
            predictions_micro,
            clf_report_dir,
            roc_dir,
            plot_dir,
            pred_mode="micro",
        )
        # show metrics for macro predictions
        targets_macro, predictions_macro = get_dict_values(
            pred_dict, mode="macro"
        )
        show_metrics(
            args,
            targets_macro,
            predictions_macro,
            clf_report_dir,
            roc_dir,
            plot_dir,
            pred_mode="macro",
        )


if __name__ == "__main__":
    args, parser = TestArguments().parse(verbose=True)
    utils.test_args_compat(args, parser, infer_mode=True)
    main(args)
