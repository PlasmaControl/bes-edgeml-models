import pickle
from multiprocessing import Pool, cpu_count, Array
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from fastdtw import fastdtw
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform, euclidean
from scipy.stats import skew, kurtosis
from sklearn.metrics import f1_score
from tqdm import tqdm

from elm_prediction.analyze import Analysis
from elm_prediction.options.test_arguments import TestArguments
from elm_prediction.src import utils


# %%

def get_elm_predictions(run_dir: str | Path):
    """Returns elm_predictions from elm_prediction.analyze.calc_inference"""

    run = Analysis(run_dir)
    run.test_data_file = Path(run_dir).parent / 'datasets/test_data_large.pkl'
    elm_predictions = run._calc_inference_full()

    return elm_predictions


# %%
def id_elms(elm_predictions, scale: str = None):
    """
    Function to return identifying features of ELM events.
    :param elm_predictions: dict of ELM predictions from analyze.calc_inference
    :param scale: ['norm'|'standard']
    :return: list[std(pre-ELM), max(ELM), min(ELM), len(active-ELM), gradient(first ELM >= 5V)]
    """
    ids = []
    for i, elm in enumerate(elm_predictions.values()):
        cs = elm['signals'][:, 2, :]
        p_elm = cs[elm['labels'] == 0]
        a_elm = cs[elm['labels'] == 1]
        precursor = p_elm[-300:]
        p_elm = p_elm[:-300]
        first_5 = a_elm[(a_elm >= 5).any(axis=1).argmax()]  # cross-section of BES array first time any is > 5V

        id = [
            np.sqrt(np.mean(p_elm ** 2, axis=0)),
            skew(p_elm),
            kurtosis(p_elm),
            np.std(precursor, axis=0),
            np.min(cs, axis=0),
            np.max(cs, axis=0),
            np.full((8,), len(a_elm)),
            np.gradient(first_5),
        ]

        ids.append(id)
    ids = np.array(ids)

    if scale == 'norm':
        # normalize between 0 and 1
        for i in range(ids.shape[1]):
            if ids[:, i, :].max() != ids[:, i, :].min():
                ids[:, i, :] = (ids[:, i, :] - ids[:, i, :].min()) / (ids[:, i, :].max() - ids[:, i, :].min())
            else:
                ids[:, i, :] = 1
    elif scale == 'standard':
        for i in range(ids.shape[1]):
            ids[:, i, :] = (ids[:, i, :] - np.mean(ids[:, i, :])) / np.std(ids[:, i, :])

    return ids


# %%
def do_calc(tup):
    """Helper function for dtw multiprocessing"""
    i, curr_id = tup
    for j, series2 in enumerate(ids_lst[i+1:]):
        j += i + 1
        dist, path = fastdtw(curr_id, series2, dist=euclidean)
        dist_matrix[i*len(ids_lst) + j] = dist
        # print('asdf')
        # print(f"{multiprocessing.current_process().name}\n{dist_matrix[:]}")


def init_arrs(dist_arr_, ids_lst_):
    """more helper functions!"""
    global dist_matrix, ids_lst
    dist_matrix = dist_arr_
    ids_lst = ids_lst_


def link_dtw(ids, distance_sort: bool = False):

    ids_lst = [i.tolist() for i in ids]
    dist_matrix = Array('f', len(ids_lst)*len(ids_lst))

    with Pool(processes=cpu_count()-10, initializer=init_arrs, initargs=(dist_matrix, ids_lst)) as pool:
        for _ in tqdm(pool.imap_unordered(do_calc, list(zip(range(len(ids_lst)), ids_lst))), desc='Calculating distances', total=len(ids_lst)):
            continue

    dist_matrix = np.frombuffer(dist_matrix.get_obj(), dtype='float32').reshape(len(ids_lst), len(ids_lst))
    dist_matrix = dist_matrix + dist_matrix.T
    square = squareform(dist_matrix)
    linkage_matrix = linkage(square, method='ward', optimal_ordering=distance_sort)

    return dist_matrix, linkage_matrix


# %%
def plot_groups(elm_predictions,
                ids: np.ndarray = None,
                thresh: float = None,
                method: str = None,
                distance_sort: bool = False,
                ax=None,
                save_fig: bool = False):
    """
    Function to plot dendrogram from elm_prediction nested dict
    :param elm_predictions: nested dict of ELMs from analyze.calc_inference
    :param thresh: (optional) threshold value for dendrogram and clustering algorithm
    :param distance_sort: (optional) sort plotted dendrogram by smallest distance first.
    :return: tuple(np.ndarray) ELM indexes (from analyze.calc_inference) grouped by distance below threshold.
    """

    # make linkage_matrix
    if method == 'agglomerative' or method is None:
        # quick check to make sure everything's there
        if ids is None:
            raise AttributeError('Agglomerative clustering must be passed with "ids"')
        ids = ids.reshape(ids.shape[0], -1)
        linkage_matrix = linkage(ids, method='ward', optimal_ordering=distance_sort)

    elif method == 'dtw':
        fname = Path(__file__).parent / 'run_dir_classification/dtw_distance_matrix.pkl'
        try:
            with open(fname, 'rb') as f:
                dist_matrix, linkage_matrix = pickle.load(f)
        except (OSError, IOError):
            signals = np.array([elm['signals'][:, 2, :].mean(axis=1) for elm in elm_predictions.values()], dtype=object)
            dist_matrix, linkage_matrix = link_dtw(signals, distance_sort=distance_sort)
            with open(fname, 'w+b') as f:
                pickle.dump((dist_matrix, linkage_matrix), f)

    # set other rcParams here
    with plt.rc_context({'lines.linewidth': 2.5, 'xtick.labelsize': 'large'}):

        ### Make plot
        if ax:
            fig = plt.gcf()
            ax1 = ax
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(16, 9))
        ax2 = ax1.twinx()

        # get order of elms in dendrogram
        dend = dendrogram(linkage_matrix, color_threshold=thresh, ax=ax1, get_leaves=True, distance_sort=distance_sort)
        # elm_idxs is the index of the ELM in the array of all elms, not the index assigned in calc_inference
        elm_idxs = dend['leaves']
        tick_locations = ax1.get_xticks()
        bar_locations = list(list(zip(*sorted(zip(elm_idxs, tick_locations))))[-1])

        # get f1 score of each elm to plot
        f1 = []
        for elm in elm_predictions.values():
            f1.append(f1_score(elm['labels'], (elm['micro_predictions'] >= 0.4)))
        # add bar graph with f1 scores
        color = 'tab:blue'
        ax2.bar(bar_locations, f1, width=np.diff(tick_locations).min(), color=color, alpha=0.5)
        ax2.tick_params(axis='y', color=color, labelcolor=color)

        # allow context manager for dendrogram
        if thresh:
            ax1.axhline(thresh, color='tab:gray', ls='--', label='Threshold')

        # Configure plot
        ax1.grid(False)
        ax2.grid(False)

        ax1.set_ylabel('ELM Feature Distance')
        ax2.set_ylabel('Model F1 Score')

        ax1.set_xlabel("ELM Index")

        ax1.set_title("Hierarchical Clustering Dendrogram of ELM Feature Distance")

        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)
        ax1.set_xticklabels(np.array([*elm_predictions.keys()])[elm_idxs])

        # set color of labels
        for tick_color, tick in zip(dend['leaves_color_list'], ax1.get_xticklabels()):
            tick.set_color(tick_color)

        plt.tight_layout()
        if save_fig:
            fig.savefig(Path(__file__).parent / 'run_dir_classification/plots/dendrogram.png')
        fig.show()

    return


# %%
def cluster_groups(elm_predictions: dict, thresh: float, ids: np.ndarray = None, scale: str = None):
    if ids is None:
        ids = id_elms(elm_predictions, scale=scale)
    linkage_matrix = linkage(ids.reshape(ids.shape[0], -1), method='ward')
    clusters = fcluster(linkage_matrix, t=thresh, criterion='distance')
    clusters_unique = np.sort(np.unique(clusters))
    group_elms = tuple(np.array([*elm_predictions.keys()])[clusters == cluster_id] for cluster_id in clusters_unique)
    return group_elms


# %%
if __name__ == '__main__':
    # %%
    run_dir = '/home/jazimmerman/PycharmProjects/bes-edgeml-models/bes-edgeml-work/run_dir_classification'
    try:
        assert Path('asdf').exists()
        with open(Path(run_dir).parent / 'datasets/elm_predictions.pkl', 'r+b') as f:
            elm_predictions = pickle.load(f)
    except AssertionError:
        elm_predictions = get_elm_predictions(run_dir)
    # %%
    ids_norm = id_elms(elm_predictions, scale='norm')
    ids_standard = id_elms(elm_predictions, scale='standard')
    ids_raw = id_elms(elm_predictions, scale=None)

    # %% Show migration of groups between scaling techniques
    thresh = 52
    n_plots = 5
    for i in range(n_plots):
        if i != 4:
            continue
        fig, axs = plt.subplots(int(np.ceil(n_plots / 2)), 2)

        axs = axs.flatten()

        plot_groups(elm_predictions, ids=ids_standard, thresh=thresh if i == 0 else 0, ax=axs[0])
        axs[0].set_title(f'Hierarchical Clustering of Standardized ELM Feature Distance')

        ids_stand_nomax = np.delete(ids_standard, 5, axis=1)
        plot_groups(elm_predictions, ids=ids_stand_nomax, thresh=thresh if i == 1 else 0, ax=axs[1])
        axs[1].set_title(f'Hierarchical Clustering of ELM Feature Distance (no-Max)')

        ids_stand_nolen = np.delete(ids_standard, 6, axis=1)
        plot_groups(elm_predictions, ids=ids_stand_nolen, thresh=thresh if i == 2 else 0, ax=axs[2])
        axs[1].set_title(f'Hierarchical Clustering of ELM Feature Distance (no-duration)')

        ids_stand_nograd = np.delete(ids_standard, 7, axis=1)
        plot_groups(elm_predictions, ids=ids_stand_nograd, thresh=thresh if i == 3 else 0, ax=axs[3])
        axs[3].set_title(f'Hierarchical Clustering of ELM Feature Distance (no-gradient)')

        plot_groups(elm_predictions, thresh=8000 if i == 4 else 0, method='dtw', ax=axs[4])
        axs[4].set_title('Hierarchical Clustering Dendrogram of DTW ELM Feature Distance')

        # Set label colors
        ax_labels = [(label.get_text(), label.get_color()) for label in axs[i].get_xticklabels()]
        for ax in np.delete(axs.flat, i):
            if not ax.title.get_text():
                continue
            labels = [label.get_text() for label in ax.get_xticklabels()]
            for text, color in ax_labels:
                idx = labels.index(text)
                ax.get_xticklabels()[idx].set_color(color)

        axs[i].set_title(axs[i].title.get_text() + '(Baseline)')

        plt.tight_layout()
        plt.show()

    # %%
    orange_group = cluster_groups(elm_predictions, ids=ids_standard, thresh=thresh)[0]
    signals = []
    for elm_id in orange_group:
        elm = elm_predictions[elm_id]
        signal = elm['signals'][:, 2, :].mean(axis=1)
        f1 = f1_score(elm['labels'], (elm['micro_predictions'] >= 0.4))
        signals.append((elm_id, signal, f1))

    n_plots = 6
    n_pages = int(np.ceil(len(signals) / n_plots))

    for page_num in range(n_pages):
        fig, axs = plt.subplots(n_plots, 1)
        for plot_num in range(n_plots):
            sig_num = page_num * n_plots + plot_num
            try:
                elm_id, sig, f1 = signals[sig_num]
            except IndexError:
                continue
            axs[plot_num].plot(sig)
            axs[plot_num].set_title(f'Mean Signal Ch17-Ch24 ELM {elm_id}')
            axs[plot_num].text(0.05, 0.95,
                               f'F1 Score {f1}',
                               transform=axs[plot_num].transAxes, fontsize=14,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

    # %%
    # ids = id_elms(elm_predictions, scale='standard')
    # fig, axs = plt.subplots(1, ids.shape[1])
    # ch_22 = ids[..., -1]
    # titles = ['RMS of Pre-ELM',
    #           'Skew of Pre-ELM',
    #           'Kurtosis of Pre-ELM',
    #           'Std. Dev. of Precursor',
    #           'Max',
    #           'Min',
    #           'Length of ELM',
    #           'Gradient']
    # for i, ax in enumerate(axs.flat):
    #     ax.hist(ch_22[:, i], bins=25)
    #     ax.set_title(titles[i])
    # fig.suptitle('Standardized ELM Feature Distribution (ch 24)')
    # plt.show()
