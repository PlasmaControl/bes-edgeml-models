import itertools
import logging

import numpy as np
import re
import h5py
import pandas as pd
from pathlib import Path
from typing import Union

from matplotlib import pyplot as plt


def make_labels(base_dir: Union[str, Path],
                logger: logging.Logger,
                df_name: str = 'confinement_database.xlsx',
                data_dir: Union[str, Path] = 'data',
                labeled_dir: Union[str, Path] = 'labeled_datasets'):
    """
    Function to create labeled datasets for turbulence regime classification.
    Shot data is sourced from base_dir / data.
    Resulting labeled datasets are stored as HDF5 files in base_dir / data / labeled_datasets.
    :param base_dir: Home directory of project. Should contain 'confinement_database.xlsx' and 'data/'
    :param df_name: Name of the confinement regime database.
    :param data_dir: Path to datasets (rel. to base_dir if str, else specify whole path.)
    :param labeled_dir: Path to labeled datasets (rel. to data_dir if str, else specify whole path.)
    :return: None
    """

    # Pathify all directories
    base_dir = Path(base_dir)
    if Path(data_dir).exists():
        data_dir = Path(data_dir)
    else:
        data_dir = Path(base_dir) / data_dir
    if Path(labeled_dir).exists():
        labeled_dir = Path(labeled_dir)
    else:
        labeled_dir = Path(data_dir) / labeled_dir
    # Find already labeled datasets
    labeled_dir.mkdir(exist_ok=True)
    labeled_shots = {}
    for file in (labeled_dir.iterdir()):
        try:
            shot_num = re.findall(r'_(\d+).+.hdf5', str(file))[0]
        except IndexError:
            continue
        if shot_num not in labeled_shots.keys():
            labeled_shots[shot_num] = file

    # Find unlabeled datasets (shots not in base_dir/data/labeled_datasets)
    shots = {}
    for file in (data_dir.iterdir()):
        try:
            shot_num = re.findall(r'_(\d+).hdf5', str(file))[0]
        except IndexError:
            continue
        if shot_num not in labeled_shots.keys():
            shots[shot_num] = file

    if len(shots) == 0:
        logger.info('No new labels to make')
        return
    logger.info(f'Making labels for shots {[sn for sn in shots.keys()]}')

    # Read labeled df.
    label_df = pd.read_excel(base_dir / df_name).fillna(0)
    for shot_num, file in shots.items():
        shot_df = label_df.loc[label_df['shot'] == float(shot_num)]
        if len(shot_df) == 0:
            print(f'{shot_num} not in confinement database.')
            continue
        else:
            print(f'Processing shot {shot_num}')

        with h5py.File(file, 'a') as shot_data:
            try:
                labels = np.array(shot_data['labels']).tolist()
            except KeyError:
                time = np.array(shot_data['time'])
                signals = np.array(shot_data['signals'])
                labels = np.zeros_like(time)

                for i, row in shot_df.iterrows():
                    tstart = row['tstart (ms)']
                    tstop = row['tstop (ms)']
                    label = row[[col for col in row.index if 'mode' in col]].values.argmax() + 1
                    labels[np.nonzero((time > tstart) & (time < tstop))] = label

                signals = signals[:, np.nonzero(labels)[0]]
                time = time[np.nonzero(labels)[0]]
                labels = labels[np.nonzero(labels)[0]] - 1

        sname = f'bes_signals_{shot_num}_labeled.hdf5'
        with h5py.File(labeled_dir / sname, 'w') as sd:
            sd.create_dataset('labels', data=labels)
            sd.create_dataset('signals', data=signals)
            sd.create_dataset('time', data=time)

    return

def plot_confusion_matrix(cm, classes,
                          ax=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if ax is None:
        ax = plt.gca()
    img = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    plt.colorbar(img, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes, rotation=45)
    ax.set_yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    return ax
