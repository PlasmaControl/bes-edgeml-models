import logging

import numpy as np
import re
import h5py
import pandas as pd
from pathlib import Path


def make_labels(base_dir: str | Path,
                logger: logging.Logger,
                df_name: str = 'confinement_database.xlsx',
                data_dir: str = 'data',
                labeled_dir: str = 'labeled_datasets'):
    """
    Function to create labeled datasets for turbulence regime classification.
    Shot data is sourced from base_dir / data.
    Resulting labeled datasets are stored as HDF5 files in base_dir / data / labeled_datasets.
    :param base_dir: Home directory of project. Should contain 'confinement_database.xlsx' and 'data/'
    :param df_name: Name of the confinement regime database.
    :param data_dir: Path to datasets (rel. to base_dir)
    :param labeled_dir: Path to labeled datasets (rel. to data_dir)
    :return: None
    """

    # Pathify all directories
    base_dir = Path(base_dir)
    data_dir = Path(base_dir) / data_dir
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
            shot_num = re.findall(r'_(\d+).+.hdf5', str(file))[0]
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
                labels = np.zeros_like(time)
                for i, row in shot_df.iterrows():
                    tstart = row['tstart (ms)']
                    tstop = row['tstop (ms)']
                    label = row[[col for col in row.index if 'mode' in col]].values.argmax() + 1
                    labels[np.nonzero((time > tstart) & (time < tstop))] = label

            with h5py.File(labeled_dir, 'w') as sd:
                for group in shot_data.keys():
                    shot_data.copy(group, sd)
                sd.create_dataset('labels', data=labels)

    return
