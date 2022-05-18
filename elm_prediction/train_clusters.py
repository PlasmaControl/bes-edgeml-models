import time
from pathlib import Path
import pickle
import numpy as np
import h5py

from dendrogram import cluster_groups, id_elms
from elm_prediction import package_dir
from elm_prediction.train import train_loop


if __name__ == '__main__':

    with open(Path(__file__).parent / 'run_dir_classification/elm_predictions.pkl', 'r+b') as f:
        elm_predictions = pickle.load(f)

    ids_standard = id_elms(elm_predictions, scale='standard')
    cluster = cluster_groups(elm_predictions, thresh=53, ids=ids_standard)
    l_cluster = cluster[np.argmax([len(i) for i in cluster])]

    try:
        h = h5py.File('clustered_datasets/labeled_elm_events_cluster_1.hdf5', 'w-')
        for elm, vals in elm_predictions.items():
            if elm not in l_cluster:
                continue
            signals = np.transpose(vals['signals'], (1, 2, 0)).reshape(64, -1)
            labels = vals['labels']
            grp = h.create_group(f'{elm:05d}')
            grp.create_dataset('labels', data=labels)
            grp.create_dataset('signals', data=signals)
        h.close()
    except FileExistsError:
        pass

    args = {'model_name': 'multi_features_ds',
            'input_data_file': 'clustered_datasets/labeled_elm_events_cluster_1.hdf5',
            'device': 'cuda',
            'batch_size': 64,
            'n_epochs': 20,
            'max_elms': -1,
            'fraction_test': 0.05,
            'fft_num_filters': 20,
            'dwt_num_filters': 20 ,
            'signal_window_size': 256,
            'output_dir': Path(__file__).parent / 'run_dir_classification_clustered',
            }

    train_loop(args, select_elms=l_cluster)
