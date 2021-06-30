import torch

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from tqdm import tqdm

import config, data, cnn_feature_model, model

sns.set_style("white")
sns.set_palette("deep")


def get_test_dataset(
    file_name: str, transforms=None
) -> Tuple[tuple, data.ELMDataset]:
    file_path = os.path.join(config.data_dir, file_name)

    with open(file_path, "rb") as f:
        test_data = pickle.load(f)

    signals = np.array(test_data["signals"])
    labels = np.array(test_data["labels"])
    sample_indices = np.array(test_data["sample_indices"])
    window_start = np.array(test_data["window_start"])
    data_attrs = (signals, labels, sample_indices, window_start)
    test_dataset = data.ELMDataset(
        *data_attrs,
        config.signal_window_size,
        config.label_look_ahead,
        stack_elm_events=config.stack_elm_events,
        transform=transforms,
    )

    return data_attrs, test_dataset