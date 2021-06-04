import os
import pickle

import torch
import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import utils, config, data, cnn_feature_model


def get_test_dataset(file_name: str):
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
    )

    return data_attrs, test_dataset


def plot(test_data, model, device):
    signals = test_data[0]
    labels = test_data[1]
    sample_indices = test_data[2]
    window_start = test_data[3]
    num_elms = len(window_start)
    i_elms = np.random.choice(num_elms, 12, replace=False)

    plt.figure(figsize=(18, 8))
    for i, i_elm in enumerate(i_elms):
        i_start = window_start[i_elm]
        if i_elm < num_elms - 1:
            i_stop = window_start[i_elm + 1] - 1
        else:
            i_stop = labels.size
        print(f"ELM {i+1} of 12 with {i_stop-i_start+1} time points")
        elm_signals = signals[i_start:i_stop, :, :]
        elm_labels = labels[i_start:i_stop]
        predictions = np.zeros(
            elm_labels.size
            - config.signal_window_size
            - config.label_look_ahead
            + 1
        )
        for j in range(predictions.size):
            if j % 500 == 0:
                print(f"  Time {j}")
            input_signals = torch.as_tensor(
                elm_signals[j : j + config.signal_window_size, :, :].reshape(
                    [1, 1, config.signal_window_size, 8, 8]
                ),
                dtype=torch.float32,
            )
            input_signals = input_signals.to(device)
            predictions[j] = model(input_signals, batch_size=12)
        # convert logits to probability
        predictions = torch.sigmoid(
            torch.as_tensor(predictions, dtype=torch.float32)
        )
        plt.subplot(3, 4, i + 1)
        elm_time = np.arange(elm_labels.size)
        plt.plot(elm_time, elm_signals[:, 2, 6], label="BES ch. 22")
        plt.plot(elm_time, elm_labels + 0.02, label="Ground truth", ls="-.")
        plt.plot(
            elm_time[
                (config.signal_window_size + config.label_look_ahead - 1) :
            ],
            predictions,
            label="Prediction",
            ls="-.",
        )
        plt.xlabel("Time (micro-s)")
        plt.ylabel("Signal | label")
        plt.ylim([None, 1.1])
        plt.legend(fontsize=14)
        plt.suptitle(f"Model output on {config.data_mode} classes", fontsize=20)
    plt.tight_layout()
    plt.show()


def show_details(test_data):
    print("Test data information")
    signals = test_data[0]
    labels = test_data[1]
    sample_indices = test_data[2]
    window_start = test_data[3]
    print(f"Signals shape: {signals.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample indices shape: {sample_indices.shape}")
    print(f"Window start indices: {window_start}")


def model_predict(model, device, data_loader):
    # put the model to eval mode
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
    print(roc_auc_score(targets, predictions))


def main(fold=None, show_info=True, plot_data=False):
    # instantiate the model and load the checkpoint
    model = cnn_feature_model.FeatureModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ckpt_path = os.path.join(
        config.model_dir,
        f"{config.model_name}_fold{fold}_best_roc_{config.data_mode}.pth",
    )
    print(f"Using model checkpoint: {model_ckpt_path}")
    model.load_state_dict(
        torch.load(
            model_ckpt_path,
            map_location=device,
        )["model"]
    )
    model = model.to(device)

    # get the test data and dataloader
    f_name = f"test_data_{config.data_mode}.pkl"
    print(f"Using test data file: {f_name}")
    test_data, test_dataset = get_test_dataset(file_name=f_name)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
    )

    if show_info:
        show_details(test_data)

    model_predict(model, device, test_loader)

    if plot_data:
        plot(test_data, model, device)


if __name__ == "__main__":
    main(plot_data=True)
