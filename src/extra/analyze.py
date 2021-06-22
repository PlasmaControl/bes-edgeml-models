import os
import pickle
from typing import Tuple
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


def plot(
    test_data: tuple,
    elm_model,
    device: torch.device,
) -> None:
    signals = test_data[0]
    labels = test_data[1]
    sample_indices = test_data[2]
    window_start = test_data[3]
    num_elms = len(window_start)
    i_elms = np.random.choice(num_elms, 12, replace=False)

    fig = plt.figure(figsize=(14, 12))
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
            predictions[j] = elm_model(input_signals, batch_size=12)
        # convert logits to probability
        predictions = torch.sigmoid(
            torch.as_tensor(predictions, dtype=torch.float32)
        )
        plt.subplot(4, 3, i + 1)
        elm_time = np.arange(elm_labels.size)
        plt.plot(elm_time, elm_signals[:, 2, 6], label="BES ch. 22")
        plt.plot(
            elm_time,
            elm_labels + 0.02,
            label="Ground truth",
            ls="-.",
            lw=2.5,
        )
        plt.plot(
            elm_time[
                (config.signal_window_size + config.label_look_ahead - 1) :
            ],
            predictions,
            label="Prediction",
            ls="-.",
            lw=2.5,
        )
        plt.xlabel("Time (micro-s)")
        plt.ylabel("Signal | label")
        plt.ylim([None, 1.1])
        plt.legend(fontsize=9, frameon=False)
        plt.suptitle(f"Model output on {config.data_mode} classes", fontsize=20)
    plt.tight_layout()
    fig.savefig(
        os.path.join(
            config.output_dir,
            f"{type(elm_model).__name__}_{config.data_mode}_classes_output.png",
        ),
        dpi=200,
    )
    plt.show()


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
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
):
    preds = (y_pred > threshold).astype(int)

    # creating a classification report
    cm = metrics.confusion_matrix(y_true, preds)
    cr = metrics.classification_report(y_true, preds, output_dict=True)
    df = pd.DataFrame(cr).transpose()
    df.to_csv(
        os.path.join(
            config.output_dir,
            f"{model_name}_classification_report_{config.data_mode}.csv",
        ),
        index=True,
    )
    print(f"Classification report:\n{df}")

    # ROC details
    fpr, tpr, thresh = metrics.roc_curve(y_true, y_pred)
    roc_details = pd.DataFrame()
    roc_details["fpr"] = fpr
    roc_details["tpr"] = tpr
    roc_details["threshold"] = thresh
    roc_details.to_csv(
        os.path.join(
            config.output_dir,
            f"{model_name}_roc_details_{config.data_mode}.csv",
        ),
        index=False,
    )

    cm_disp = metrics.ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    cm_disp.plot()
    fig = cm_disp.figure_
    fig.savefig(
        os.path.join(
            config.output_dir,
            f"{model_name}_confusion_matrix_{config.data_mode}.png",
        ),
        dpi=200,
    )
    plt.show()


def model_predict(
    elm_model,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    # put the elm_model to eval mode
    elm_model.eval()
    predictions = []
    targets = []
    for images, labels in tqdm(data_loader):
        images = images.to(device)

        with torch.no_grad():
            preds = elm_model(images)
        preds = preds.view(-1)
        predictions.append(torch.sigmoid(preds).cpu().numpy())
        targets.append(labels.cpu().numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    print(predictions[:10], targets[:10])
    print(metrics.roc_auc_score(targets, predictions))
    return targets, predictions


def main(
    fold: None = None,
    show_info: bool = True,
    plot_data: bool = False,
    display_metrics: bool = False,
) -> None:
    # instantiate the elm_model and load the checkpoint
    elm_model = cnn_feature_model.CNNModel()
    # elm_model = model.StackedELMModel()
    model_name = type(elm_model).__name__
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ckpt_path = os.path.join(
        config.model_dir,
        f"{model_name}_fold{fold}_best_roc_{config.data_mode}.pth",
    )
    print(f"Using elm_model checkpoint: {model_ckpt_path}")
    elm_model.load_state_dict(
        torch.load(
            model_ckpt_path,
            map_location=device,
        )["model"]
    )
    elm_model = elm_model.to(device)

    # get the test data and dataloader
    f_name = f"test_data_{config.data_mode}.pkl"
    print(f"Using test data file: {f_name}")
    test_transforms = data.get_transforms()
    test_data, test_dataset = get_test_dataset(
        file_name=f_name, transforms=test_transforms
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
    )

    if show_info:
        show_details(test_data)

    targets, predictions = model_predict(elm_model, device, test_loader)

    if plot_data:
        plot(test_data, elm_model, device)

    if display_metrics:
        show_metrics(model_name, targets, predictions)


if __name__ == "__main__":
    main(plot_data=True, display_metrics=True)
