import os
import pickle
from typing import Tuple
import argparse

# import matplotlib

# matplotlib.use("TkAgg")
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from tqdm import tqdm

from src import data, utils, run
from options.test_arguments import TestArguments

sns.set_style("white")
sns.set_palette("deep")


def get_test_dataset(
    args: argparse.Namespace, file_name: str, logger=None, transforms=None
) -> Tuple[tuple, data.ELMDataset]:
    with open(file_name, "rb") as f:
        test_data = pickle.load(f)

    signals = np.array(test_data["signals"])
    labels = np.array(test_data["labels"])
    sample_indices = np.array(test_data["sample_indices"])
    window_start = np.array(test_data["window_start"])
    data_attrs = (signals, labels, sample_indices, window_start)
    test_dataset = data.ELMDataset(args, *data_attrs, logger=logger)

    return data_attrs, test_dataset


def predict(
    args: argparse.Namespace,
    test_data: tuple,
    model: object,
    device: torch.device,
):
    signals = test_data[0]
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
        elm_signals = signals[i_start:i_stop]
        elm_labels = labels[i_start:i_stop]
        active_elm = np.where(elm_labels > 0.0)[0]
        elm_start = active_elm[0]
        elm_end = active_elm[-1]
        # add a buffer of a given number of time frames
        # in both start and end
        elm_start_extended = (elm_start - 500) if (elm_start - 500) > 0 else 0
        elm_end_extended = (
            (elm_end + 500)
            if (elm_end + 500) < len(elm_labels) - 1
            else len(elm_labels) - 1
        )
        elm_signals_extended = elm_signals[elm_start_extended:elm_end_extended]
        elm_labels_extended = elm_labels[elm_start_extended:elm_end_extended]
        predictions = np.zeros(
            elm_labels_extended.size
            - args.signal_window_size
            - args.label_look_ahead
            + 1
        )
        for j in range(predictions.size):
            input_signals = torch.as_tensor(
                elm_signals_extended[
                    j : j + args.signal_window_size, :, :
                ].reshape([1, 1, args.signal_window_size, 8, 8]),
                dtype=torch.float32,
            )
            input_signals = input_signals.to(device)
            predictions[j] = model(input_signals)
        # convert logits to probability
        predictions = (
            torch.sigmoid(torch.as_tensor(predictions, dtype=torch.float32))
            .cpu()
            .numpy()
        )
        elm_predictions[window_start[i_elm]] = {
            "signals": elm_signals_extended,
            "labels": elm_labels_extended,
            "predictions": predictions,
            "elm_time": np.arange(elm_start_extended, elm_end_extended),
        }
    return elm_predictions


def plot(
    args: argparse.Namespace,
    test_data: tuple,
    model: object,
    device: torch.device,
    plot_dir: str,
) -> None:
    signals = test_data[0]
    labels = test_data[1]
    sample_indices = test_data[2]
    window_start = test_data[3]
    num_elms = len(window_start)
    i_elms = np.random.choice(num_elms, args.plot_num, replace=False)

    fig = plt.figure(figsize=(14, 12))
    for i, i_elm in enumerate(i_elms):
        i_start = window_start[i_elm]
        if i_elm < num_elms - 1:
            i_stop = window_start[i_elm + 1] - 1
        else:
            i_stop = labels.size
        if (i_stop - i_start + 1) <= args.label_look_ahead:
            print(
                f"Skipping ELM {i+1} of 12 with {i_stop-i_start+1} time points"
            )
            continue
        else:
            print(f"ELM {i+1} of 12 with {i_stop-i_start+1} time points")
            elm_signals = signals[i_start:i_stop, :, :]
            elm_labels = labels[i_start:i_stop]
            predictions = np.zeros(
                elm_labels.size
                - args.signal_window_size
                - args.label_look_ahead
                + 1
            )
            for j in range(predictions.size):
                if j % 500 == 0:
                    print(f"  Time {j}")
                input_signals = torch.as_tensor(
                    elm_signals[j : j + args.signal_window_size, :, :].reshape(
                        [1, 1, args.signal_window_size, 8, 8]
                    ),
                    dtype=torch.float32,
                )
                input_signals = input_signals.to(device)
                predictions[j] = model(input_signals)
        # convert logits to probability
        predictions = (
            torch.sigmoid(torch.as_tensor(predictions, dtype=torch.float32))
            .cpu()
            .numpy()
        )
        plt.subplot(args.num_rows, args.num_cols, i + 1)
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
            elm_time[(args.signal_window_size + args.label_look_ahead - 1) :]
            - args.label_look_ahead,
            predictions,
            label="Prediction",
            ls="-.",
            lw=2.5,
        )
        plt.xlabel("Time (micro-s)")
        plt.ylabel("Signal | label")
        plt.ylim([None, 1.1])
        plt.legend(fontsize=9, frameon=False)
        plt.suptitle(f"Model output on {args.data_mode} classes", fontsize=20)
    plt.tight_layout()
    if not args.dry_run:
        fig.savefig(
            os.path.join(
                plot_dir,
                f"{args.model_name}_{args.data_mode}_lookahead_{args.label_look_ahead}_time_series.png",
            ),
            dpi=100,
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
    args: argparse.Namespace,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    report_dir: str,
    roc_dir: str,
    plot_dir: str,
):
    preds = (y_pred > args.threshold).astype(int)

    # creating a classification report
    cm = metrics.confusion_matrix(y_true, preds)
    cr = metrics.classification_report(y_true, preds, output_dict=True)
    df = pd.DataFrame(cr).transpose()
    print(f"Classification report:\n{df}")

    # ROC details
    fpr, tpr, thresh = metrics.roc_curve(y_true, y_pred)
    roc_details = pd.DataFrame()
    roc_details["fpr"] = fpr
    roc_details["tpr"] = tpr
    roc_details["threshold"] = thresh

    cm_disp = metrics.ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    cm_disp.plot()
    if not args.dry_run:
        df.to_csv(
            os.path.join(
                report_dir,
                f"{args.model_name}_classification_report_{args.data_mode}_lookahead_{args.label_look_ahead}.csv",
            ),
            index=True,
        )
        roc_details.to_csv(
            os.path.join(
                roc_dir,
                f"{args.model_name}_roc_details_{args.data_mode}_lookahead_{args.label_look_ahead}.csv",
            ),
            index=False,
        )
        fig = cm_disp.figure_
        fig.savefig(
            os.path.join(
                plot_dir,
                f"{args.model_name}_confusion_matrix_{args.data_mode}_lookahead_{args.label_look_ahead}.png",
            ),
            dpi=100,
        )
    plt.show()


def model_predict(
    model,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
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
    print(metrics.roc_auc_score(targets, predictions))
    return targets, predictions


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
        f"{args.model_name}_{args.data_mode}_lookahead_{args.label_look_ahead}.pth",
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
        f"test_data_{args.data_mode}_lookahead_{args.label_look_ahead}.pkl",
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

    if args.test_data_info:
        show_details(test_data)

    targets, predictions = model_predict(model, device, test_loader)

    if args.plot_data:
        plot(args, test_data, model, device, plot_dir)

    if args.show_metrics:
        show_metrics(
            args, targets, predictions, clf_report_dir, roc_dir, plot_dir
        )
    pred_dict = predict(args, test_data, model, device)
    print(pred_dict)


if __name__ == "__main__":
    args, parser = TestArguments().parse(verbose=True)
    utils.test_args_compat(args, parser, infer_mode=True)
    main(args)
