import os
import pickle
from typing import Tuple
import argparse

# import matplotlib

# matplotlib.use("TkAgg")
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from tqdm import tqdm

from data_preprocessing import *
from src import data, utils, dataset
from options.test_arguments import TestArguments

sns.set_style("white")
colors = sns.color_palette("deep").as_hex()
# colors = ["#ef476f", "#e5989b", "#fcbf49", "#06d6a0", "#118ab2", "#073b4c"]


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
    test_dataset = dataset.ELMDataset(
        args, *data_attrs, logger=logger, transform=transforms
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
                    elm_signals[j : j + args.signal_window_size, :, :],
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
                    elm_signals[j : j + args.signal_window_size, :, :].reshape(
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
        elm_time = np.arange(elm_labels.size)
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
        active_elm_lower_buffer = active_elm_start - 75
        active_elm_upper_buffer = active_elm_start + 75
        predictions = np.zeros(
            elm_labels.size
            - args.signal_window_size
            - args.label_look_ahead
            + 1
        )
        for j in range(predictions.size):
            if args.use_gradients:
                input_signals = np.array(
                    elm_signals[j : j + args.signal_window_size, :, :].reshape(
                        [1, args.signal_window_size, 8, 8, 6]
                    ),
                    dtype=np.float32,
                )
                input_signals = np.transpose(
                    input_signals, axes=(0, 4, 1, 2, 3)
                )
            else:
                input_signals = np.array(
                    elm_signals[j : j + args.signal_window_size, :, :].reshape(
                        [1, 1, args.signal_window_size, 8, 8]
                    ),
                    dtype=np.float32,
                )
            input_signals = torch.as_tensor(input_signals, dtype=torch.float32)
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
        active_elm_prediction_count = np.sum(
            micro_predictions_active_elms > 0.4
        )
        macro_predictions_active_elms = np.array(
            [(active_elm_prediction_count >= 1).astype(int)]
        )

        pre_active_elm_prediction_count = np.sum(
            micro_predictions_pre_active_elms > 0.4
        )
        macro_predictions_pre_active_elms = np.array(
            [(pre_active_elm_prediction_count >= 1).astype(int)]
        )

        macro_labels = np.array([0, 1], dtype="int")
        macro_predictions = np.concatenate(
            [macro_predictions_pre_active_elms, macro_predictions_active_elms]
        )
        elm_time = np.arange(elm_labels.size)
        elm_predictions[window_start[i_elm]] = {
            "signals": elm_signals,
            "labels": elm_labels,
            "micro_predictions": micro_predictions,
            "macro_labels": macro_labels,
            "macro_predictions": macro_predictions,
            "elm_time": elm_time,
        }
    return elm_predictions


def plot(
    args: argparse.Namespace,
    elm_predictions: dict,
    plot_dir: str,
) -> None:
    state = np.random.RandomState(seed=args.seed)
    elm_id = list(elm_predictions.keys())
    i_elms = state.choice(elm_id, args.plot_num, replace=False)

    fig = plt.figure(figsize=(14, 12))
    for i, i_elm in enumerate(i_elms):
        signals = elm_predictions[i_elm]["signals"]
        labels = elm_predictions[i_elm]["labels"]
        elm_start = np.where(labels > 0)[0][0]
        predictions = elm_predictions[i_elm]["micro_predictions"]
        elm_time = elm_predictions[i_elm]["elm_time"]
        print(f"ELM {i+1} of 12 with {len(elm_time)} time points")
        plt.subplot(args.num_rows, args.num_cols, i + 1)
        if args.use_gradients:
            plt.plot(
                elm_time,
                signals[:, 2, 6, 0] / 10,
                label="BES ch. 22",
                c=colors[0],
            )
        else:
            plt.plot(
                elm_time, signals[:, 2, 6], label="BES ch. 22", c=colors[0]
            )
        plt.plot(
            elm_time,
            labels + 0.02,
            label="Ground truth",
            ls="-.",
            lw=1.5,
            c=colors[1],
        )
        plt.plot(
            elm_time - args.label_look_ahead,
            predictions,
            label="Prediction",
            ls="-.",
            lw=1.5,
            c=colors[2],
        )
        plt.axvline(
            elm_start - 75,
            ymin=0,
            ymax=0.9,
            c=colors[3],
            ls=":",
            lw=2.0,
            label="Buffer limits",
        )
        plt.axvline(
            elm_start + 75,
            ymin=0,
            ymax=0.9,
            c=colors[3],
            ls=":",
            lw=2.0,
        )
        plt.xlabel("Time (micro-s)")
        plt.ylabel("Signal | label")
        plt.ylim([None, 1.1])
        plt.legend(fontsize=8, frameon=False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.grid(axis="y")
    plt.suptitle(f"Model output on {args.data_mode} classes", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not args.dry_run:
        fig.savefig(
            os.path.join(
                plot_dir,
                f"{args.model_name}_{args.data_mode}_lookahead_{args.label_look_ahead}_time_series{args.filename_suffix}.png",
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
        cm_log = np.log(cm)
        x, y = np.where(~np.eye(cm.shape[0], dtype=bool))
        coords = tuple(zip(x, y))
        total_error = np.sum(cm_log[coords])
        cm_log /= total_error

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
            cm_log, annot=True, ax=ax, annot_kws={"size": 14}, fmt=".3f"
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
    pred_dict = predict_v2(args, test_data, model, device)

    if args.plot_data:
        plot(args, pred_dict, plot_dir)

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
