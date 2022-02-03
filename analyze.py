import os
import pickle
from typing import Tuple, List, Union
import argparse
import logging

import torch
import numpy as np
from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from sklearn import metrics
from tqdm import tqdm

from data_preprocessing import *
from src import utils, dataset
from options.test_arguments import TestArguments

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.25)
palette = list(sns.color_palette("muted").as_hex())
LABELS = ["no ELM", "ELM"]


def get_test_dataset(args: argparse.Namespace, file_name: str, logger: logging.getLogger = None, ) -> Tuple[
    tuple, dataset.ELMDataset]:
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
    test_dataset = dataset.ELMDataset(args, *data_attrs, logger=logger, phase="testing")

    return data_attrs, test_dataset


def predict(args: argparse.Namespace, test_data: tuple, model: object, hook_layer: str, device: torch.device) -> dict:
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
            o = output.cpu().detach().squeeze().tolist()
            activations.append(o)
        return hook

    (act_layer, weight_layer) = get_layer(model, hook_layer)

    act_layer.register_forward_hook(get_activation())
    weights = weight_layer.weight.cpu().detach().squeeze().numpy()
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

        elm_signals = elm_signals[: (-args.signal_window_size - args.label_look_ahead + 1), ...]
        elm_labels = elm_labels[: (-args.signal_window_size - args.label_look_ahead + 1)]
        # convert logits to probability
        # calculate micro predictions for each time step
        micro_predictions = (torch.sigmoid(torch.as_tensor(predictions, dtype=torch.float32)).cpu().numpy()
        )
        # filter labels and micro-predictions for active elm regions
        elm_labels_active_elms = elm_labels[active_elm_lower_buffer:active_elm_upper_buffer]
        micro_predictions_active_elms = micro_predictions[active_elm_lower_buffer:active_elm_upper_buffer]

        # filter labels and micro-predictions for non-active elm regions
        elm_labels_pre_active_elms = elm_labels[:active_elm_lower_buffer]
        micro_predictions_pre_active_elms = micro_predictions[:active_elm_lower_buffer]

        # calculate macro predictions for each region
        macro_predictions_active_elms = np.array(
            [np.any(micro_predictions_active_elms).astype(int)]
        )

        macro_predictions_pre_active_elms = np.array([np.any(micro_predictions_pre_active_elms > 0.5).astype(int)])

        macro_labels = np.array([0, 1], dtype="int")
        macro_predictions = np.concatenate([macro_predictions_pre_active_elms, macro_predictions_active_elms])
        elm_time = np.arange(elm_labels.size) + window_start[i_elm]
        activations_ = activations
        activations = []
        elm_predictions[window_start[i_elm]] = {"activations": np.asarray(activations_), "weights": weights,
                "signals": elm_signals, "labels": elm_labels, "micro_predictions": micro_predictions,
                "macro_labels": macro_labels, "macro_predictions": macro_predictions, "elm_time": elm_time, }
    return elm_predictions


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


def plot(args: argparse.Namespace, elm_predictions: dict, plot_dir: str, elms: List[int], elm_range: str,
        n_rows: Union[int, None] = None, n_cols: Union[int, None] = None, figsize: tuple = (14, 12), ) -> None:
    flag = False
    fig = plt.figure(figsize=figsize)
    for i, i_elm in enumerate(elms):
        signals = elm_predictions[i_elm]["signals"]
        signal_max = np.max(signals)
        labels = elm_predictions[i_elm]["labels"]
        try:
            elm_start = np.where(labels > 0)[0][0]
            elm_end = np.where(labels > 0)[0][-1]
        except IndexError:
            elm_start = len(labels) - 80
            elm_end = len(labels)
            flag = True
        predictions = elm_predictions[i_elm]["micro_predictions"]
        elm_time = elm_predictions[i_elm]["elm_time"]
        print(f"ELM {i + 1} of {len(elms)} with {len(elm_time)} time points")
        if (n_rows is not None) and (n_cols is not None):
            plt.subplot(n_rows, n_cols, i + 1)
        else:
            plt.subplot(args.num_rows, args.num_cols, i + 1)
        # if args.use_gradients:
        #     plt.plot(
        #         elm_time,
        #         signals[:, 2, 6, 0] / np.max(signals),
        #         label="BES ch. 22",
        #         lw=1.25,
        #         # c=colors[0],
        #     )
        # else:
        #     # plt.plot(
        #     #     elm_time,
        #     #     signals[:, 0, 0] / signal_max,
        #     #     label="Ch. 1",  # c=colors[0]
        #     #     lw=1.25,
        #     # )
        plt.plot(elm_time, signals[:, 2, 6] / np.max(signals),  # / signal_max,
                label="Ch. 22",  # c=colors[0]
                lw=1.25, )
        # plt.plot(
        #     elm_time,
        #     signals[:, 7, 7],  # / signal_max,
        #     label="Ch. 64",  # c=colors[0]
        #     lw=1.25,
        # )
        plt.plot(elm_time, labels + 0.02, label="Ground truth", ls="-", lw=1.25, # c=colors[1],
        )
        plt.plot(elm_time,  # - args.label_look_ahead,
                predictions, label="Prediction", ls="-", lw=1.25, # c="slategrey",
        )
        if flag:
            plt.axvline(elm_start - args.truncate_buffer, ymin=0, ymax=0.9, c="r", ls="--", alpha=0.65, lw=1.5,
                    label="Buffer limits", )
            plt.axvline(elm_start + args.truncate_buffer, ymin=0, ymax=0.9, c="r", ls="--", alpha=0.65, lw=1.5, )
        else:
            plt.axvline(elm_start - args.truncate_buffer, ymin=0, ymax=0.9, c="k", ls="--", alpha=0.65, lw=1.5,
                    label="Buffer limits", )
            plt.axvline(elm_end, ymin=0, ymax=0.9, c="k", ls="--", alpha=0.65, lw=1.5, )
        plt.xlabel("Time (micro-s)", fontsize=10)
        plt.ylabel("Signal | label", fontsize=10)
        plt.tick_params(axis="x", labelsize=8)
        plt.tick_params(axis="y", labelsize=8)
        plt.ylim([None, 1.1])
        sns.despine(offset=10, trim=False)
        plt.legend(fontsize=7, ncol=2, frameon=False)
        plt.gca().spines["left"].set_color("lightgrey")
        plt.gca().spines["bottom"].set_color("lightgrey")
        plt.grid(axis="y")
        flag = False
        if i == 36:
            break
    plt.suptitle(f"Model output, ELM index: {elm_range}", fontsize=20, )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not args.dry_run:
        fig.savefig(os.path.join(plot_dir,
                f"{args.model_name}_lookahead_{args.label_look_ahead}_{args.data_preproc}_time_series_{elm_range}.png", ),
                dpi=200, )
    plt.show()


def plot_all(args: argparse.Namespace, elm_predictions: dict, plot_dir: str, ) -> None:
    state = np.random.RandomState(seed=args.seed)
    elm_id = list(elm_predictions.keys())
    # i_elms = state.choice(elm_id, args.plot_num, replace=False)
    i_elms_1_12 = elm_id[:12]
    i_elms_12_24 = elm_id[12:24]
    i_elms_24_36 = elm_id[24:36]
    # i_elms_36_42 = elm_id[36:]

    # plot 1-12
    plot(args, elm_predictions, plot_dir, i_elms_1_12, elm_range="1-12")
    # plot 12-24
    plot(args, elm_predictions, plot_dir, i_elms_12_24, elm_range="12-24")
    # plot 24-36
    plot(args, elm_predictions, plot_dir, i_elms_24_36,
         elm_range="24-36")  # plot 36-42  # plot(  #     args,  #     elm_predictions,  #     plot_dir,  #     i_elms_36_42,  #     elm_range="36-42",  #     n_rows=2,  #     n_cols=3,  #     figsize=(14, 6),  # )


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


def show_metrics(args: argparse.Namespace, y_true: np.ndarray, y_probas: np.ndarray, report_dir: str, roc_dir: str,
        plot_dir: str, pred_mode: str, ):
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
        cm_scaled = cm / total_error

        cr = metrics.classification_report(y_true, y_preds, output_dict=True)
        df = pd.DataFrame(cr).transpose()
        print(f"Classification report:\n{df}")

        # ROC details
        fpr, tpr, thresh = metrics.roc_curve(y_true, y_probas)
        roc_details = pd.DataFrame()
        roc_details["fpr"] = fpr
        roc_details["tpr"] = tpr
        roc_details["threshold"] = thresh

        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot()
        sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True, ax=ax, annot_kws={"size": 14}, # fmt=".3f",
                fmt="d", norm=LogNorm(), )
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
        ax.text(x=0.5, y=1.01, s=f"Signal window: {args.signal_window_size}, Label look ahead: {args.label_look_ahead}",
                fontsize=12, alpha=0.75, ha="center", va="bottom", transform=ax.transAxes, )
        plt.tight_layout()
        if not args.dry_run:
            df.to_csv(
                os.path.join(report_dir,
                        f"{args.model_name}_classification_report_micro_lookahead_{args.label_look_ahead}_{args.data_preproc}.csv", ),
                index=True,
            )
            roc_details.to_csv(
                os.path.join(roc_dir,
                        f"{args.model_name}_roc_details_micro_lookahead_{args.label_look_ahead}_{args.data_preproc}.csv", ),
                index=False,
            )
            fig.savefig(
                os.path.join(plot_dir,
                        f"{args.model_name}_confusion_matrix_micro_lookahead_{args.label_look_ahead}_{args.data_preproc}.png", ),
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
        sns.heatmap(cm, annot=True, xticklabels=LABELS, yticklabels=LABELS, ax=ax, annot_kws={"size": 14}, fmt="d", )
        plt.setp(ax.get_yticklabels(), rotation=0)
        ax.set_xlabel("Predicted Label", fontsize=14)
        ax.set_ylabel("True Label", fontsize=14)
        ax.text(x=0.5, y=1.05, s="Macro predictions", fontsize=18, ha="center", va="bottom",
            transform=ax.transAxes,
        )
        ax.text(x=0.5, y=1.01, s=f"Signal window: {args.signal_window_size}, Label look ahead: {args.label_look_ahead}",
                fontsize=12, alpha=0.75, ha="center", va="bottom", transform=ax.transAxes, )
        plt.tight_layout()
        if not args.dry_run:
            df.to_csv(
                os.path.join(report_dir,
                        f"{args.model_name}_classification_report_macro_lookahead_{args.label_look_ahead}_{args.data_preproc}.csv", ),
                index=True,
            )
            roc_details.to_csv(
                os.path.join(roc_dir,
                        f"{args.model_name}_roc_details_macro_lookahead_{args.label_look_ahead}_{args.data_preproc}.csv", ),
                index=False,
            )
            fig.savefig(
                os.path.join(plot_dir,
                        f"{args.model_name}_confusion_matrix_macro_lookahead_{args.label_look_ahead}_{args.data_preproc}.png", ),
                dpi=100,
            )
        plt.show()
    else:
        raise ValueError(
            f"Expected pred_mode to be either `micro` or`macro` but {pred_mode} is passed."
        )


def model_predict(model, device: torch.device, data_loader: torch.utils.data.DataLoader, ) -> None:
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
    weight_idx = layer_idx + 1 if layer_idx != len(layer_dict) - 1 else layer_idx

    act_layer = layer_dict[layer]
    weight_layer = list(layer_dict.items())[weight_idx][1]

    return (act_layer, weight_layer)


def main(
        args: argparse.Namespace,
) -> None:
    logger = utils.make_logger(script_name=__name__, stream_handler=True, )
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
    model_ckpt_path = os.path.join(model_ckpt_dir, f"{args.model_name}_lookahead_{args.label_look_ahead}_"
                                                   f"{args.data_preproc}"
                                                   f"{'_' + args.balance_data if args.balance_data else ''}.pth", )
    print(f"Using elm_model checkpoint: {model_ckpt_path}")
    model.load_state_dict(
        torch.load(
            model_ckpt_path,
            map_location=device,
        )["model"]
    )

    # get the test data and dataloader
    test_fname = os.path.join(test_data_dir, f"test_data_lookahead_{args.label_look_ahead}_{args.data_preproc}.pkl", )

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
    pred_dict = predict(args, test_data, model, device)

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
