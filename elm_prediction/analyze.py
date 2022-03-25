"""
Main inference and error analysis script to run inference on the test data 
using the trained model. It calculates micro and macro predictions for each ELM 
event in the test data and create metrics like confusion metrics, classification
report. It also creates (and saves) various plots such as time series plots for 
the ELM events with the ground truth and model predictions as well as the confusion
matrices for both macro and micro predictions. Using the  command line argument 
`--dry_run` will just show the plots, it will not save them.
"""
import argparse
import shutil
import logging
import os
import pickle
from pathlib import Path
from typing import Tuple, Union
from xmlrpc.client import Boolean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn
import torch.utils.data
from matplotlib.colors import LogNorm
from sklearn import metrics
from tqdm import tqdm

try:
    from .data_preprocessing import *
    from .src import utils, dataset
    from .options.test_arguments import TestArguments
    from . import package_dir
except ImportError:
    from elm_prediction.data_preprocessing import *
    from elm_prediction.src import utils, dataset
    from elm_prediction.options.test_arguments import TestArguments
    from elm_prediction import package_dir

# sns.set_theme(style="whitegrid", palette="muted")
LABELS = ["no ELM", "ELM"]


def calc_inference(
        args: argparse.Namespace,
        logger,
        model: torch.nn.Module,
        device: torch.device,
        test_data: tuple,
) -> dict:
    """Function to create micro and macro predictions for each ELM event in the
    test data. Micro predictions are basically the model predictions calculated
    for each time step for each ELM event. Macro predictions, on the other hand,
    are calculated after dividing the micro predictions into two regions -
    `micro_predictions_pre_active_elms` and `micro_predictions_active_elms` using
    the buffer limits (75 us before and after the first time step of ELM onset).
    For both regions, a macro prediction will predict high (i.e. prediction=1)
    if atleast one micro prediction in that region predicted high. This custom
    prediction metric will put a strong restriction for a model prediction and
    will result in lot more number of false positives than false negatives for
    macro predictions.

    Args:
    -----
        args (argparse.Namespace): Argparse namespace object containing all the
            base and test arguments.
        model (object): Instance of the model used for inference.
        device (torch.device): Device where the predictions are being made.
        test_data (tuple): Tuple containing the test signals, labels, valid_indices
            and window_start_indices.

    Returns:
    --------
        Python dictionary containing the signals and true labels alongwith micro
        and macro predictions.
    """
    signals = test_data[0]
    logger.info(f"Signals shape: {signals.shape}")
    labels = test_data[1]
    _ = test_data[2]  # sample_indices
    window_start = test_data[3]
    elm_indices = test_data[4]
    num_elms = len(window_start)
    elm_predictions = dict()
    # iterate through each ELM event

    logger.info("Calculating inference on ELM events")
    for i_elm in range(num_elms):
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
        logger.info(f"  ELM {elm_indices[i_elm]:05d} ({i_elm + 1} of {num_elms})  Data size: {elm_signals.shape}")
        predictions = []
        effective_len = (
                elm_labels.size
                - args.signal_window_size
                - args.label_look_ahead
                + 1
        )
        # iterate through the each allowed time step
        for j in range(effective_len):
            # reshape the data accroding to the data preprocessing technique
            if args.data_preproc == "gradient":
                input_signals = np.array(
                    elm_signals[j:j + args.signal_window_size, :, :].reshape(
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
            outputs = model(input_signals)
            predictions.append(outputs.item())
        predictions = np.array(predictions)
        # elm_time = np.arange(elm_labels.size)
        # convert logits to probability
        # calculate micro predictions for each time step
        micro_predictions = (
            torch.sigmoid(torch.as_tensor(predictions, dtype=torch.float32))
                .cpu()
                .numpy()
        )
        micro_predictions = np.pad(
            micro_predictions,
            pad_width=(
                (args.signal_window_size - 1) + args.label_look_ahead,
                0,
            ),
            mode="constant",
            constant_values=0,
        )
        # filter labels and micro-predictions for active elm regions
        # elm_labels_active_elms = elm_labels[
        #     active_elm_lower_buffer:active_elm_upper_buffer
        # ]
        # filter labels and micro-predictions for non-active elm regions
        micro_predictions_pre_active_elms = \
            micro_predictions[:active_elm_lower_buffer]
        micro_predictions_active_elms = \
            micro_predictions[
                active_elm_lower_buffer:active_elm_upper_buffer
            ]
        # calculate macro predictions for each region
        macro_predictions_pre_active_elms = np.array(
            [np.any(micro_predictions_pre_active_elms > 0.5).astype(int)]
        )
        macro_predictions_active_elms = np.array(
            [np.any(micro_predictions_active_elms > 0.5).astype(int)]
        )

        macro_labels = np.array([0, 1], dtype="int")
        macro_predictions = np.concatenate(
            [
                macro_predictions_pre_active_elms,
                macro_predictions_active_elms,
            ]
        )
        elm_time = np.arange(elm_labels.size)
        # print(f"Signals shape: {elm_signals.shape}")
        # print(f"Labels shape: {elm_labels.shape}")
        # print(f"Time shape: {elm_time.shape}")
        elm_predictions[window_start[i_elm]] = {
            "signals": elm_signals,
            "labels": elm_labels,
            "micro_predictions": micro_predictions,
            "macro_labels": macro_labels,
            "macro_predictions": macro_predictions,
            "elm_time": elm_time,
            "elm_index": elm_indices[i_elm],
        }
    return elm_predictions


def plot_inference_on_elm_events(
        args: argparse.Namespace,
        logger,
        elm_predictions: dict,
        plot_dir: Union[str, Path] = '',
        click_through_pages: Boolean = True,  # True to click through multiple pages
        save: Boolean = True,  # save PDFs
) -> None:
    """Helper function to plot the time series plots for all the ELM events in
    the test set on multiple pages.
    Default behavior is interactive mode, and click through pages to view.
    """
    elm_ids = list(elm_predictions.keys())
    # print('elm_ids:', elm_ids)
    # n_elms = len(elm_ids)
    # num_pages = n_elms // 12 + 1 if n_elms % 12 > 0 else n_elms // 12

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(exist_ok=True, parents=True)
    i_page = 1

    for i_elm, elm in enumerate(elm_ids):
        if i_elm % 12 == 0:
            fig, axes = plt.subplots(ncols=4, nrows=3, figsize=(16, 9))
        # get ELM data
        elm_data = elm_predictions[elm]
        signals = elm_data["signals"]
        labels = elm_data["labels"]
        predictions = elm_data["micro_predictions"]
        elm_time = elm_data["elm_time"]
        elm_index = elm_data["elm_index"]
        # active_elm = np.where(labels > 0)[0]
        # active_elm_start = active_elm[0]
        # active_elm_end = active_elm[-1]
        # plot signal, labels, and prediction
        plt.sca(axes.flat[i_elm % 12])
        plt.plot(elm_time, signals[:, 2, 6] / np.max(signals[:, 2, 6]), label="BES ch 22")
        plt.plot(elm_time, labels + 0.02, label="Ground truth")
        plt.plot(elm_time, predictions, label="Prediction", lw=1.5)
        # plt.axvline(active_elm_start - args.truncate_buffer,
        #             ymin=0, ymax=0.9, c="k", ls="--", alpha=0.65, label="Buffer limits")
        plt.xlabel("Time (micro-s)")
        plt.ylabel("Signal | label")
        # plt.tick_params(axis="x", fontsize='small')
        # plt.tick_params(axis="y", fontsize='small')
        plt.ylim([None, 1.1])
        plt.legend(fontsize='small')
        plt.title(f'ELM index {elm_index}', fontsize='medium')
        if i_elm % 12 == 11 or i_elm == len(elm_ids)-1:
            plt.tight_layout()
            if save:
                filepath = plot_dir / f'elm_event_inference_plot_pg{i_page:02d}.pdf'
                logger.info(f'Saving file: {filepath.as_posix()}')
                plt.savefig(filepath.as_posix(), format='pdf', transparent=True)
                i_page += 1
            if plt.isinteractive():
                if click_through_pages:
                    # interactive; halt/block after each page
                    print('Close plot window to continue')
                    plt.show(block=True)
                else:
                    # interactive; do not halt/block after each page
                    plt.show(block=False)
            else:
                # non-interactive for figure generation in scripts without viewing
                pass
            plt.close(fig)


def plot_confusion_matrix(
        args: argparse.Namespace,
        y_true: np.ndarray,
        y_probas: np.ndarray,
        report_dir: str,
        roc_dir: str,
        plot_dir: str,
        pred_mode: str,
        save: Boolean = False,
) -> None:
    """Show metrics like confusion matrix and classification report for both
    micro and macro predictions.

    Args:
    -----
        args (argparse.Namespace): Argparse namespace object.
        y_true (np.ndarray): True labels.
        y_probas (np.ndarray): Prediction probabilities (output of sigmoid).
        report_dir (str): Output directory path to save classification reports.
        roc_dir (str): Output directory path to save TPR, FPR and threshold arrays
            to calculate ROC curves.
        plot_dir (str): Output directory path to save confusion matrix plots.
        pred_mode (str): Whether to calculate metrics for micro or macro predictions.
    """
    assert pred_mode in ['micro', 'macro']

    if pred_mode == 'micro':
        assert y_probas.dtype not in [np.dtype('int'), np.dtype('bool')]
        # calculate predictions from the probabilities
        y_preds = (y_probas > args.threshold).astype(int)
        # creating a classification report
        cm = metrics.confusion_matrix(y_true, y_preds)
        cr = metrics.classification_report(
            y_true,
            y_preds,
            output_dict=True,
            zero_division=0,
        )
    else:
        assert y_probas.dtype != np.dtype('float')
        # creating a classification report
        cm = metrics.confusion_matrix(y_true, y_probas)
        cr = metrics.classification_report(
            y_true,
            y_probas,
            output_dict=True,
            zero_division=0,
        )

    # calculate the log of the confusion matrix scaled by the
    # total error (false positives + false negatives)
    # cm_log = np.log(cm)
    # x, y = np.where(~np.eye(cm.shape[0], dtype=bool))
    # coords = tuple(zip(x, y))
    # total_error = np.sum(cm[coords])
    # cm_scaled = cm / total_error

    # classification report
    df = pd.DataFrame(cr).transpose()
    print(f"Classification report:\n{df}")
    df.to_csv(
        os.path.join(
            report_dir,
            f"classification_report_{pred_mode}.csv",
        ),
        index=True,
    )

    # ROC curve
    fpr, tpr, thresh = metrics.roc_curve(y_true, y_probas)
    roc_details = pd.DataFrame()
    roc_details["fpr"] = fpr
    roc_details["tpr"] = tpr
    roc_details["threshold"] = thresh
    roc_details.to_csv(
        os.path.join(
            roc_dir,
            f"roc_details_{pred_mode}.csv",
        ),
        index=False,
    )

    # plots confusion matrix
    plt.figure()
    ax = plt.subplot()
    sns.heatmap(
        cm,
        xticklabels=LABELS,
        yticklabels=LABELS,
        annot=True,
        ax=ax,
        annot_kws={"size": 14},
        # fmt=".3f",
        fmt="d",
        norm=LogNorm() if pred_mode == 'micro' else None,
    )
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    ax.text(
        x=0.5,
        y=1.08,
        s=f"{pred_mode} predictions",
        fontsize=14,
        ha="center",
        va="bottom",
        transform=ax.transAxes,
    )
    ax.text(
        x=0.5,
        y=1.02,
        s=f"Signal window: {args.signal_window_size}, Label look ahead: {args.label_look_ahead}",
        fontsize=12,
        alpha=0.75,
        ha="center",
        va="bottom",
        transform=ax.transAxes,
    )
    plt.tight_layout()
    if save:
        filepath = Path(plot_dir) / f"confusion_matrix_{pred_mode}.pdf"
        print(f'Saving matrix figure: {filepath.as_posix()}')
        plt.savefig(filepath.as_posix(), format='pdf', transparent=True)

    # plot ROC curve if micro
    if pred_mode == 'micro':
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.tight_layout()
        if save:
            filepath = Path(plot_dir) / f"roc_{pred_mode}.pdf"
            print(f'Saving roc plot: {filepath.as_posix()}')
            plt.savefig(filepath.as_posix(), format='pdf', transparent=True)

    if plt.isinteractive():
        plt.show(block=False)


def calc_roc_and_f1(
        args: argparse.Namespace,
        logger: logging.Logger,
        model: torch.nn.Module,
        device: torch.device,
        data: tuple,
        threshold: float = None,
        save=True,
) -> Tuple[float, float]:
    """Make predictions on the validation set to assess the model's performance
    on the test/validation set using metrics like ROC or F1-scores.
    """
    # put the model to eval mode
    model.eval()
    predictions = []
    targets = []
    # create pytorch dataset for test set
    test_dataset = dataset.ELMDataset(args, *data[0:4], logger=logger, phase="testing")
    # dataloader
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )
    inputs, _ = next(iter(data_loader))
    logger.info(f"  Input size: {inputs.shape}")
    # iterate through the dataloader
    logger.info(f"  Evaluating model")
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        with torch.no_grad():
            preds = model(images)
        preds = preds.view(-1)
        predictions.append(torch.sigmoid(preds).cpu().numpy())
        targets.append(labels.cpu().numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    # # plot confusion matrix
    # plot_confusion_matrix(
    #     args,
    #     targets,
    #     predictions,
    #     clf_report_dir.as_posix(),
    #     roc_dir.as_posix(),
    #     plot_dir.as_posix(),
    #     pred_mode='micro',
    #     save=save,
    # )
    # display ROC and F1-score
    roc_auc = metrics.roc_auc_score(targets, predictions)
    logger.info(f"  ROC score on test data: {roc_auc:.4f}")
    if threshold is None:
        threshold = args.threshold
    logger.info(f'  Threshold for F1: {threshold:.2f}')
    # f1_thresh = 0.35  # threshold for F1-score
    f1 = metrics.f1_score(
        targets,
        (predictions > threshold).astype(int),
        zero_division=0,
    )
    logger.info(f"  F1 score on test data: {f1:.4f}")
    return roc_auc, f1


def get_micro_macro_values(pred_dict: dict, mode: str):
    """Helper function to extract values from the prediction dictionary."""
    assert mode in ['micro', 'macro']
    targets = []
    predictions = []
    for vals in pred_dict.values():
        predictions.append(vals[f"{mode}_predictions"])
        label_key = 'labels' if mode == 'micro' else 'macro_labels'
        targets.append(vals[label_key])
    return np.concatenate(targets), np.concatenate(predictions)


def do_analysis(
        args_file: Union[Path, str, None] = None,
        device=None,
        interactive: Boolean = True,  # True to view immediately; False to only generate PDFs in script without viewing
        click_through_pages: Boolean = True,  # True to click through multiple pages of ELM inference
        save: Boolean = True,
) -> None:
    """Actual function encapsulating all analysis function and making inference."""
    args_file = Path(args_file)
    with args_file.open('rb') as f:
        args = pickle.load(f)
    args = TestArguments().parse(existing_namespace=args)

    output_dir = Path(args.output_dir)
    assert output_dir.exists()

    if interactive:
        plt.ion()
    else:
        plt.ioff()
        click_through_pages = False

    logger = utils.get_logger(
        script_name=__name__,
        stream_handler=True,
        log_file=(output_dir / 'analysis.log').as_posix(),
    )

    model_cls = utils.create_model_class(args.model_name)
    model = model_cls(args)

    if device is None:
        if args.device.startswith('cuda'):
            args.device = 'cuda'
        if args.device == 'auto':
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        args.device = device
    device = torch.device(args.device)

    model = model.to(device)

    # restore paths
    test_data_file, checkpoint_file, clf_report_dir, plot_dir, roc_dir = \
        utils.create_output_paths(args, infer_mode=True)
    for analysis_dir in [clf_report_dir, plot_dir, roc_dir]:
        shutil.rmtree(analysis_dir.as_posix())
        analysis_dir.mkdir()

    # load the model checkpoint
    logger.info(f"  Model checkpoint: {checkpoint_file.as_posix()}")
    load_obj = torch.load(checkpoint_file.as_posix(), map_location=device)
    model_dict = load_obj['model']
    model.load_state_dict(model_dict)

    # restore training output
    with (output_dir/'output.pkl').open('rb') as f:
        training_output = pickle.load(f)
    train_loss = training_output['train_loss']
    valid_loss = training_output['valid_loss']
    roc_scores = training_output['roc_scores']
    f1_scores = training_output['f1_scores']
    epochs = np.arange(f1_scores.size) + 1

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8,6))
    plt.sca(axes.flat[0])
    plt.plot(epochs, train_loss, label='Training loss')
    plt.plot(epochs, valid_loss, label='Valid. loss')
    plt.title('Loss')
    plt.sca(axes.flat[1])
    plt.plot(epochs, roc_scores, label='ROC-AUC')
    plt.title('ROC-AUC')
    plt.sca(axes.flat[2])
    plt.plot(epochs, f1_scores, label='F1 score')
    plt.title("F1 score")
    for axis in axes.flat:
        plt.sca(axis)
        plt.xlabel('Epoch')
        plt.xlim([0,None])
        plt.legend()
    plt.tight_layout()
    if save:
        filepath = plot_dir / f"training.pdf"
        print(f'Saving training plot: {filepath.as_posix()}')
        plt.savefig(filepath.as_posix(), format='pdf', transparent=True)

    # restore test data
    logger.info(f"  Test data file: {test_data_file.as_posix()}")
    with test_data_file.open("rb") as f:
        test_data_dict = pickle.load(f)

    logger.info("-------->  Test data information")
    for key, value in test_data_dict.items():
        logger.info(f"  {key} shape: {value.shape}")

    # convert to tuple
    test_data = (
        test_data_dict["signals"],
        test_data_dict["labels"],
        test_data_dict["sample_indices"],
        test_data_dict["window_start"],
        test_data_dict["elm_indices"],
    )

    # ROC and F1 for valid indices
    calc_roc_and_f1(args, logger, model, device, test_data)

    # get micro/macro predictions for truncated signals, labels,
    pred_dict = calc_inference(args, logger, model, device, test_data)

    # plot micro/macro confusion matrices
    for mode in ['micro', 'macro']:
        targets, predictions = get_micro_macro_values(pred_dict, mode=mode)
        plot_confusion_matrix(
            args,
            targets,
            predictions,
            clf_report_dir.as_posix(),
            roc_dir.as_posix(),
            plot_dir.as_posix(),
            pred_mode=mode,
            save=save,
        )

    plot_inference_on_elm_events(
        args,
        logger,
        pred_dict,
        plot_dir=plot_dir.as_posix(),
        click_through_pages=click_through_pages,
        save=save,
    )

    if plt.isinteractive():
        plt.show(block=True)


class Analysis(object):

    def __init__(
        self,
        args_file: Union[Path, str, None] = None,
        device: Union[str, None] = None,
        # interactive: Boolean = True,  # True to view immediately; False to only generate PDFs in script without viewing
        # click_through_pages: Boolean = True,  # True to click through multiple pages of ELM inference
        save: Boolean = True,
    ):
        self.args_file = Path(args_file)
        self.device = device
        # self.interactive = interactive
        # self.click_through_pages = click_through_pages
        self.save = save

        with self.args_file.open('rb') as f:
            args = pickle.load(f)
        self.args = TestArguments().parse(existing_namespace=args)

        if self.device is None:
            self.device = self.args.device
        if self.device.startswith('cuda'):
            self.device = 'cuda'
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)

        self.output_dir = Path(self.args.output_dir)
        self.analysis_dir = self.output_dir / 'analysis'
        shutil.rmtree(self.analysis_dir, ignore_errors=True)
        self.analysis_dir.mkdir()

        # load paths
        self.test_data_file, checkpoint_file = utils.create_output_paths(self.args)

        # instantiate model
        model_cls = utils.create_model_class(self.args.model_name)
        self.model = model_cls(self.args)
        self.model = self.model.to(self.device)
        self.model.eval()

        # load the model checkpoint
        print(f"Model checkpoint: {checkpoint_file.as_posix()}")
        load_obj = torch.load(checkpoint_file.as_posix(), map_location=device)
        model_dict = load_obj['model']
        self.model.load_state_dict(model_dict)

        self.training_output = None
        self.test_data = None
        self.valid_indices_data_loader = None
        self.elm_predictions = None

    def _load_training_output(self):
        outputs_file = self.output_dir / 'output.pkl'
        print(f'Loading training outputs: {outputs_file.as_posix()}')
        with outputs_file.open('rb') as f:
            self.training_output = pickle.load(f)

    def _load_test_data(self):
        # restore test data
        print(f"Loading test data file: {self.test_data_file.as_posix()}")
        with self.test_data_file.open("rb") as f:
            self.test_data = pickle.load(f)

        print("Test data:")
        for key, value in self.test_data.items():
            print(f"  {key} shape: {value.shape}")

    def _make_valid_indices_data_loader(self):
        if self.test_data is None:
            self._load_test_data()
        print('Creating data loader for valid indices')
        test_dataset = dataset.ELMDataset(
            self.args, 
            self.test_data['signals'],
            self.test_data['labels'],
            self.test_data['sample_indices'],
            self.test_data['window_start'],
        )
        # dataloader
        self.valid_indices_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=True,
        )
        inputs, _ = next(iter(self.valid_indices_data_loader))
        print(f"Input size: {inputs.shape}")

    def _calc_inference_full(
        self,
        threshold=None,
    ):
        if self.test_data is None:
            self._load_test_data()
        signals = self.test_data['signals']
        labels = self.test_data['labels']
        window_start = self.test_data['window_start']
        elm_indices = self.test_data['elm_indices']

        if threshold is None:
            threshold = self.args.threshold

        n_elms = elm_indices.size
        sws_plus_la = (self.args.signal_window_size - 1) + self.args.label_look_ahead

        print('Running inference on full test data')
        elm_predictions = {}
        for i_elm, elm_index in enumerate(elm_indices):
            i_start = window_start[i_elm]
            if i_elm < n_elms - 1:
                i_stop = window_start[i_elm + 1] - 1
            else:
                i_stop = labels.size
            elm_signals = signals[i_start:i_stop, ...]
            elm_labels = labels[i_start:i_stop]
            active_elm = np.where(elm_labels > 0.0)[0]
            active_elm_start = active_elm[0]
            active_elm_lower_buffer = active_elm_start - self.args.truncate_buffer
            active_elm_upper_buffer = active_elm_start + self.args.truncate_buffer
            print(f"ELM {elm_indices[i_elm]:5d} ({i_elm+1:3d} of {n_elms})  "
                  f"Signal size: {elm_signals.shape}")
            predictions = []
            effective_len = elm_labels.size - sws_plus_la
            for j in range(effective_len):
                input_signals = elm_signals[j: j + self.args.signal_window_size, ...]
                input_signals = input_signals.reshape([1, 1, self.args.signal_window_size, 8, 8])
                input_signals = torch.as_tensor(input_signals, dtype=torch.float32)
                input_signals = input_signals.to(self.device)
                outputs = self.model(input_signals)
                predictions.append(outputs.item())
            predictions = np.array(predictions)
            # micro predictions
            micro_predictions = torch.sigmoid(
                torch.as_tensor(predictions, dtype=torch.float32)
            ).cpu().numpy()
            micro_predictions = np.pad(
                micro_predictions,
                pad_width=(sws_plus_la, 0),
                mode="constant",
                constant_values=0,
            )
            # macro predictions
            micro_predictions_pre_active_elms = \
                micro_predictions[:active_elm_lower_buffer]
            macro_predictions_pre_active_elms = np.array(
                [np.any(micro_predictions_pre_active_elms > threshold).astype(int)]
            )
            micro_predictions_active_elms = \
                micro_predictions[active_elm_lower_buffer:active_elm_upper_buffer]
            macro_predictions_active_elms = np.array(
                [np.any(micro_predictions_active_elms > threshold).astype(int)]
            )
            macro_labels = np.array([0, 1], dtype="int")
            macro_predictions = np.concatenate(
                [
                    macro_predictions_pre_active_elms,
                    macro_predictions_active_elms,
                ]
            )
            elm_predictions[elm_index] = {
                "signals": elm_signals,
                "labels": elm_labels,
                "micro_predictions": micro_predictions,
                "macro_labels": macro_labels,
                "macro_predictions": macro_predictions,
            }

        self.elm_predictions = elm_predictions

    def plot_training_epochs(self):
        if self.training_output is None:
            self._load_training_output()
        train_loss = self.training_output['train_loss']
        valid_loss = self.training_output['valid_loss']
        roc_scores = self.training_output['roc_scores']
        f1_scores = self.training_output['f1_scores']
        epochs = np.arange(f1_scores.size) + 1
        _, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,3))
        plt.sca(axes.flat[0])
        plt.plot(epochs, train_loss, label='Training loss')
        plt.plot(epochs, valid_loss, label='Valid. loss')
        plt.title('Training/validation loss')
        plt.ylabel('Loss')
        plt.sca(axes.flat[1])
        plt.plot(epochs, roc_scores, label='ROC-AUC')
        plt.plot(epochs, f1_scores, label=f'F1 (thr={self.args.threshold:.2f})')
        plt.title('Validation scores')
        plt.ylabel('Score')
        for axis in axes.flat:
            plt.sca(axis)
            plt.xlabel('Epoch')
            plt.xlim([0,None])
            plt.legend()
        plt.tight_layout()
        if self.save:
            filepath = self.analysis_dir / "training.pdf"
            print(f'Saving training plot: {filepath.as_posix()}')
            plt.savefig(filepath.as_posix(), format='pdf', transparent=True)

    def plot_valid_indices_analysis(
        self, 
        threshold=None
    ):
        if self.valid_indices_data_loader is None:
            self._make_valid_indices_data_loader()
        predictions = []
        targets = []
        print('Running inference on valid indices')
        for images, labels in tqdm(self.valid_indices_data_loader):
            images = images.to(self.device)
            with torch.no_grad():
                preds = self.model(images)
            preds = preds.view(-1)
            predictions.append(torch.sigmoid(preds).cpu().numpy())
            targets.append(labels.cpu().numpy())
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        # display ROC and F1-score
        roc_auc = metrics.roc_auc_score(targets, predictions)
        print(f"ROC-AUC on valid indices: {roc_auc:.4f}")
        if threshold is None:
            threshold = self.args.threshold
        print(f'F1 threshold: {threshold:.2f}')
        f1 = metrics.f1_score(
            targets,
            (predictions > threshold).astype(int),
            zero_division=0,
        )
        print(f"F1 score on valid indices: {f1:.4f}")
        # calc TPR, FPR for ROC
        fpr, tpr, thresh = metrics.roc_curve(targets, predictions)
        # plot ROC
        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,3))
        plt.sca(axes.flat[0])
        plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC for valid indices')
        plt.annotate(
            f'ROC-AUC = {roc_auc:.2f}',
            xy=(0.5, 0.03),
            xycoords='axes fraction',
        )
        plt.sca(axes.flat[1])
        plt.plot(thresh, tpr, label='True pos. rate')
        plt.plot(thresh, fpr, label='False pos. rate')
        plt.title('TPR/FPR for valid indices')
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.xlim(0,1)
        plt.legend()
        plt.tight_layout()
        if self.save:
            filepath = self.analysis_dir / f"roc_valid_indices.pdf"
            print(f'Saving roc plot: {filepath.as_posix()}')
            plt.savefig(filepath.as_posix(), format='pdf', transparent=True)
        # calc confusion matrix
        bool_predictions = (predictions > threshold).astype(int)
        cm = metrics.confusion_matrix(targets, bool_predictions)
        # plot confusion matrix
        plt.figure(figsize=(4.5, 3.5))
        ax = plt.subplot(111)
        sns.heatmap(
            cm, 
            ax=ax, 
            annot=True, 
            norm=LogNorm(),
            xticklabels=['No ELM', 'ELM'],
            yticklabels=['No ELM', 'ELM'],
            fmt='d',
        )
        plt.title('Confusion matrix for valid indices')
        plt.annotate(
            f'Threshold={threshold:.2f}',
            xy=(0.65, 0.03),
            xycoords='figure fraction',
        )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        if self.save:
            filepath = self.analysis_dir / f"cm_valid_indices.pdf"
            print(f'Saving matrix figure: {filepath.as_posix()}')
            plt.savefig(filepath.as_posix(), format='pdf', transparent=True)
    
    def plot_full_inference(
            self,
            max_pages = None,
        ):
        if self.elm_predictions is None:
            self._calc_inference_full()
        elm_indices = self.test_data['elm_indices']
        n_elms = elm_indices.size
        i_page = 1
        for i_elm, elm_index in enumerate(elm_indices):
            if i_elm % 6 == 0:
                _, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))
            elm_data = self.elm_predictions[elm_index]
            signals = elm_data["signals"]
            labels = elm_data["labels"]
            predictions = elm_data["micro_predictions"]
            elm_time = np.arange(labels.size)
            # plot signal, labels, and prediction
            plt.sca(axes.flat[i_elm % 6])
            plt.plot(elm_time, signals[:, 2, 6] / np.max(signals[:, 2, 6]), label="BES ch 22")
            plt.plot(elm_time, labels + 0.02, label="Ground truth")
            plt.plot(elm_time, predictions, label="Prediction")
            plt.xlabel("Time (micro-s)")
            plt.ylabel("Signal | label")
            plt.ylim([None, 1.1])
            plt.legend(fontsize='small')
            plt.title(f'ELM index {elm_index}')
            if i_elm % 6 == 5 or i_elm == n_elms-1:
                plt.tight_layout()
                if self.save:
                    filepath = self.analysis_dir / f'inference_pg{i_page:02d}.pdf'
                    print(f'Saving inference file: {filepath.as_posix()}')
                    plt.savefig(filepath.as_posix(), format='pdf', transparent=True)
                    i_page += 1
                if max_pages and i_page > max_pages:
                    break

    def plot_full_analysis(
            self,
            threshold = None,
        ):
        if threshold is None:
            threshold = self.args.threshold
        if self.elm_predictions is None:
            self._calc_inference_full(threshold=threshold)
        for mode in ['micro', 'macro']:
            targets = []
            predictions = []
            for vals in self.elm_predictions.values():
                predictions.append(vals[f"{mode}_predictions"])
                label_key = 'labels' if mode == 'micro' else 'macro_labels'
                targets.append(vals[label_key])
            predictions = np.concatenate(predictions)
            targets = np.concatenate(targets)
            # plot confusion matrices
            if mode == 'micro':
                bool_predictions = (predictions > threshold).astype(int)
                cm = metrics.confusion_matrix(targets, bool_predictions)
            else:
                cm = metrics.confusion_matrix(targets, predictions)
            plt.figure(figsize=(4.5, 3.5))
            ax = plt.subplot(111)
            sns.heatmap(
                cm, 
                ax=ax, 
                annot=True, 
                norm=LogNorm() if mode=='micro' else None,
                xticklabels=['No ELM', 'ELM'],
                yticklabels=['No ELM', 'ELM'],
                fmt='d',
            )
            plt.title('Confusion matrix for full test data')
            plt.annotate(
                f'Threshold={threshold:.2f}',
                xy=(0.65, 0.03),
                xycoords='figure fraction',
            )
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.tight_layout()
            if self.save:
                filepath = self.analysis_dir / f"cm_full_{mode}.pdf"
                print(f'Saving matrix figure: {filepath.as_posix()}')
                plt.savefig(filepath.as_posix(), format='pdf', transparent=True)        
            # plot ROC (micro only)
            if mode == 'micro':
                fpr, tpr, thresh = metrics.roc_curve(targets, predictions)
                _, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,3))
                plt.sca(axes.flat[0])
                plt.plot(fpr, tpr)
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.title('ROC for full test data')
                roc_auc = metrics.roc_auc_score(targets, predictions)
                plt.annotate(
                    f'ROC-AUC = {roc_auc:.2f}',
                    xy=(0.5, 0.03),
                    xycoords='axes fraction',
                )
                plt.sca(axes.flat[1])
                plt.plot(thresh, tpr, label='True pos. rate')
                plt.plot(thresh, fpr, label='False pos. rate')
                plt.title('TPR/FPR for full test')
                plt.xlabel('Threshold')
                plt.ylabel('Rate')
                plt.xlim(0,1)
                plt.legend()
                plt.tight_layout()
                if self.save:
                    filepath = self.analysis_dir / f"roc_valid_indices.pdf"
                    print(f'Saving roc plot: {filepath.as_posix()}')
                    plt.savefig(filepath.as_posix(), format='pdf', transparent=True)

    def show(self, **kwargs):
        plt.show(**kwargs)


if __name__ == "__main__":
    plt.close('all')
    args_file = package_dir / 'run_dir/args.pkl'
    do_analysis(args_file, interactive=True, click_through_pages=True, save=True)
