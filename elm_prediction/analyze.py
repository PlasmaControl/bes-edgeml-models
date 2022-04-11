"""
Main inference and error analysis script to run inference on the test data 
using the trained model. It calculates micro and macro predictions for each ELM 
event in the test data and create metrics like confusion metrics, classification
report. It also creates (and saves) various plots such as time series plots for 
the ELM events with the ground truth and model predictions as well as the confusion
matrices for both macro and micro predictions. Using the  command line argument 
`--dry_run` will just show the plots, it will not save them.
"""
import shutil
import subprocess
import pickle
from pathlib import Path
from typing import Union
from xmlrpc.client import Boolean

import matplotlib.pyplot as plt
import numpy as np
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
except ImportError:
    from elm_prediction.data_preprocessing import *
    from elm_prediction.src import utils, dataset
    from elm_prediction.options.test_arguments import TestArguments

class Analysis(object):

    def __init__(
        self,
        run_dir: Union[Path, str, None] = None,
        device: Union[str, None] = None,
        save: Boolean = True,
    ):
        self.run_dir = Path(run_dir).resolve()
        if self.run_dir.is_file():
            self.run_dir = self.run_dir.parent
        assert self.run_dir.exists() and self.run_dir.is_dir()
        self.args_file = self.run_dir / 'args.pkl'
        assert self.args_file.exists()
        self.run_dir_short = self.run_dir.relative_to(self.run_dir.parent.parent)
        self.device = device
        self.save = save

        with self.args_file.open('rb') as f:
            args = pickle.load(f)
            # self.args = pickle.load(f)
        self.args = TestArguments().parse(existing_namespace=args)

        if self.device is None:
            self.device = self.args.device
        # if self.device.startswith('cuda'):
        #     self.device = 'cuda'
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args.device = self.device
        self.device = torch.device(self.device)

        self.output_dir = Path(self.args.output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = self.args_file.parent
        assert self.output_dir.exists()
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
        load_obj = torch.load(checkpoint_file.as_posix(), map_location=self.device)
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
        with torch.no_grad():
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
        plt.suptitle(f"{self.run_dir_short}")
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
        threshold: Union[float, None] = None,
    ):
        if self.valid_indices_data_loader is None:
            self._make_valid_indices_data_loader()
        predictions = []
        targets = []
        print('Running inference on valid indices')
        with torch.no_grad():
            for images, labels in tqdm(self.valid_indices_data_loader):
                images = images.to(self.device)
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
        _, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6))
        plt.suptitle(f"{self.run_dir_short} | Test data (valid indices)")
        plt.sca(axes.flat[0])
        plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC')
        plt.annotate(
            f'ROC-AUC = {roc_auc:.2f}',
            xy=(0.5, 0.03),
            xycoords='axes fraction',
        )
        plt.sca(axes.flat[1])
        plt.plot(thresh, tpr, label='True pos. rate')
        plt.plot(thresh, fpr, label='False pos. rate')
        plt.title('TPR/FPR')
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.xlim(0,1)
        plt.legend()
        # calc confusion matrix
        bool_predictions = (predictions > threshold).astype(int)
        cm = metrics.confusion_matrix(targets, bool_predictions)
        # plot confusion matrix
        plt.sca(axes.flat[2])
        sns.heatmap(
            cm, 
            annot=True, 
            norm=LogNorm(),
            xticklabels=['No ELM', 'ELM'],
            yticklabels=['No ELM', 'ELM'],
        )
        plt.title(f'Confusion matrix (thr={threshold:.2f})')
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        if self.save:
            filepath = self.analysis_dir / f"valid_indices_analysis.pdf"
            print(f'Saving matrix figure: {filepath.as_posix()}')
            plt.savefig(filepath.as_posix(), format='pdf', transparent=True)
    
    def plot_full_inference(self):
        if self.elm_predictions is None:
            self._calc_inference_full()
        elm_indices = self.test_data['elm_indices']
        n_elms = elm_indices.size
        i_page = 1
        for i_elm, elm_index in enumerate(elm_indices):
            if i_elm % 6 == 0:
                _, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))
                plt.suptitle(f"{self.run_dir_short} | Test data (full)")
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
                    filepath = self.analysis_dir / f'inference_{i_page:02d}.pdf'
                    print(f'Saving inference file: {filepath.as_posix()}')
                    plt.savefig(filepath.as_posix(), format='pdf', transparent=True)
                    i_page += 1
        # merge PDFs
        if self.save:
            pdf_files = sorted(self.analysis_dir.glob('inference_*.pdf'))
            output = self.analysis_dir/'inference.pdf'
            utils.merge_pdfs(pdf_files, output, delete_inputs=True)

    def plot_full_analysis(
            self,
            threshold: Union[float, None] = None,
        ):
        if threshold is None:
            threshold = self.args.threshold
        if self.elm_predictions is None:
            self._calc_inference_full(threshold=threshold)
        _, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6))
        plt.suptitle(f"{self.run_dir_short} | Test data (full)")
        for mode in ['micro', 'macro']:
            # gather micro/macro results
            targets = []
            predictions = []
            for vals in self.elm_predictions.values():
                predictions.append(vals[f"{mode}_predictions"])
                label_key = 'labels' if mode == 'micro' else 'macro_labels'
                targets.append(vals[label_key])
            predictions = np.concatenate(predictions)
            targets = np.concatenate(targets)
            # plot ROC (micro only)
            if mode == 'micro':
                fpr, tpr, thresh = metrics.roc_curve(targets, predictions)
                plt.sca(axes.flat[0])
                plt.plot(fpr, tpr)
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.title('ROC')
                roc_auc = metrics.roc_auc_score(targets, predictions)
                plt.annotate(
                    f'ROC-AUC = {roc_auc:.2f}',
                    xy=(0.5, 0.03),
                    xycoords='axes fraction',
                )
                plt.sca(axes.flat[1])
                plt.plot(thresh, tpr, label='True pos. rate')
                plt.plot(thresh, fpr, label='False pos. rate')
                plt.title('TPR/FPR')
                plt.xlabel('Threshold')
                plt.ylabel('Rate')
                plt.xlim(0,1)
                plt.legend()
            # confusion matrix heatmaps
            if mode == 'micro':
                bool_predictions = (predictions > threshold).astype(int)
                cm = metrics.confusion_matrix(targets, bool_predictions)
            else:
                cm = metrics.confusion_matrix(targets, predictions)
            # plt.figure(figsize=(4.5, 3.5))
            # ax = plt.subplot(111)
            axis = axes.flat[2] if mode == 'micro' else axes.flat[3]
            plt.sca(axis)
            sns.heatmap(
                cm, 
                annot=True, 
                norm=LogNorm() if mode=='micro' else None,
                xticklabels=['No ELM', 'ELM'],
                yticklabels=['No ELM', 'ELM'],
            )
            plt.title(f'Conf. matrix ({mode}, thr={threshold:.2f})')
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
        plt.tight_layout()
        if self.save:
            filepath = self.analysis_dir / f"full_analysis.pdf"
            print(f'Saving full-data analysis: {filepath.as_posix()}')
            plt.savefig(filepath.as_posix(), format='pdf', transparent=True)

    def merge_all_pdfs(self):
        pdf_files = sorted(
            self.analysis_dir.glob('*.pdf'),
            key=lambda path: path.stat().st_mtime_ns,
        )
        output = self.analysis_dir/'all_figures.pdf'
        print(f"Merging all PDFs into file: {output.as_posix()}")
        utils.merge_pdfs(pdf_files, output, delete_inputs=False)

    def plot_all(self):
        self.plot_training_epochs()
        self.plot_valid_indices_analysis()
        self.plot_full_analysis()
        self.plot_full_inference()
        self.merge_all_pdfs()

    def show(self, **kwargs):
        plt.show(**kwargs)


if __name__ == "__main__":
    plt.close('all')

    run = Analysis('run_dir')
    run.plot_all()
    run.show()
