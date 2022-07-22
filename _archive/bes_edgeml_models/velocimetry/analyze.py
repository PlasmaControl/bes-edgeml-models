"""
Main inference and error analysis script to run inference on the test data
using the trained model. It calculates micro and macro predictions for each ELM
event in the test data and create metrics like confusion metrics, classification
report. It also creates (and saves) various plots such as time series plots for
the ELM events with the ground truth and model predictions as well as the confusion
matrices for both macro and micro predictions. Using the  command line argument
`--dry_run` will just show the plots, it will not save them.
"""
import pickle
from typing import Union
import shutil
from pathlib import Path

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
    from .src import utils, dataset
    from .options.test_arguments import TestArguments
except ImportError:
    from bes_edgeml_models.base.src import utils, dataset
    from bes_edgeml_models.base.options.test_arguments import TestArguments

class Analysis(object):

    def __init__(
        self,
        run_dir: Union[Path, str, None] = None,
        device: Union[str, None] = None,
        save: bool = True,
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
        self.args = TestArguments().parse(existing_namespace=args)
        self.is_classification = not self.args.regression

        if self.device is None:
            self.device = self.args.device
        if self.device == 'auto' or self.device.startswith('cuda'):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args.device = self.device
        self.device = torch.device(self.device)

        self.output_dir = Path(self.args.output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = self.args_file.parent
        assert self.output_dir.exists(), f"{self.output_dir} does not exist."
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
        self.vel_predictions = None

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
        max_elms=None,
    ):
        if self.test_data is None:
            self._load_test_data()
        signals = self.test_data['signals']
        vZ_labels = self.test_data['vZ'][self.args.signal_window_size:]
        vR_labels = self.test_data['vR'][self.args.signal_window_size:]

        print('Running inference on full test data')
        with torch.no_grad():
            predictions = []
            effective_len = signals.shape[0] - self.args.signal_window_size
            for j in tqdm(range(effective_len), desc="Processing BES velocimetry."):
                input_signals = signals[j: j + self.args.signal_window_size, ...]
                input_signals = input_signals.reshape([1, 1, self.args.signal_window_size, 8, 8])
                input_signals = torch.as_tensor(input_signals, dtype=torch.float32)
                input_signals = input_signals.to(self.device)
                outputs = self.model(input_signals)
                predictions.append(outputs.cpu().squeeze().numpy())
            predictions = np.array(predictions)
            vR_predictions = predictions[:, :64].reshape((-1, 8, 8))
            vZ_predictions = predictions[:, 64:].reshape((-1, 8, 8))

            vel_predictions = {
                "signals": signals,
                "vR_labels": vR_labels,
                "vZ_labels": vZ_labels,
                "vR_predictions": vR_predictions,
                "vZ_predictions": vZ_predictions
            }

        self.vel_predictions = vel_predictions

        return vel_predictions

    def plot_training_epochs(self):
        if self.training_output is None:
            self._load_training_output()
        train_loss = self. training_output['train_loss']
        valid_loss = self.training_output['valid_loss']
        epochs = np.arange(train_loss.size) + 1
        _, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,3))
        plt.suptitle(f"{self.run_dir_short}")
        plt.sca(axes.flat[0])
        plt.plot(epochs, train_loss, label='Training loss')
        plt.plot(epochs, valid_loss, label='Valid. loss')
        plt.title('Training/validation loss')
        plt.ylabel('Loss')
        plt.sca(axes.flat[1])
        # if self.model.model_type == 'elm_prediction':
        scores = self.training_output.get('scores', None)
        roc_scores = self.training_output.get('roc_scores', None)

        if self.model.model_type == 'elm_prediction':
            if self.args.regression:
                plt.plot(epochs, scores, label=f'R2')
            else:
                plt.plot(epochs, roc_scores, label='ROC-AUC')
                plt.plot(epochs, scores, label=f'F1 (thr={self.args.threshold:.2f})')

        elif self.model.model_type == 'turbulence_regime_classification':
            plt.plot(epochs, roc_scores, label='ROC-AUC')

        elif self.model.model_type == 'velocimetry':
            plt.plot(epochs, scores, label='MAPE')

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
        if not self.valid_indices_data_loader:
            self._make_valid_indices_data_loader()
        predictions = []
        targets = []
        print('Running inference on valid indices')
        with torch.no_grad():
            for images, labels in tqdm(self.valid_indices_data_loader):
                images = images.to(self.device)
                preds = self.model(images)
                preds = preds.view(-1)
                if self.args.regression is False:
                    predictions.append(torch.sigmoid(preds).cpu().numpy())
                else:
                    predictions.append(preds.cpu().numpy())
                targets.append(labels.cpu().numpy())
        predictions = np.concatenate(predictions)
        # print(predictions.min(), predictions.max())
        targets = np.concatenate(targets)
        # print(targets.min(), targets.max())
        print(targets.shape, targets.dtype, predictions.shape, predictions.dtype)
        if self.args.regression is False:
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
        else:
            r2 = metrics.r2_score(targets[:, 2, 6], predictions[:, 2, 6])
            print(f"R2: {r2:.2f}")

    def plot_full_inference(
            self,
            max_elms=None,
        ):
        if not self.vel_predictions:
            self._calc_inference_full(max_elms=max_elms)
        _, axes = plt.subplots(ncols=1, nrows=2, figsize=(12, 6))
        plt.suptitle(f"{self.run_dir_short} | Test data (full)")
        signals = self.vel_predictions["signals"]
        vZ_labels = self.vel_predictions["vZ_labels"]
        vR_labels = self.vel_predictions["vR_labels"]
        vZ_predictions = self.vel_predictions["vZ_predictions"]
        vR_predictions = self.vel_predictions["vR_predictions"]

        # Pad labels and predictions to account for signal window
        l_diff = len(signals) - len(vZ_labels)
        vZ_labels = np.pad(vZ_labels, ((l_diff, 0), (0, 0), (0, 0)))
        vR_labels = np.pad(vR_labels, ((l_diff, 0), (0, 0), (0, 0)))
        vZ_predictions = np.pad(vZ_predictions, ((l_diff, 0), (0, 0), (0, 0)))
        vR_predictions = np.pad(vR_predictions, ((l_diff, 0), (0, 0), (0, 0)))

        elm_time = np.arange(len(signals))
        # plot signal, labels, and prediction
        axes[0].plot(elm_time, signals[:, 2, 6] / np.max(signals[:, 2, 6]), label="BES ch 22", alpha=0.5, zorder=0.0)
        axes[0].plot(elm_time, vZ_labels.mean(axis=(1, 2)) / np.max(vZ_labels.mean(axis=(1, 2))),
                     label="Ground truth",
                     alpha=0.5,
                     zorder=1.0)
        axes[0].plot(elm_time,
                     vZ_predictions.mean(axis=(1, 2)) / np.max(vZ_predictions.mean(axis=(1, 2))),
                     label="Mean Prediction",
                     alpha=0.5,
                     zorder=2.0)
        axes[0].set_xlabel("Time (micro-s)")
        axes[0].set_ylabel("Signal")
        axes[0].legend(fontsize='small')
        axes[0].set_title(f'vZ')

        axes[1].plot(elm_time, signals[:, 2, 6] / np.max(signals[:, 2, 6]), label="BES ch 22", alpha=0.5, zorder=0.0)
        axes[1].plot(elm_time, vR_labels.mean(axis=(1, 2)) / np.max(vR_labels.mean(axis=(1, 2))),
                     label="Ground truth",
                     alpha=0.5,
                     zorder=1.0)
        axes[1].plot(elm_time,
                     vR_predictions.mean(axis=(1, 2)) / np.max(vR_predictions.mean(axis=(1, 2))),
                     label="Mean Prediction",
                     alpha=0.5,
                     zorder=2.0)
        axes[1].set_xlabel("Time (micro-s)")
        axes[1].set_ylabel("Signal")
        axes[1].legend(fontsize='small')
        axes[1].set_title(f'vR')

        plt.tight_layout()
        if self.save:
            filepath = self.analysis_dir / f'inference.pdf'
            print(f'Saving inference file: {filepath.as_posix()}')
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
        self.plot_full_inference()
        self.merge_all_pdfs()

    def show(self, **kwargs):
        plt.show(**kwargs)


if __name__ == "__main__":
    plt.close('all')

    # run = Analysis('run_dir')
    # run = Analysis('/home/dsmith/scratch/edgeml/work/study_05/s05_cnn_logreg_adam/trial_0006')
    # run.plot_training_epochs()
    # run2 = Analysis('/home/dsmith/scratch/edgeml/work/study_05/s05_cnn_class_adam/trial_0003')
    # run2.plot_training_epochs()

    for run_dir in [
        # '/home/dsmith/scratch/edgeml/work/study_05/s05_cnn_class_adam/trial_0003',
        '/home/jazimmerman/PycharmProjects/bes-edgeml-models/bes-edgeml-work/vel_cnn_10e_sws256',
    ]:
        run = Analysis(run_dir=run_dir)
        vpd = run.plot_all()