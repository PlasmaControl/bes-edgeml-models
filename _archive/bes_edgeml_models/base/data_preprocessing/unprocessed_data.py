"""
Data class to package BES data for training using PyTorch without any 
modifications and transformations.
"""
from typing import Tuple
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt

try:
    from .base_data import BaseData
    from ..options.train_arguments import TrainArguments
    from ..src import utils
except ImportError:
    from models.bes_edgeml_models.data_preprocessing.base_data import BaseData
    from models.bes_edgeml_models.options.train_arguments import TrainArguments
    from models.bes_edgeml_models.src import utils


class UnprocessedData(BaseData):
    def _preprocess_data(
        self,
        elm_indices: np.ndarray = None,
        shuffle_sample_indices: bool = False,
        oversample_active_elm: bool = False,
        save_filename: str = '',
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Helper function to preprocess the data: reshape the input signal, use
        allowed indices to upsample the class minority labels [active ELM events].

        Args:
        -----
            elm_indices (np.ndarray, optional): ELM event indices for the corresponding
                mode. Defaults to None.
            shuffle_sample_indices (bool, optional): Whether to shuffle the sample
                indices. Defaults to False.

        Returns:
        --------
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing
                original signals, correponding labels, sample indices obtained
                after upsampling and start index for each ELM event.
        """
        packaged_signals = None
        packaged_window_start = None
        packaged_valid_t0 = []
        packaged_labels = []

        # get ELM indices from the data file if not provided
        if elm_indices is None:
            elm_indices = self.elm_indices

        # iterate through all the ELM indices
        with h5py.File(self.datafile, 'r') as hf:
            if save_filename:
                # if self.args.regression:
                #     save_filename += '_regression'
                #     if self.args.regression == 'log':
                #         save_filename += '_log'
                plt.ioff()
                _, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9))
                output_dir = Path(self.args.output_dir)
                save_filename_extended = f"{save_filename}_valid_indices"
                self.logger.info(f"  Saving valid indices as `{save_filename_extended}.pdf`")
                i_page = 1
            for i_elm, elm_index in enumerate(elm_indices):
                if save_filename and i_elm%12==0:
                    for axis in axes.flat:
                        plt.sca(axis)
                        plt.cla()
                elm_key = f"{elm_index:05d}"
                if verbose:
                    self.logger.info(f' ELM index {elm_index}')
                elm_event = hf[elm_key]
                signals = np.array(elm_event["signals"], dtype=np.float32)
                # transposing so that the time dimension comes forward
                signals = np.transpose(signals, (1, 0)).reshape(-1, 8, 8)
                if self.args.automatic_labels:
                    labels = np.array(elm_event["automatic_labels"], dtype=np.float32)
                else:
                    try:
                        labels = np.array(elm_event["labels"], dtype=np.float32)
                    except KeyError:
                        labels = np.array(elm_event["manual_labels"], dtype=np.float32)

                if self.args.normalize_data:
                    signals = signals.reshape(-1, 64)
                    signals[:, :32] = signals[:, :32] / np.max(signals[:, :32])
                    signals[:, 32:] = signals[:, 32:] / np.max(signals[:, 32:])
                    signals = signals.reshape(-1, 8, 8)

                if self.args.truncate_inputs:
                    active_elm_indices = np.where(labels > 0)[0]
                    elm_end_index = active_elm_indices[-1] + self.args.truncate_buffer
                    signals = signals[:elm_end_index, ...]
                    labels = labels[:elm_end_index]
                
                result = self._get_valid_indices(
                    signals=signals,
                    labels=labels,
                    verbose=verbose,
                )
                if result:
                    (signals, labels, valid_t0) = result
                else:
                    continue

                if save_filename:
                    plt.sca(axes.flat[i_elm%12])
                    plt.plot(signals[:,2,3]/10, label='BES 20')
                    plt.plot(signals[:,2,5]/10, label='BES 22')
                    plt.plot(labels, label='Label')
                    _valid_t0 = np.array(valid_t0, copy=True, dtype=float)
                    _valid_tstop = np.zeros(labels.size)
                    _valid_label = np.zeros(labels.size)
                    valid_t0_indices = np.nonzero(valid_t0)[0]
                    valid_tstop_indices = valid_t0_indices + self.args.signal_window_size - 1
                    valid_label_indices = valid_tstop_indices + self.args.label_look_ahead
                    assert _valid_t0[0] == 1
                    assert valid_tstop_indices[-1] <= labels.size-1
                    assert valid_label_indices[-1] <= labels.size-1
                    _valid_t0[valid_t0_indices] = 0.1
                    _valid_tstop[valid_tstop_indices] = 0.2
                    _valid_label[valid_label_indices] = 0.3
                    _valid_t0[_valid_t0 == 0] = np.nan
                    _valid_tstop[_valid_tstop == 0] = np.nan
                    _valid_label[_valid_label == 0] = np.nan
                    plt.plot(_valid_t0, label='Valid t0')
                    plt.plot(_valid_tstop, label='Valid tstop')
                    plt.plot(_valid_label, label='Valid label')
                    plt.title(f"ELM index {elm_key}")
                    plt.legend(fontsize='x-small')
                    plt.xlabel('Time (mu-s)')
                    if i_elm%12==11 or i_elm==elm_indices.size-1:
                        plt.tight_layout()
                        filename = output_dir / f"{save_filename_extended}_{i_page:02d}.pdf"
                        plt.savefig(filename.as_posix(), format="pdf", transparent=True)
                        i_page += 1

                if packaged_signals is None:
                    packaged_window_start = np.array([0])
                    packaged_valid_t0 = valid_t0
                    packaged_signals = signals
                    packaged_labels = labels
                else:
                    last_index = packaged_labels.size - 1
                    packaged_window_start = np.append(
                        packaged_window_start, 
                        last_index + 1
                    )
                    packaged_valid_t0 = np.concatenate([packaged_valid_t0, valid_t0])
                    packaged_signals = np.concatenate([packaged_signals, signals], axis=0)
                    packaged_labels = np.concatenate([packaged_labels, labels], axis=0)                

        if save_filename:
            plt.close()
            pdf_files = sorted(output_dir.glob(f'{save_filename_extended}_*.pdf'))
            output = output_dir / f'{save_filename_extended}.pdf'
            utils.merge_pdfs(pdf_files, output, delete_inputs=True)

        # valid indices for data sampling
        packaged_valid_t0_indices = np.arange(packaged_valid_t0.size, dtype="int")
        packaged_valid_t0_indices = packaged_valid_t0_indices[packaged_valid_t0 == 1]

        packaged_label_indices_for_valid_t0 = (
            packaged_valid_t0_indices 
            + (self.args.signal_window_size-1)
            + self.args.label_look_ahead
            )
        packaged_labels_for_valid_t0 = packaged_labels[packaged_label_indices_for_valid_t0]
        n_active_elm = np.count_nonzero(packaged_labels_for_valid_t0)
        n_inactive_elm = np.count_nonzero(packaged_labels_for_valid_t0-1)
        active_elm_fraction = n_active_elm/(n_active_elm+n_inactive_elm)
        self.logger.info(" Dataset summary")
        self.logger.info(f"  Count of inactive ELM labels: {n_inactive_elm}")
        self.logger.info(f"  Count of active ELM labels: {n_active_elm}")
        self.logger.info(f"  % active: {active_elm_fraction*1e2:.1f} %")

        min_active_elm_fraction = 0.2
        if oversample_active_elm and active_elm_fraction < min_active_elm_fraction:
            oversample_factor = int(min_active_elm_fraction * n_inactive_elm / (n_active_elm*(1-min_active_elm_fraction)))+1
            self.logger.info(f"  Oversample active ELM factor: {oversample_factor}")
            assert oversample_factor >= 1
            packaged_active_elm_label_indices_for_valid_t0 = packaged_label_indices_for_valid_t0[
                packaged_labels[packaged_label_indices_for_valid_t0] == 1
            ]
            packaged_active_elm_valid_t0_indices = (
                packaged_active_elm_label_indices_for_valid_t0
                - (self.args.signal_window_size-1)
                - self.args.label_look_ahead
            )
            for i in np.arange(oversample_factor-1):
                packaged_valid_t0_indices = np.append(
                    packaged_valid_t0_indices,
                    packaged_active_elm_valid_t0_indices,
                )
            packaged_label_indices_for_valid_t0 = (
                packaged_valid_t0_indices
                + (self.args.signal_window_size-1)
                + self.args.label_look_ahead
                )
            packaged_labels_for_valid_t0 = packaged_labels[ packaged_label_indices_for_valid_t0 ]
            n_active_elm = np.count_nonzero(packaged_labels_for_valid_t0)
            n_inactive_elm = np.count_nonzero(packaged_labels_for_valid_t0-1)
            active_elm_fraction = n_active_elm/(n_active_elm+n_inactive_elm)
            self.logger.info(f"  New count of inactive ELM labels: {n_inactive_elm}")
            self.logger.info(f"  New count of active ELM labels: {n_active_elm}")
            self.logger.info(f"  New % active: {active_elm_fraction*1e2:.1f} %")

        if shuffle_sample_indices:
            np.random.shuffle(packaged_valid_t0_indices)

        self.logger.info( "Data tensors -> signals, labels, sample_indices, window_start_indices:")
        for tensor in [
            packaged_signals,
            packaged_labels,
            packaged_valid_t0_indices,
            packaged_window_start,
        ]:
            tmp = f" shape {tensor.shape}, dtype {tensor.dtype},"
            tmp += f" min {np.min(tensor):.3f}, max {np.max(tensor):.3f}"
            if hasattr(tensor, "device"):
                tmp += f" device {tensor.device[-5:]}"
            self.logger.info(tmp)

        return (packaged_signals, packaged_labels, packaged_valid_t0_indices, packaged_window_start, elm_indices)


if __name__=="__main__":
    arg_list = [
        '--use_all_data', 
        '--label_look_ahead', '400',
    ]
    args = TrainArguments().parse(arg_list=arg_list)
    LOGGER = utils.get_logger(script_name=__name__)
    data_cls = utils.create_data_class(args.data_preproc)
    data_obj = data_cls(args, LOGGER)
    elm_indices, all_data = data_obj.get_data(verbose=True)
