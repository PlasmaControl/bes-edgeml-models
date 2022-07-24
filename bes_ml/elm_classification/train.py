from typing import Tuple, Union
from pathlib import Path
import inspect

import numpy as np

from bes_data.sample_data import sample_elm_data_file
try:
    from ..base.train_base import _Trainer
    from ..base import utilities
except ImportError:
    from bes_ml.base.train_base import _Trainer
    from bes_ml.base import utilities


class ELM_Classification_Trainer(_Trainer):

    # __init__ must have exact copy of all kwargs from parent class
    def __init__(
        self,
        # subclass parameters
        max_elms: int = None,  # limit ELMs
        prediction_horizon: int = 200,  # prediction horizon in time samples
        threshold: float = 0.5,  # threshold for binary classification
        oversample_active_elm: bool = True,  # if True, oversample active ELMs to balance data
        # parent class parameters
        input_data_file: Union[Path,str] = sample_elm_data_file,  # path to data file
        output_dir: Union[Path,str] = 'run_dir',  # path to output dir.
        results_file: str = 'results.yaml',  # output training results
        log_file: str = 'log.txt',  # output log file
        trainer_inputs_file: str = 'trainer_inputs.yaml',  # save inputs to yaml
        model_inputs_file: str = 'model_inputs.yaml',  # save inputs to yaml
        test_data_file: str = 'test_data.pkl',  # if None, do not save test data (can be large)
        checkpoint_file: str = 'checkpoint.pytorch',  # pytorch save file; if None, do not save
        export_onnx: bool = False,  # export ONNX format
        device: str = 'auto',  # auto (default), cpu, cuda, or cuda:X
        num_workers: int = 1,  # number of subprocess workers for pytorch dataloader
        n_epochs: int = 2,  # training epochs
        batch_size: int = 64,  # power of 2, like 16-128
        minibatch_interval: int = 2000,  # print minibatch info
        signal_window_size: int = 128,  # power of 2, like 32-512
        fraction_validation: float = 0.1,  # fraction of dataset for validation
        fraction_test: float = 0.15,  # fraction of dataset for testing
        optimizer_type: str = 'adam',  # adam (default) or sgd
        sgd_momentum: float = 0.0,  # momentum for SGD optimizer
        sgd_dampening: float = 0.0,  # dampening for SGD optimizer
        learning_rate: float = 1e-3,  # optimizer learning rate
        weight_decay: float = 5e-3,  # optimizer L2 regularization factor
        batches_per_print: int = 5000,  # train/validation batches per print update
        # kwargs for model
        **model_kwargs,
    ) -> None:

        # ensure subclass implements all parent class inputs
        self._validate_subclass_signature()

        # __init__ parent class
        parent_class_parameters = inspect.signature(_Trainer).parameters
        locals_copy = locals().copy()
        kwargs_for_parent_class = {
            p_name: locals_copy[p_name] for p_name in parent_class_parameters
        }
        kwargs_for_parent_class.pop('model_kwargs') # remove kwargs dict and replace with items
        kwargs_for_parent_class.update(model_kwargs)
        super().__init__(**kwargs_for_parent_class)

        # save and print inputs
        utilities._print_inputs(cls=self.__class__, locals_copy=locals_copy, logger=self.logger)
        utilities._save_inputs_to_yaml(
            cls=self.__class__, 
            locals_copy=locals_copy,
            filename=self.output_dir/self.trainer_inputs_file,
        )

        self.regression = False
        self._set_regression_or_classification_defaults()

        # subclass attributes
        self.max_elms = max_elms
        self.prediction_horizon = prediction_horizon
        self.threshold = threshold
        self.oversample_active_elm = oversample_active_elm

        self._finish_subclass_initialization()

    def _get_valid_indices(
        self,
        labels: np.ndarray = None,
        signals: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # indices for active elm times in each elm event
        active_elm_indices = np.nonzero(labels == 1)[0]
        active_elm_start_index = active_elm_indices[0]
        # `t0` is first index (or earliest time, or trailing time point) for signal window
        # `valid_t0` denotes valid `t0` time points for signal window
        # initialize to zeros
        valid_t0 = np.zeros(labels.shape, dtype=np.int32)
        # largest `t0` index with signal window in pre-ELM period
        largest_t0_index_for_pre_elm_period = active_elm_start_index - self.signal_window_size
        if largest_t0_index_for_pre_elm_period < 0:
            # insufficient pre-elm period for signal window size
            return None
        assert labels[largest_t0_index_for_pre_elm_period + (self.signal_window_size-1)    ] == 0
        assert labels[largest_t0_index_for_pre_elm_period + (self.signal_window_size-1) + 1] == 1
        # `t0` time points up to `largest_t0` are valid
        valid_t0[0:largest_t0_index_for_pre_elm_period+1] = 1
        assert valid_t0[largest_t0_index_for_pre_elm_period    ] == 1
        assert valid_t0[largest_t0_index_for_pre_elm_period + 1] == 0
        # labels after ELM onset should be active ELM, even if in post-ELM period
        last_label_for_active_elm_in_pre_elm_signal = (
            largest_t0_index_for_pre_elm_period
            + (self.signal_window_size - 1)
            + self.prediction_horizon
        )
        labels[ active_elm_start_index : last_label_for_active_elm_in_pre_elm_signal+1 ] = 1
        assert labels[last_label_for_active_elm_in_pre_elm_signal] == 1
        return labels, signals, valid_t0

    def _check_for_balanced_data(
        self, 
        packaged_labels: np.ndarray = None,
        packaged_valid_t0_indices: np.ndarray = None,
        oversample_active_elm: bool = False,
    ) -> np.ndarray:
        packaged_label_indices_for_valid_t0 = (
            packaged_valid_t0_indices 
            + (self.signal_window_size-1)
            + self.prediction_horizon
            )
        packaged_labels_for_valid_t0 = packaged_labels[packaged_label_indices_for_valid_t0]
        n_active_elm = np.count_nonzero(packaged_labels_for_valid_t0)
        n_inactive_elm = np.count_nonzero(packaged_labels_for_valid_t0-1)
        active_elm_fraction = n_active_elm/(n_active_elm+n_inactive_elm)
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
                - (self.signal_window_size-1)
                - self.prediction_horizon
            )
            for i in np.arange(oversample_factor-1):
                packaged_valid_t0_indices = np.append(
                    packaged_valid_t0_indices,
                    packaged_active_elm_valid_t0_indices,
                )
            packaged_label_indices_for_valid_t0 = (
                packaged_valid_t0_indices
                + (self.signal_window_size-1)
                + self.prediction_horizon
                )
            packaged_labels_for_valid_t0 = packaged_labels[packaged_label_indices_for_valid_t0]
            n_active_elm = np.count_nonzero(packaged_labels_for_valid_t0)
            n_inactive_elm = np.count_nonzero(packaged_labels_for_valid_t0-1)
            active_elm_fraction = n_active_elm/(n_active_elm+n_inactive_elm)
            self.logger.info(f"  New count of inactive ELM labels: {n_inactive_elm}")
            self.logger.info(f"  New count of active ELM labels: {n_active_elm}")
            self.logger.info(f"  New % active: {active_elm_fraction*1e2:.1f} %")
        return packaged_valid_t0_indices


if __name__=='__main__':
    model = ELM_Classification_Trainer(
        batch_size=32, 
        minibatch_interval=50, 
        fraction_validation=0.2,
        fraction_test=0.2,
    )
    model.train()