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


class ELM_Regression_Trainer(_Trainer):

    # __init__ must have exact copy of all kwargs from parent class
    def __init__(
        self,
        # subclass parameters
        max_elms: int = None,  # limit ELMs
        log_time: bool = False,  # if True, use log(time_to_elm_onset)
        inverse_weight_label: bool = False,  # must be False if log_time is False
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

        self.regression = True
        self._set_regression_or_classification_defaults()

        # subclass attributes
        self.max_elms = max_elms
        self.log_time = log_time
        self.inverse_weight_label = inverse_weight_label

        self._finish_subclass_initialization()

    def _get_valid_indices(
        self,
        labels: np.ndarray = None,
        signals: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # indices for active elm times in each elm event
        active_elm_indices = np.nonzero(labels == 1)[0]
        active_elm_start_index = active_elm_indices[0]
        # concat on axis 0 (time dimension)
        valid_t0 = np.ones(active_elm_start_index-1, dtype=np.int32)
        valid_t0[-self.signal_window_size + 1:] = 0
        labels = np.arange(active_elm_start_index, 1, -1, dtype=float)
        signals = signals[:active_elm_start_index-1, :, :]
        if self.log_time:
            if np.any(labels == 0):
                assert False
            labels = np.log10(labels, dtype=float)
        return labels, signals, valid_t0

    def _check_for_balanced_data(self, *args, **kwargs):
        pass


if __name__=='__main__':
    model = ELM_Regression_Trainer(
        batch_size=32, 
        minibatch_interval=50, 
        log_time=True,
        inverse_weight_label=True,
        fraction_validation=0.4,
        fraction_test=0.0,
    )
    model.train()