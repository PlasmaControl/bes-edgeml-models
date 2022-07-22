from typing import Tuple
import numpy as np

try:
    from ..main.train_base import _Trainer
except ImportError:
    from bes_ml.main.train_base import _Trainer


class ELM_Regression_Trainer(_Trainer):

    def __init__(
        self,
        max_elms: int = None,  # limit ELMs
        log_time: bool = False,  # if True, use log(time_to_elm_onset)
        inverse_weight_label: bool = False,  # must be False if log_time is False
        **kwargs,
    ) -> None:

        # init parent class
        super().__init__(**kwargs)

        self._print_kwargs(self.__class__, locals().copy())

        self.regression = True
        self._set_regression_or_classification_defaults()

        # subclass attributes
        self.max_elms = max_elms
        self.log_time = log_time
        self.inverse_weight_label = inverse_weight_label

        self._finish_initialization()

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