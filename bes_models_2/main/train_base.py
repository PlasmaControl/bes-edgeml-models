from pathlib import Path
import logging
import inspect
from typing import Union

import torch

try:
    from .. import main
    from .. import sample_data_dir
except ImportError:
    from bes_models_2 import main
    from bes_models_2 import sample_data_dir


default_data_file = sample_data_dir / 'sample_elm_events.hdf5'

class _Trainer(object):

    @classmethod
    def _get_init_kwargs_and_defaults(cls) -> dict:
        init_signature = inspect.signature(cls)
        init_kwargs_and_defaults = {parameter.name: parameter.default 
            for parameter in init_signature.parameters.values()}
        return init_kwargs_and_defaults

    def __init__(
        self,
        input_data_file: Union[str,Path] = default_data_file,  # path to data file
        output_dir: Union[str,Path] = Path('run_dir'),  # path to output dir.
        results_file: str = 'results.pkl',  # output training results
        log_file: str = 'log.txt',  # output log file
        args_file: str = 'args.pkl',  # output file containing kwargs
        test_data_file: str = 'test_data.pkl',  # if None, do not save test data (can be large)
        checkpoint_file: str = 'checkpoint.pytorch',  # pytorch save file; if None, do not save
        export_onnx: bool = False,  # export ONNX format
        device: str = 'auto',  # auto (default), cpu, cuda, or cuda:X
        num_workers: int = 1,  # number of subprocess workers for pytorch dataloader
        n_epochs: int = 2,  # training epochs
        batch_size: int = 64,  # power of 2, like 16-128
        signal_window_size: int = 128,  # power of 2, like 32-512
        fraction_validation: float = 0.1,  # fraction of dataset for validation
        fraction_test: float = 0.15,  # fraction of dataset for testing
        optimizer_type: str = 'adam',  # adam (default) or sgd
        sgd_momentum: float = 0.0,  # momentum for SGD optimizer
        sgd_dampening: float = 0.0,  # dampening for SGD optimizer
        learning_rate: float = 1e-3,  # optimizer learning rate
        weight_decay: float = 5e-3,  # optimizer L2 regularization factor
        batches_per_print: int = 5000,  # train/validation batches per print update
    ) -> None:

        # input data file and output directory
        input_data_file = Path(input_data_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        self.input_data_file = input_data_file
        self.output_dir = output_dir
        self.results_file = results_file
        self.log_file = log_file
        self.args_file = args_file
        self.test_data_file = test_data_file
        self.checkpoint_file = checkpoint_file
        self.export_onnx = export_onnx
        self.device = device
        self.num_workers = num_workers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.signal_window_size = signal_window_size
        self.fraction_validation = fraction_validation
        self.fraction_test = fraction_test
        self.optimizer_type = optimizer_type
        self.sgd_momentum = sgd_momentum
        self.sgd_dampening = sgd_dampening
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batches_per_print = batches_per_print

        # create logger (logs to file and terminal)
        self.logger = None
        self._create_logger()

        self._print_kwargs()
        self._setup_device()

    def _print_kwargs(self):
        # print kwargs for parent class
        self.logger.info(f"Parent class `{_Trainer.__name__}` keyword arguments:")
        parent_kwargs = _Trainer._get_init_kwargs_and_defaults()
        for key, default_value in parent_kwargs.items():
            value = getattr(self, key)
            if value == default_value:
                self.logger.info(f"  {key:22s}:  {value}")
            else:
                self.logger.info(f"  {key:22s}:  {value} (default {default_value})")
        # print kwargs for subclass, if any
        if self.__class__ is not _Trainer:
            self.logger.info(f"Subclass class `{self.__class__.__name__}` keyword arguments:")
            subclass_kwargs = self.__class__._get_init_kwargs_and_defaults()
            for key, default_value in subclass_kwargs.items():
                if key == 'kwargs': continue
                value = getattr(self, key)
                if value == default_value:
                    self.logger.info(f"  {key:22s}:  {value}")
                else:
                    self.logger.info(f"  {key:22s}:  {value} (default {default_value})")

    def _setup_device(self):

        # setup device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        self.logger.info(f"Device: {self.device}")

    def _create_logger(self):
        """
        Use python's logging to allow simultaneous print to log file and console.
        """
        self.logger = logging.getLogger(name=__name__)
        self.logger.setLevel(logging.INFO)

        # logs to log file
        log_file = self.output_dir / self.log_file
        f_handler = logging.FileHandler(log_file.as_posix(), mode="a")
        # create formatters and add it to the handlers
        f_format = logging.Formatter("%(asctime)s:%(name)s: %(levelname)s:  %(message)s")
        f_handler.setFormatter(f_format)
        # add handlers to the logger
        self.logger.addHandler(f_handler)

        # logs to console
        s_handler = logging.StreamHandler()
        # s_format = logging.Formatter("%(name)s:  %(message)s")
        # s_handler.setFormatter(s_format)
        self.logger.addHandler(s_handler)


if __name__=='__main__':
    m = _Trainer()
