"""Various utility functions used for data preprocessing, training and validation.
"""

from genericpath import exists
import os
import pickle
import sys
import logging
import time
import math
import argparse
import importlib
from collections import OrderedDict
from typing import Union, Tuple
from pathlib import Path
from traceback import print_tb

import numpy as np
import torch
from torchinfo import summary

from elm_prediction import package_dir


class MetricMonitor:
    """Calculates and stores the average value of the metrics/loss"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all the parameters to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1):
        """Update the value of the metrics and calculate their
        average value over the whole dataset.
        Args:
        -----
            val (float): Computed metric (per batch)
            n (int, optional): Batch size. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class logParse:
    """Initiate the logger to log the progress into a file.

    Args:
    -----
        script_name (str): Name of the scripts outputting the logs.
        log_file (str): Name of the log file.
        stream_handler (bool, optional): Whether or not to show logs in the
            console. Defaults to True.

    Returns:
    --------
        logging.getLogger: Logger object.
    """

    def __init__(self, script_name: str = None, args: argparse.Namespace = None, stream_handler: bool = True,
                 log_exceptions: bool = True):

        self.logger = None
        self.script_name = script_name
        self.log_file = os.path.join(package_dir, 'logs', f'{args.model_name}.log')
        self.stream_handler = stream_handler
        self.log_exceptions = log_exceptions

        if not (stream_handler and self.log_file):
            raise TypeError('Logger must have Handler')

    def __call__(self):

        logger = logging.getLogger(name=self.script_name)
        logger.setLevel(logging.INFO)

        if self.log_file is not None:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)  # make dir. for log file
            # create handlers
            f_handler = logging.FileHandler(log_path.as_posix(), mode="w")
            # create formatters and add it to the handlers
            f_format = logging.Formatter("%(asctime)s:%(name)s: %(levelname)s:%(message)s")
            f_handler.setFormatter(f_format)
            # add handlers to the logger
            logger.addHandler(f_handler)

        # display the logs in console
        if self.stream_handler:
            s_handler = logging.StreamHandler()
            s_format = logging.Formatter("%(name)s: %(levelname)s:%(message)s")
            s_handler.setFormatter(s_format)
            logger.addHandler(s_handler)

        self.logger = logger

        if self.log_exceptions:
            sys.excepthook = self.log_exceptions_()

        return logger

    def log_exceptions_(self):
        def my_handler(type, value, tb):
            print_tb(tb)
            self.logger.exception(f' {type.__name__}: {value}')

        return my_handler

    @staticmethod
    def getGlobalLogger():
        logger = logging.getLogger('__main__')
        if not logger.hasHandlers():
            raise AttributeError('No logger exists. Logger must be declared.')
        return logger


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

# log the model and data preprocessing outputs
def get_logger(
    script_name: Union[str, None] = None,
    log_file: Union[str, Path, None] = 'output.log',
    stream_handler: bool = True,
) -> logging.getLogger:
    """Initiate the logger to log the progress into a file.

    Args:
    -----
        script_name (str): Name of the scripts outputting the logs.
        log_file (str): Name of the log file.
        stream_handler (bool, optional): If true, show logs in the console. Defaults to True.

    Returns:
    --------
        logging.getLogger: Logger object.
    """
    logger = logging.getLogger(name=script_name)
    logger.setLevel(logging.INFO)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)  # make dir. for log file
        # create handlers
        f_handler = logging.FileHandler(log_file.as_posix(), mode="w")
        # create formatters and add it to the handlers
        f_format = logging.Formatter("%(asctime)s:%(name)s: %(levelname)s:%(message)s")
        f_handler.setFormatter(f_format)
        # add handlers to the logger
        logger.addHandler(f_handler)

    # display the logs in console
    if stream_handler:
        s_handler = logging.StreamHandler()
        s_format = logging.Formatter("%(name)s: %(levelname)s:%(message)s")
        s_handler.setFormatter(s_format)
        logger.addHandler(s_handler)

    return logger




def time_since(since: int, percent: float) -> str:
    """Helper function to time the training and evaluation process"""
    def as_minutes_seconds(s: float) -> str:
        m = math.floor(s / 60)
        s -= m * 60
        m, s = int(m), int(s)
        return f"{m:2d}m {s:2d}s"
    now = time.time()
    elapsed = now - since
    total_estimated = elapsed / percent
    remaining = total_estimated - elapsed
    return f"{as_minutes_seconds(elapsed)} (remain {as_minutes_seconds(remaining)})"


def create_data_class(data_name: str) -> object:
    """
    Helper function to import the data preprocessing module as per the command
    line argument `--data_preproc`.

    Args:
        data_name (str): `--data_preproc` argument.

    Returns:
        Object of the data class.
    """
    data_filename = data_name + "_data"
    data_class_path = "..data_preprocessing." + data_filename
    data_lib = importlib.import_module(
        data_class_path,
        package='elm_prediction.src',
    )
    data_class = None
    _data_name = data_name.replace("_", "") + "data"
    for name, cls in data_lib.__dict__.items():
        if name.lower() == _data_name.lower():
            data_class = cls

    return data_class


def create_output_paths(
    args: argparse.Namespace, 
    infer_mode: bool = False,
) -> Union[Tuple[Path,Path],
           Tuple[Path,Path,Path,Path,Path]]:
    """
    Helper function to create various output paths to save model checkpoints,
    test data, plots, etc.

    Args:
        args (argparse.Namespace): Argparse object containing command line args.
        infer_mode (bool): If true, return file output paths for inference as well.

    Returns:
        Tuple containing output paths.
    """

    output_dir = Path(args.output_dir)

    test_data_file = output_dir / args.test_data_file
    checkpoint_file = output_dir / args.checkpoint_file

    if infer_mode:
        clf_report_dir = output_dir / "classification_reports"
        plot_dir = output_dir / "plots"
        roc_dir = output_dir / "roc"
        for p in [clf_report_dir, plot_dir, roc_dir]:
            p.mkdir(exist_ok=True)
        output = (
            test_data_file,
            checkpoint_file,
            clf_report_dir,
            plot_dir,
            roc_dir,
        )
    else:
        output = (test_data_file, checkpoint_file)
    return output


def get_params(model: object) -> int:
    """Helper function to find the total number of trainable parameters in the
    PyTorch model.

    Args:
        model (object): Instance of the PyTorch model being used.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_details(model: object, x: torch.Tensor, input_size: tuple) -> None:
    """
    Print Keras like model details on the screen before training.

    Args:
        model (object): Instance of the PyTorch model being used.
        x (torch.Tensor): Dummy input.
        input_size (tuple): Size of the input.

    Returns:
        None
    """
    print("\t\t\t\tMODEL SUMMARY")
    summary(model, input_size=input_size)
    print(f'Batched input size: {x.shape}')
    print(f"Batched output size: {model(x).shape}")
    print(f"Model contains {get_params(model)} trainable parameters!")


def create_model_class(
        model_name: str
    ) -> torch.nn.Module:
    """
    Helper function to import the module for the model being used as per the
    command line argument `--model_name`.

    Args:
        model_name (str): `--model_name` argument.

    Returns:
        Object of the model class.
    """
    model_filename = model_name + "_model"
    model_path = "..models." + model_filename
    model_lib = importlib.import_module(
        model_path,
        package='elm_prediction.src',
    )
    model = None
    _model_name = model_name.replace("_", "") + "model"
    for name, cls in model_lib.__dict__.items():
        if name.lower() == _model_name.lower():
            model = cls

    return model

def get_model(args: argparse.Namespace,
              logger: logging.Logger):
    _, model_cpt_path = src.utils.create_output_paths(args)
    gen_type_suffix = '_' + re.split('[_.]', args.input_file)[-2] if args.generated else ''
    model_name = args.model_name + gen_type_suffix
    accepted_preproc = ['wavelet', 'unprocessed']

    model_cpt_file = os.path.join(model_cpt_path, f'{args.model_name}_lookahead_{args.label_look_ahead}'
                                                  f'{gen_type_suffix}'
                                                  f'{"_" + args.data_preproc if args.data_preproc in accepted_preproc else ""}'
                                                  f'{"_" + args.balance_data if args.balance_data else ""}.pth')

    raw_model = (multi_features_model.RawFeatureModel(args) if args.raw_num_filters > 0 else None)
    fft_model = (multi_features_model.FFTFeatureModel(args) if args.fft_num_filters > 0 else None)
    cwt_model = (multi_features_model.CWTFeatureModel(args) if args.wt_num_filters > 0 else None)
    features = [type(f).__name__ for f in [raw_model, fft_model, cwt_model] if f]

    logger.info(f'Found {model_name} state dict at {model_cpt_file}.')
    model_cls = src.utils.create_model(args.model_name)
    if 'MULTI' in args.model_name.upper():
        model = model_cls(args, raw_model, fft_model, cwt_model)
    else:
        model = model_cls(args)
    state_dict = torch.load(model_cpt_file, map_location=torch.device(args.device))['model']
    model.load_state_dict(state_dict)
    logger.info(f'Loaded {model_name} state dict.')

    model.layers = OrderedDict([child for child in model.named_modules() if hasattr(child[1], 'weight')])

    return model.to(args.device)