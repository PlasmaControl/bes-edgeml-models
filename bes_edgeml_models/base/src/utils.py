"""Various utility functions used for data preprocessing, training and validation.
"""
import shutil
import subprocess
import pickle
import sys
import logging
import time
import math
import argparse
import importlib
from typing import Union, Tuple, Sequence, Callable
from pathlib import Path
from traceback import print_tb
import inspect

import numpy as np
import torch
from torchinfo import summary

try:
    from .. import package_dir
    from . import dataset
except ImportError:
    from bes_edgeml_models import package_dir
    from bes_edgeml_models.base.src import dataset


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

    def __init__(self, script_name: str = None, log_file: str = None, stream_handler: bool = True,
                 log_exceptions: bool = True):

        self.logger = None
        self.script_name = script_name
        self.log_file = log_file
        self.stream_handler = stream_handler
        self.log_exceptions = log_exceptions

        global logging_script_name
        logging_script_name = script_name

        if not (stream_handler or self.log_file):
            raise TypeError('Logger must have Handler')

    def create_logger(self):

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

    def getGlobalLogger(self):
        logger = logging.getLogger(__name__)
        if not logger:
            raise AttributeError('No logger exists. Logger must be declared.')
        elif not logger.hasHandlers():
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
        f_handler = logging.FileHandler(log_file.as_posix(), mode="a")
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


def create_data_class(data_name: str) -> Callable:
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
        package='bes_edgeml_models.base.src',
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

    output_dir = Path(args.output_dir).resolve()

    test_data_file = output_dir / args.test_data_file
    checkpoint_file = output_dir / args.checkpoint_file

    # if args.regression:
    #     addon = "_regression"
    #     if args.regression == "log":
    #         addon += "_log"
    #     checkpoint_file = checkpoint_file.parent / (checkpoint_file.stem + addon + checkpoint_file.suffix)
    #     test_data_file = test_data_file.parent / (test_data_file.stem + addon + test_data_file.suffix)

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


def get_params(model: torch.nn.Module) -> int:
    """Helper function to find the total number of trainable parameters in the
    PyTorch model.

    Args:
        model (object): Instance of the PyTorch model being used.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_details(model: torch.nn.Module, x: torch.Tensor, input_size: tuple) -> None:
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


def merge_pdfs(
    inputs: Sequence,
    output: Union[str,Path],
    delete_inputs: bool = False,
):
    inputs = [Path(input) for input in inputs]
    output = Path(output)
    gs_cmd = shutil.which('gs')
    if gs_cmd is None:
        return
    print(f"Merging PDFs into file: {output.as_posix()}")
    cmd = [
        gs_cmd,
        '-q',
        '-dBATCH',
        '-dNOPAUSE',
        '-sDEVICE=pdfwrite',
        '-dPDFSETTINGS=/prepress',
        '-dCompatibilityLevel=1.4',
    ]
    cmd.append(f"-sOutputFile={output.as_posix()}")
    for pdf_file in inputs:
        cmd.append(f"{pdf_file.as_posix()}")
    result = subprocess.run(cmd, check=True)
    assert result.returncode == 0 and output.exists()
    if delete_inputs is True:
        for pdf_file in inputs:
            pdf_file.unlink()


def create_model_class(
        model_name: str,
        caller = None
    ) -> torch.nn.Module:
    """
    Helper function to import the module for the model being used as per the
    command line argument `--model_name`.

    Args:
        model_name (str): `--model_name` argument.
        caller (path | str): which file the function is called from.

    Returns:
        Object of the model class.
    """
    if caller is None:
        model_type = str(Path(inspect.stack()[1].filename).parent.stem)
    else:
        model_type = str(caller)
    model_filename = model_name + "_model"
    model_path = ".models." + model_filename
    model_lib = importlib.import_module(
        model_path,
        package='bes_edgeml_models.' + model_type,
    )
    model = None
    _model_name = model_name.replace("_", "") + "model"
    for name, cls in model_lib.__dict__.items():
        if name.lower() == _model_name.lower():
            model = cls
            model.model_type = model_type  # assigning a type to a model makes it easier to share scripts between them.

    return model
