"""Various utility functions used for data preprocessing, training and validation.
"""
import os
import logging
import time
import math
import argparse
import importlib
from typing import Union, Tuple
from pathlib import Path

import torch
from torchinfo import summary


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
        log_file = Path(log_file).resolve()
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


def as_minutes_seconds(s: float) -> str:
    m = math.floor(s / 60)
    s -= m * 60
    m, s = int(m), int(s)
    return f"{m:2d}m {s:2d}s"


def time_since(since: int, percent: float) -> str:
    """Helper function to time the training and evaluation process"""
    now = time.time()
    elapsed = now - since
    total_estimated = elapsed / percent
    remaining = total_estimated - elapsed
    return f"{as_minutes_seconds(elapsed)} (remain {as_minutes_seconds(remaining)})"


def create_data(data_name: str) -> object:
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
    infer_mode: bool = False
) -> Tuple[str]:
    """
    Helper function to create various output paths to save model checkpoints,
    test data, plots, etc.

    Args:
        args (argparse.Namespace): Argparse object containing command line args.
        infer_mode (bool): If true, return file output paths for inference as well.

    Returns:
        Tuple containing output paths.
    """
    test_data_file = (args.output_dir / args.metrics_file).as_posix()
    model_ckpt_file = (args.output_dir / args.model_ckpts).as_posix()

    if infer_mode:
        clf_report_dir = os.path.join(args.output_dir, "classification_reports")
        plot_dir = os.path.join(args.output_dir, "plots")
        roc_dir = os.path.join(args.output_dir, "roc")
        for p in [clf_report_dir, plot_dir, roc_dir]:
            os.makedirs(p, exist_ok=True)
        output = (
            test_data_file,
            model_ckpt_file,
            clf_report_dir,
            plot_dir,
            roc_dir,
        )
    else:
        output = (test_data_file, model_ckpt_file)
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
    print(f"Output size: {model(x).shape}")
    print(f"Model contains {get_params(model)} trainable parameters!")


def create_model(model_name: str) -> object:
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
