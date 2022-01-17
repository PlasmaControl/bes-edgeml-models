import os
import logging
import time
import math
import argparse
import importlib
from typing import Union, Tuple
from traceback import print_tb

import torch
from torchinfo import summary
from torchviz import make_dot


class MetricMonitor:
    """Calculates and stores the average value of the metrics/loss"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all the parameters to zero."""
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: float = 0

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
def make_logger(script_name: str, log_file: Union[str, None] = None,
                stream_handler: bool = True, ) -> logging.getLogger:
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
    logger = logging.getLogger(name=script_name)
    logger.setLevel(logging.INFO)

    if log_file is not None:
        # create handlers
        f_handler = logging.FileHandler(os.path.join(log_file), mode="a")
        # create formatters and add it to the handlers
        f_format = logging.Formatter("%(asctime)s:%(name)s %(levelname)s:%(message)s")
        f_handler.setFormatter(f_format)
        # add handlers to the logger
        logger.addHandler(f_handler)

    # display the logs in console
    if stream_handler:
        s_handler = logging.StreamHandler()
        s_format = logging.Formatter("%(filename)s->%(funcName)s: %(levelname)s:%(message)s")
        s_handler.setFormatter(s_format)
        logger.addHandler(s_handler)

    return logger


def log_exceptions(logger):
    def my_handler(type, value, tb):
        print_tb(tb)
        logger.exception(f' {type.__name__}: {value}')

    return my_handler


def as_minutes_seconds(s: int) -> str:
    m = math.floor(s / 60)
    s -= m * 60
    m, s = int(m), int(s)
    return f"{m:2d}m {s:2d}s"


def time_since(since: int, percent: float) -> str:
    now = time.time()
    elapsed = now - since
    total_estimated = elapsed / percent
    remaining = total_estimated - elapsed
    return f"{as_minutes_seconds(elapsed)} (remain {as_minutes_seconds(remaining)})"


def test_args_compat(args: argparse.Namespace, parser: argparse.ArgumentParser, infer_mode: bool = False, ):
    """Checks if all the parameters with dependencies are passed."""
    compat = True
    # check the inference related parameters
    if infer_mode:
        if ((args.plot_num == 12) and (args.num_rows != 4) and (args.num_cols != 3)) or (
                (args.plot_num == 6) and (args.num_rows != 3) and (args.num_cols != 2)):
            parser.error(f"number of rows: {args.num_rows} and number of columns: {args.num_cols} "
                         f"are not compatible with total number of plots: {args.plot_num}")
            compat = False
    else:
        raise ValueError("Function only required for inference.")
    if compat:
        print("All the parsed parameters are compatible with each other!")


def create_data(data_name: str):
    data_filename = data_name + "_data"
    data_class_path = "data_preprocessing." + data_filename
    data_lib = importlib.import_module(data_class_path)
    data_class = None
    _data_name = data_name.replace("_", "") + "data"
    for name, cls in data_lib.__dict__.items():
        if name.lower() == _data_name.lower():
            data_class = cls

    return data_class


def create_output_paths(args: argparse.Namespace, infer_mode: bool = False) -> Tuple[str]:
    if args.signal_window_size == 8:
        test_data_path = os.path.join(args.test_data_dir, "signal_window_8")
        model_ckpt_path = os.path.join(args.model_ckpts, "signal_window_8")
    elif args.signal_window_size == 16:
        test_data_path = os.path.join(args.test_data_dir, "signal_window_16")
        model_ckpt_path = os.path.join(args.model_ckpts, "signal_window_16")
    else:
        test_data_path = os.path.join(args.test_data_dir, f"signal_window_{args.signal_window_size}")
        model_ckpt_path = os.path.join(args.model_ckpts, f"signal_window_{args.signal_window_size}")
        if not os.path.exists(test_data_path):
            os.makedirs(test_data_path, exist_ok=True)
        if not os.path.exists(model_ckpt_path):
            os.makedirs(model_ckpt_path, exist_ok=True)
    if infer_mode:
        if args.signal_window_size == 8:
            base_path = os.path.join(args.output_dir, "signal_window_8")
            look_ahead_path = os.path.join(base_path, f"label_look_ahead_{args.label_look_ahead}")
            clf_report_path = os.path.join(look_ahead_path, "classification_reports")
            plot_path = os.path.join(look_ahead_path, "plots")
            roc_path = os.path.join(look_ahead_path, "roc")
            paths = [clf_report_path, plot_path, roc_path]
            for p in paths:
                if not os.path.exists(p):
                    os.makedirs(p, exist_ok=True)
        elif args.signal_window_size == 16:
            base_path = os.path.join(args.output_dir, "signal_window_16")
            look_ahead_path = os.path.join(base_path, f"label_look_ahead_{args.label_look_ahead}")
            clf_report_path = os.path.join(look_ahead_path, "classification_reports")
            plot_path = os.path.join(look_ahead_path, "plots")
            roc_path = os.path.join(look_ahead_path, "roc")
            paths = [clf_report_path, plot_path, roc_path]
            for p in paths:
                if not os.path.exists(p):
                    os.makedirs(p, exist_ok=True)
        else:
            base_path = os.path.join(args.output_dir, f"signal_window_{args.signal_window_size}", )
            look_ahead_path = os.path.join(base_path, f"label_look_ahead_{args.label_look_ahead}")
            clf_report_path = os.path.join(look_ahead_path, "classification_reports")
            plot_path = os.path.join(look_ahead_path, "plots")
            roc_path = os.path.join(look_ahead_path, "roc")
            paths = [clf_report_path, plot_path, roc_path]
            for p in paths:
                if not os.path.exists(p):
                    os.makedirs(p, exist_ok=True)
        return (test_data_path, model_ckpt_path, clf_report_path, plot_path, roc_path,)
    return test_data_path, model_ckpt_path


def get_params(model: object) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_details(model: object, x: torch.Tensor, input_size: tuple) -> None:
    print("\t\t\t\tMODEL SUMMARY")
    summary(model, input_size=input_size)
    print(f"Output size: {model(x)[-1].shape}")
    print(f"Model contains {get_params(model)} trainable parameters!")


def create_model(model_name: str):
    model_filename = model_name + "_model"
    model_path = "models." + model_filename
    model_lib = importlib.import_module(model_path)
    model = None
    _model_name = model_name.replace("_", "") + "model"
    for name, cls in model_lib.__dict__.items():
        if name.lower() == _model_name.lower():
            model = cls

    return model


def model_viz(model: object, x: torch.Tensor, show_autograd: bool = False):
    dot = make_dot(model(x), params=dict(model.named_parameters()), show_attrs=show_autograd, show_saved=show_autograd)
    dot.render(f'visualizations/torchviz_diagrams/{model.args.model_name}{"_w_autograd" if show_autograd else ""}',
               format='png', view=True)


def get_lr_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader, ):
    # learning rate scheduler
    if args.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=args.lr_plateau_mode,
                factor=args.decay_factor, patience=args.patience, verbose=True, )
    elif args.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min,
                verbose=False)
    elif args.scheduler == "CyclicLR":
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.base_lr, max_lr=args.max_lr,
                mode=args.cyclic_mode, cycle_momentum=args.cycle_momentum, verbose=False, )
    elif args.scheduler == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=args.n_epochs,
                steps_per_epoch=len(dataloader), max_lr=args.max_lr, pct_start=args.pct_start,
                anneal_strategy=args.anneal_policy, )
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_factor, verbose=True)

    return scheduler


def get_optimizer(args: argparse.ArgumentParser, model: object) -> torch.optim.Optimizer:
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer
