import os
import logging
import time
import math
import argparse

from . import config


# log the model and data preprocessing outputs
def get_logger(
    script_name: str, log_file: str, stream_handler: bool = True
) -> logging.getLogger:
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

    # create handlers
    f_handler = logging.FileHandler(
        os.path.join(config.output_dir, log_file), mode="w"
    )

    # create formatters and add it to the handlers
    f_format = logging.Formatter(
        "%(asctime)s:%(name)s: %(levelname)s:%(message)s"
    )
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


def test_args_compat(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    infer_mode: bool = False,
):
    """Checks if all the parameters with dependencies are passed."""
    compat = True
    # check the basic arguments and their dependencies
    if (
        args.model_name == "StackedELMModel"
        and (not args.stack_elm_events)
        and ("size" in vars(args))
    ):
        parser.error(
            f"{args.model_name} requires arguments `size` and `stack_elm_events` set to True."
        )
        compat = False

    if "smoothen_transition" in vars(
        args
    ) and "transition_halfwidth" not in vars(args):
        parser.error(
            "`smoothen_transition` argument requires argument `transition_halfwidth`."
        )
        compat = False
    if (
        (args.add_noise)
        and ("mu" not in vars(args))
        and ("sigma" not in vars(args))
    ):
        parser.error(
            "`add_noise` argument requires arguments `mu` and `sigma`."
        )
        compat = False
    # check the inference related parameters
    if infer_mode:
        if (
            (args.plot_num == 12)
            and (args.num_rows != 4)
            and (args.num_cols != 3)
        ) or (
            (args.plot_num == 6)
            and (args.num_rows != 3)
            and (args.num_cols != 2)
        ):
            parser.error(
                f"number of rows: {args.num_rows} and number of columns: {args.num_cols} "
                f"are not compatible with total number of plots: {args.plot_num}"
            )
            compat = False
    if compat:
        print("All the parsed parameters are compatible with each other!")
