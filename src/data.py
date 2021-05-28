"""
Data class to package BES data for training using PyTorch
"""
import os
import sys
import logging
from typing import Tuple, Callable, List

import h5py
import numpy as np
import pandas as pd
from sklearn import model_selection
import torch

# run the code from top level directory
sys.path.append("../model_tools")
from model_tools import utilities, config

# log the model and data preprocessing outputs
def get_logger(stream_handler=True):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create handlers
    f_handler = logging.FileHandler(
        os.path.join(config.output_dir, "output_logs.log")
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


# create the logger object
LOGGER = get_logger()


class Data:
    def __init__(
        self,
        datafile: str = None,
        fraction_validate: float = 0.2,
        fraction_test: float = 0.1,
        signal_dtype: str = "float32",
        kfold: bool = False,
        smoothen_transition: bool = False,
    ):
        """Helper class that takes care of all the data preparation steps: reading
        the HDF5 file, split all the ELM events into training, validation and test
        sets, upsample the data to reduce class imbalance and create a sample signal
        window.

        Args:
        -----
            datafile (str, optional): Path to the input datafile. Defaults to None.
            fraction_validate (float, optional): Fraction of the total data to
                be used as a validation set. Ignored when using K-fold cross-
                validation. Defaults to 0.2.
            fraction_test (float, optional): Fraction of the total data to be
                used as test set. Defaults to 0.1.
            signal_dtype (str, optional): Datatype of the signals. Defaults to "float32".
            kfold (bool, optional): Boolean showing whether to use K-fold cross-
                validation or not. Defaults to False.
            smoothen_transition (bool, optional): Boolean showing whether to smooth
                the labels so that there is a gradual transition of the labels from
                0 to 1 with respect to the input time series. Defaults to False.
        """
        self.datafile = datafile
        if self.datafile is None:
            self.datafile = os.path.join(utilities.data_dir, config.file_name)
        self.fraction_validate = fraction_validate
        self.fraction_test = fraction_test
        self.signal_dtype = signal_dtype
        self.kfold = kfold
        self.smoothen_transition = smoothen_transition
        self.max_elms = config.max_elms

        self.transition = np.linspace(0, 1, 2 * config.transition_halfwidth + 3)
