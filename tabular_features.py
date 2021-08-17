import os

import numpy as np
import pandas as pd

from data_preprocessing import *
from options.base_arguments import BaseArguments
from src import utils

if __name__ == "__main__":
    args, parser = BaseArguments().parse(verbose=True)
    utils.test_args_compat(args, parser)
    LOGGER = utils.get_logger(script_name=__name__)
    data_cls = utils.create_data(args.data_preproc)
    data_obj = data_cls(args, LOGGER)
    print(data_obj)
