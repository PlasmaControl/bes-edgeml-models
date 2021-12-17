import argparse
import logging

import torch
import numpy as np
import os

from visualization import Visualizations
from src.utils import get_logger, create_output_paths
from options.test_arguments import TestArguments


def ELBOLoss(reconstruction, x, mu, logvar):
    """ELBO assuming entries of x are binary variables, with closed form KLD."""

    bce = torch.nn.BCELoss(reconstruction, x.view_as(reconstruction), reduction='none')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= x.view(-1).data.shape[0] * input_size

    return bce + KLD


def train(args: argparse.Namespace, logger: logging.Logger):
    test_dataset = Visualizations(args=args, logger=logger).test_set
    return


if __name__ == '__main__':
    args, parser = TestArguments().parse(verbose=True)
    LOGGER = get_logger(script_name=__name__, log_file=os.path.join(args.log_dir,
                                                                    f" output_logs_{args.model_name}_{args.filename_suffix}.log", ), )
    train(args, LOGGER)
