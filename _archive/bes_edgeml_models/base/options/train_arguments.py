import sys
import argparse
from .base_arguments import BaseArguments


class TrainArguments(BaseArguments):
    """Command line arguments specifically used for training. This inherits the
    BaseArguments for all other common arguments.
    """

    def create_parser(self, parser: argparse.ArgumentParser):
        parser = BaseArguments.create_parser(self, parser)
        # training parameters
        parser.add_argument(
            "--focal_loss",
            action="store_true",
            default=False,
            help="if true, use focal loss instead of `BCELossWithLogits`.",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="learning rate.",
        )
        parser.add_argument(
            "--hidden_size",
            type=int,
            required=("rnn" or "lstm_ae") in sys.argv,
            help="hidden size of the RNN. Must be used "
            "if rnn is used as a Model or for data_preprocessing.",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.005,
            help="L2 regularization factor.",
        )
        # training progress display parameters
        parser.add_argument(
            "--train_print_every",
            type=int,
            default=5000,
            help="frequency of showing training results on screen.",
        )
        parser.add_argument(
            "--valid_print_every",
            type=int,
            default=2000,
            help="frequency of showing validation results on screen.",
        )
        parser.add_argument(
                "--save_onnx",
                action="store_true",
                default=False,
                help="Save the trained model in onnx format."
        )
        self.is_train = True

        return parser
