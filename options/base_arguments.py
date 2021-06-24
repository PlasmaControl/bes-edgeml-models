import os
import argparse
import torch


class BaseArguments:
    """This class defines command line arguments common for both training and testing.
    It also implements some helper functions related to parsing and printing these
    command line options.
    """

    def __init__(self):
        """Reset the class; used to further initialize the class."""
        self.initialized = False

    def create_parser(self, parser: argparse.ArgumentParser):
        """Define the options common for training and testing."""
        # basic parameters
        parser.add_argument(
            "--input_file", required=True, help="path to the input hdf5 file."
        )
        parser.add_argument(
            "--model_name",
            type=str,
            required=True,
            help="name of the model to be used for training, "
            "[FeatureModel | CNNModel | ...].",
        )
        parser.add_argument(
            "--n_epochs",
            type=int,
            default=10,
            help="total number of epochs for training.",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cpu",
            help="device to be used for training and testing, [cpu | cuda].",
        )
        parser.add_argument(
            "--k_fold",
            action="store_false",
            help="if true, use K-fold cross-validation other makes standard train-test split.",
        )
        parser.add_argument(
            "--max_elms",
            type=int,
            default=-1,
            help="total number of elm events to be used. Use -1 to use all of them.",
        )
        # dataset parameters
        parser.add_argument(
            "--batch_size",
            type=int,
            default=64,
            help="batch size for model training and testing.",
        )
        parser.add_argument(
            "--signal_window_size",
            type=int,
            default=8,
            help="number of time data points to use for the input. "
            "The size of each input will then become `signal_window_size x spatial_dims x spatial_dims`, "
            "[8 | 16].",
        )
        parser.add_argument(
            "--label_look_ahead",
            type=int,
            default=0,
            help="`look ahead`, meaning the label for the entire signal window is taken to "
            "be label corresponding to the last element (0 ahead) of the signal window, "
            "[0 | 4 | 8 | ...].",
        )
        parser.add_argument(
            "--fraction_valid",
            type=float,
            default=0.2,
            help="size of the validation dataset.",
        )
        parser.add_argument(
            "--fraction_test",
            type=float,
            default=0.1,
            help="size of the test dataset.",
        )
        parser.add_argument(
            "--size",
            type=int,
            default=8,
            help="size of the input. Must be specified when using stacked ELM model.",
        )
        parser.add_argument(
            "--data_mode",
            type=str,
            default="unbalanced",
            help="which data mode to use, `balanced` upsamples the data to reduce class imbalance "
            "`unbalanced` uses the raw data as is.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=0,
            help="turns on the multi-processing data loading with `num_workers` loader "
            "worker processes. Default: 0 meaning the data loading step will be done by "
            "the main process.",
        )
        parser.add_argument(
            "--add_noise",
            action="store_false",
            help="if true, add Gaussian noise to the data with user supplied mean and std dev.",
        )
        parser.add_argument(
            "--mu",
            type=float,
            default=0,
            help="mean of the Gaussian noise. Must be passed "
            "when `add_noise=True`.",
        )
        parser.add_argument(
            "--sigma",
            type=float,
            default=0.01,
            help="standard deviation of the Gaussian noise. Must be passed "
            "when `add_noise=True`.",
        )
        self.initialized = True

        return parser

    def _gather_args(self):
        """Initialize the parser."""
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.create_parser(parser)

        # get the base options
        self.parser = parser
        args = parser.parse_args()

        return args

    def _print_args(self, args):
        """Print command line arguments.
        It will print both current arguments as well as the default values (if different).
        """
        message = ""
        message += "----------- Parameters used: -----------\n"
        for k, v in sorted(vars(args).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = f"\t[default: {default}]"
            message += f"{str(k):>25}: {str(v):<30} {comment}\n"
        message += "---------- End -----------\n"

        print(message)

    def parse(self):
        """Parse our arguments."""
        args = self._gather_args()
        self._print_args(args)

        self.args = args
        return args
