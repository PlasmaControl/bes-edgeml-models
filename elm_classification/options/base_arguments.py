import sys
import argparse


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
            "--input_file",
            type=str,
            default="labeled-elm-events-large.hdf5",
            help="path to the input hdf5 file.",
        )
        parser.add_argument(
            "--model_name",
            type=str,
            required=True,
            help="name of the model to be used for training, "
            "[feature | feature_v2 | cnn | cnn_v2 | cnn_2d | rnn | lstm_ae | fc_ae].",
        )
        parser.add_argument(
            "--model_ckpts",
            type=str,
            default="model_checkpoints",
            help="path to the pretrained weights of the saved models.",
        )
        parser.add_argument(
            "--use_all_data",
            action="store_true",
            default=False,
            help="if true, don't split the data into training, testing and validation "
            "sets.",
        )
        parser.add_argument(
            "--multi_features",
            action="store_true",
            default=False,
            help="if true, create additional features based on FFT and CWT",
        )
        parser.add_argument(
            "--use_fft",
            action="store_true",
            default=False,
            help="if true, create additional features from FFT. As of version 1.7.0, "
            "PyTorch FFT function does not work on systems that do not support "
            "Intel MKL library(like PPC systems).",
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            default="data",
            help="path to the input data.",
        )
        parser.add_argument(
            "--test_data_dir",
            type=str,
            default="data/test_data",
            help="path to save the test data.",
        )
        parser.add_argument(
            "--log_dir",
            type=str,
            default="logs",
            help="path to save the logs.",
        )
        parser.add_argument(
            "--add_tensorboard",
            action="store_true",
            default=False,
            help="if true, write loss summary to a tensorboard log file.",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cpu",
            help="device to use, [cuda | cpu].",
        )
        parser.add_argument(
            "--dry_run",
            action="store_true",
            default=False,
            help="if true, train (test) the model without saving anything.",
        )
        parser.add_argument(
            "--n_epochs",
            type=int,
            default=10,
            help="total number of epochs for training.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="seed of the PRNG for reproducibity of results.",
        )
        # parser.add_argument(
        #     "--kfold",
        #     action="store_true",
        #     default=False,
        #     help="if true, use K-fold cross-validation other makes standard train-test split.",
        # )
        # parser.add_argument(
        #     "--n_folds",
        #     type=int,
        #     help="number of folds for k-fold cross validation. Only passed when "
        #     "`kfold` is set to True.",
        # )
        parser.add_argument(
            "--max_elms",
            type=int,
            default=-1,
            help="total number of elm events to be used. Use -1 to use all of them.",
        )
        parser.add_argument(
            "--filename_suffix",
            type=str,
            default="",
            help="suffix in the file name. It can be passed when args like interpolate, "
            "etc. are passed. Must be passed with a leading underscore '_'.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=64,
            help="batch size for model training and testing.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=0,
            help="turns on the multi-processing data loading with `num_workers` loader "
            "worker processes. Default: 0 meaning the data loading step will be done by "
            "the main process.",
        )

        # data preparation parameters
        parser.add_argument(
            "--data_preproc",
            type=str,
            required=True,
            help="name of the data manipulator to be used. Selecting any of the "
            "mentioned techniques will create data ready corresponding to that "
            "technique for training, "
            "[unprocessed | automatic_labels | wavelet | gradient | interpolate | "
            "balance | rnn | gaussian_noise].",
        )
        parser.add_argument(
            "--signal_window_size",
            type=int,
            default=16,
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
            default=0.15,
            help="size of the validation dataset.",
        )
        parser.add_argument(
            "--fraction_test",
            type=float,
            default=0.1,
            help="size of the test dataset.",
        )
        parser.add_argument(
            "--truncate_inputs",
            action="store_true",
            default=False,
            help="if true, truncates the time dimension upto `truncate_buffer` "
            "time frames beyond the first frame beyond active elm events",
        )
        parser.add_argument(
            "--truncate_buffer",
            type=int,
            default=75,
            help="number of frames beyond first active elm event to consider when "
            "`truncate_inputs` is passed. ",
        )
        parser.add_argument(
            "--normalize_data",
            action="store_true",
            default=False,
            help="if true, normalizes the data in spatial dimensions. Divides the "
            "channels 1 to 32 by 10 and channels 33 to 64 by 5.",
        )
        parser.add_argument(
            "--interpolate_size",
            type=int,
            required="interpolate" in sys.argv,
            help="final size of the spatial dimensions of the input if interpolation is done. "
            "Must be passed if `data_preproc` == `interpolate`.",
        )
        parser.add_argument(
            "--mu",
            type=float,
            required="gaussian_noise" in sys.argv,
            help="mean of the Gaussian noise. Must be passed "
            "when `gaussian_noise` is passed as `data_preproc`.",
        )
        parser.add_argument(
            "--sigma",
            type=float,
            required="gaussian_noise" in sys.argv,
            help="standard deviation of the Gaussian noise. Must be passed "
            "when `gaussian_noise` is passed as `data_preproc`.",
        )
        parser.add_argument(
            "--shuffle_sample_indices",
            action="store_true",
            default=False,
            help="if true, shuffle the sample indices calculated based on `signal_window_size` "
            "and `label_look_ahead`.",
        )
        self.initialized = True

        return parser

    def _gather_args(self, arg_list: list = []):  # implement `arg_list`
        """Initialize the parser."""
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.create_parser(parser)

        # get the base options
        self.parser = parser
        args = parser.parse_args(arg_list)

        return args, parser

    def _print_args(self, args):
        """Print command line arguments.
        It will print both current arguments as well as the default values (if different).
        """
        message = ""
        message += "------------- Parameters used: -------------\n"
        for k, v in sorted(vars(args).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = f"\t[default: {default}]"
            message += f"{str(k):>20}: {str(v):<15} {comment}\n"
        message += "-------------- End ---------------\n"

        print(message)

    def parse(self, verbose: bool = False, arg_list: list = []):  # implement `arg_list`
        """Parse the arguments."""
        args, parser = self._gather_args(arg_list)
        if verbose:
            self._print_args(args)

        self.args = args
        return args, parser
