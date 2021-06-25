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
            default="labeled-elm-events.hdf5",
            help="path to the input hdf5 file.",
        )
        parser.add_argument(
            "--model_name",
            type=str,
            default="feature_model",
            help="name of the model to be used for training, "
            "[feature_model | cnn_model | stacked_elm_model].",
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            default="data",
            help="path to the input data.",
        )
        parser.add_argument(
            "--log_dir",
            type=str,
            default="logs",
            help="path to save the logs.",
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
            "--device",
            type=str,
            default="cpu",
            help="device to be used for training and testing, [cpu | cuda].",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="seed of the PRNG for reproducibity of results.",
        )
        parser.add_argument(
            "--kfold",
            action="store_true",
            default=False,
            help="if true, use K-fold cross-validation other makes standard train-test split.",
        )
        parser.add_argument(
            "--n_folds",
            type=int,
            help="number of folds for k-fold cross validation. Only passed when "
            "`kfold` is set to True.",
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
            "--signal_dtype",
            type=str,
            default="float32",
            help="data type of input BES signals.",
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
            "--shuffle_sample_indices",
            action="store_true",
            default=False,
            help="if true, shuffle the sample indices calculated based on `signal_window_size` "
            "and `label_look_ahead`.",
        )
        parser.add_argument(
            "--stack_elm_events",
            action="store_true",
            default=False,
            help="if true, elm events must be stacked together. Only used when "
            "`StackedELMModel` is used.",
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
            help="size of the input, [8 | 12 | 16 | 24]. Must be specified when using stacked ELM model.",
        )
        parser.add_argument(
            "--data_mode",
            type=str,
            default="unbalanced",
            help="which data mode to use, [balanced | unbalanced]. `balanced` upsamples "
            "the data to reduce class imbalance, `unbalanced` uses the raw data as is.",
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
            action="store_true",
            default=False,
            help="if true, add Gaussian noise to the data with user supplied mean and std dev.",
        )
        parser.add_argument(
            "--mu",
            type=float,
            help="mean of the Gaussian noise. Must be passed "
            "when `add_noise=True`.",
        )
        parser.add_argument(
            "--sigma",
            type=float,
            help="standard deviation of the Gaussian noise. Must be passed "
            "when `add_noise=True`.",
        )
        parser.add_argument(
            "--smoothen_transition",
            action="store_true",
            help="if true, smoothen the labels so that there is a gradual transition "
            "of the labels from 0 to 1 with respect to the input time series.",
        )
        parser.add_argument(
            "--transition_halfwidth",
            type=int,
            help="transition halfwidth (used to smoothen the labels). Only applies "
            "when `smoothen_transition` is set to True.",
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

    def parse(self, verbose: bool = False):
        """Parse our arguments."""
        args, parser = self._gather_args()
        if verbose:
            self._print_args(args)

        self.args = args
        return args, parser
