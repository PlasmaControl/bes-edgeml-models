import sys
import argparse
from typing import Union

try:
    from .. import sample_labeled_elm_data_file
except ImportError:
    from elm_prediction import sample_labeled_elm_data_file


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
            "--input_data_file",
            type=str,
            default=sample_labeled_elm_data_file.as_posix(),
            help="path to the input hdf5 file.",
        )
        parser.add_argument(
            "--model_name",
            type=str,
            default='multi_features_ds',
            help="name of the model to be used for training. Do not add a trailing "
            "`_model` in the name."
        )
        parser.add_argument(
            "--device",
            type=str,
            default="auto",
            help="device to use: [ auto (default, cuda if avail, else cpu) | cpu | cuda | cuda:N ].",
        )
        parser.add_argument(
            "--n_epochs",
            type=int,
            default=2,
            help="total number of epochs for training.",
        )
        parser.add_argument(
            "--max_elms",
            type=int,
            default=10,
            help="number of elm events to use. -1 to use all ELM events.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=64,
            help="batch size for model training and testing. (default 64)",
        )
        parser.add_argument(
            "--signal_window_size",
            type=int,
            default=64,
            help="number of time data points to use for the input. "
            "The size of each input will then become `signal_window_size x spatial_dims x spatial_dims`, "
            "power of 2 like 32, 128, etc. (default 64)",
        )
        parser.add_argument(
            "--label_look_ahead",
            type=int,
            default=50,
            help="`look ahead`, meaning the label for the entire signal window is taken to "
            "be label corresponding to the last element (0 ahead) of the signal window, "
            "[ int >= 0 (50 default) ].",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="run_dir",
            help="path to output directory, rel. or abs.",
        )
        parser.add_argument(
            "--output_file",
            type=str,
            default="output.pkl",
            help="output training metrics file, rel. to `run_dir`",
        )
        parser.add_argument(
            "--log_file",
            type=str,
            default="output.log",
            help="log file, rel. to `run_dir`",
        )
        parser.add_argument(
            "--args_file",
            type=str,
            default="args.pkl",
            help="arguments file, rel. to `run_dir`",
        )
        parser.add_argument(
            "--test_data_file",
            type=str,
            default="test_data.pkl",
            help="test data file, rel. to `run_dir`",
        )
        parser.add_argument(
            "--checkpoint_file",
            type=str,
            default="checkpoint.pytorch",
            help="model checkpoint file, rel. to `run_dir`",
        )
        parser.add_argument(
            "--dry_run",
            action="store_true",
            default=False,
            help="if true, train (test) the model without saving anything.",
        )
        parser.add_argument(
            "--data_preproc",
            type=str,
            default='unprocessed',
            help="name of the data manipulator to be used. Selecting any of the "
            "mentioned techniques will create data ready corresponding to that "
            "technique for training. Do not add a trailing `_data` in the name."
            "[unprocessed | automatic_labels | wavelet | gradient | interpolate | "
            "balance | rnn | gaussian_noise].",
        )
        parser.add_argument(
            "--automatic_labels",
            action="store_true",
            default=False,
            help="if true, use automatic labels from the HDF5 file instead of manual labels..",
        )
        parser.add_argument(
                "--balance_data",
                nargs='?',
                const='clip_outside',
                default=False,
                help="Balance pre-ELM and ELM classes. Must be used with truncate_data."
                     "clip_outside to remove leading or trailing indices."
                     "clip_inside to remove interior indices between ELM and pre-ELM."
                     "clip_even to resize larger class evenly to match smaller class.",
                choices=['clip_outside', 'clip_inside', 'clip_even']
            )
        parser.add_argument("--use_all_data",
                action="store_true",
                default=False,
                help="if true, don't split the data into training, testing and validation "
                     "sets.", )
        parser.add_argument(
            "--add_tensorboard",
            action="store_true",
            default=False,
            help="if true, write loss summary to a tensorboard log file.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="seed of the PRNG for reproducibity of results.",
        )
        parser.add_argument(
            "--filename_suffix",
            type=str,
            default="",
            help="suffix in the file name. It can be passed when args like interpolate, "
            "etc. are passed. Must be passed with a leading underscore '_'.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=0,
            help="turns on the multi-processing data loading with `num_workers` loader "
            "worker processes. Default: 0 meaning the data loading step will be done by "
            "the main process.  Rec.: 1 for cpu or single gpu; 0 for distributed multi-gpu.",
        )
        parser.add_argument(
            "--distributed",
            type=int,
            default=1,
            help="data parallel distributed training: -1 all GPUs|1 single GPU (default)|N gpus",
        )
        parser.add_argument(
            "--fraction_valid",
            type=float,
            default=0.1,
            help="size of the validation dataset.",
        )
        parser.add_argument(
            "--fraction_test",
            type=float,
            default=0.15,
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
            default=True,
            help="if true, shuffle the sample indices calculated based on `signal_window_size` "
            "and `label_look_ahead`.",
        )
        # arguments for `multi_features_model.py`
        parser.add_argument(
            "--mf_maxpool_size",
            type=int,
            default=1,
            help="spatial maxpool: 1(no pooling)|2|4",
        )
        parser.add_argument(
            "--mf_time_slice_interval",
            type=int,
            default=1,
            help="Time slice interval (data[::interval]): power of 2: 1(default)|2|4|8 ...",
        )
        parser.add_argument(
            "--mf_dropout_rate",
            type=float,
            default=0.2,
            help="Dropout rate",
        )
        parser.add_argument(
            "--mf_negative_slope",
            type=float,
            default=0.02,
            help="RELU negative slope",
        )
        parser.add_argument(
            "--fc1_size",
            type=int,
            default=128,
            help="Size for fully-connected layer #1: int, default 128",
        )
        parser.add_argument(
            "--fc2_size",
            type=int,
            default=32,
            help="Size for fully-connected layer #2: int, default 32",
        )
        parser.add_argument(
            "--raw_num_filters",
            type=int,
            default=8,
            help="Number of features for RawFeatureModel: int >= 0",
        )
        parser.add_argument(
            "--fft_num_filters",
            type=int,
            default=0,
            help="Number of features for FFTFeatureModel: int >= 0",
        )
        parser.add_argument(
            "--fft_nbins",
            type=int,
            default=1,
            help="FFT bins for FFTFeatureModel; power of 2: 1(default)|2|4...",
        )
        parser.add_argument(
            "--dct_num_filters",
            type=int,
            default=0,
            help="Number of features for DCTFeatureModel: int >= 0",
        )
        parser.add_argument(
            "--dct_nbins",
            type=int,
            default=1,
            help="DCT bins for DCTFeatureModel; power of 2: 1(default)|2|4...",
        )
        parser.add_argument(
            "--dwt_num_filters",
            type=int,
            default=0,
            help="Number of features for DWTFeatureModel: int >= 0",
        )
        parser.add_argument(
            "--dwt_wavelet",
            type=str,
            default="db4",
            help="Wavelet string for DWTFeatureModel: default `db4`",
        )
        parser.add_argument(
            "--dwt_level",
            type=int,
            default=-1,
            help="Wavelet decomposition level: int >=1 (-1 (default) = max dwt level)",
        )

        self.initialized = True

        return parser

    def _gather_args(
        self, 
        arg_list: Union[list, None] = None,
        existing_namespace = None,
    ):
        """Initialize the parser."""
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.create_parser(parser)

        # get the base options
        self.parser = parser
        args = parser.parse_args(
            args=arg_list,
            namespace=existing_namespace,
            )

        # return args, parser
        return args

    def make_args_summary_string(self):
        message = "\n"
        message += "------------- Parameters used: -------------\n"
        for k, v in sorted(vars(self.args).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = f"\t[default: {default}]"
            message += f"{str(k):>20}: {str(v):<15} {comment}\n"
        message += "-------------- End ---------------\n"
        return message


    def _print_args(self, args):
        """Print command line arguments.
        It will print both current arguments as well as the default values (if different).
        """
        message = self.make_args_summary_string(args)
        print(message)

    def parse(
        self,
        arg_list: Union[list, None] = None,
        existing_namespace = None,
    ):
        """
        Parse arguments
        Note: `arg_list` will override command line inputs from `sys.argv`
        """
        args = self._gather_args(
            arg_list=arg_list,
            existing_namespace=existing_namespace,
            )

        self.args = args
        return args
