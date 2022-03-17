import sys
import pytest

from elm_prediction.options.train_arguments import TrainArguments
from elm_prediction.train import train_loop

arg_list_default = [
    '--max_elms', '10',
    '--n_epochs', '1',
    '--fraction_valid', '0.2',
    '--fraction_test', '0.2'
]

def test_train_loop_raw_only():
    arg_list = arg_list_default.copy()
    arg_list.extend([
        '--output_dir', 'run_dir/raw_only',
    ])
    args = TrainArguments().parse(verbose=True, arg_list=arg_list)
    train_loop(args)

def test_train_loop_fft():
    arg_list = arg_list_default.copy()
    arg_list.extend([
        '--fft_num_filters', '8',
        '--output_dir', 'run_dir/fft',
    ])
    args = TrainArguments().parse(verbose=True, arg_list=arg_list)
    train_loop(args)

def test_train_loop_dwt():
    arg_list = arg_list_default.copy()
    arg_list.extend([
        '--dwt_num_filters', '8',
        '--output_dir', 'run_dir/dwt',
    ])
    args = TrainArguments().parse(verbose=True, arg_list=arg_list)
    train_loop(args)

def test_train_loop_dct():
    arg_list = arg_list_default.copy()
    arg_list.extend([
        '--dct_num_filters', '8',
        '--output_dir', 'run_dir/dct',
    ])
    args = TrainArguments().parse(verbose=True, arg_list=arg_list)
    train_loop(args)


if __name__=="__main__":
    sys.exit(pytest.main())