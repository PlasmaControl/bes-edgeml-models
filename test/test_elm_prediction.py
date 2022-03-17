import sys
import pytest

from elm_prediction.train import train_loop


DEFAULT_INPUT_ARGS = {
    'max_elms':10,
    'n_epochs':1,
    'fraction_valid':0.2,
    'fraction_test':0.2,
}


def test_train_loop_raw_only():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['output_dir'] = 'run_dir/raw_only'
    train_loop(input_args)

def test_train_loop_list_args():
    input_args = [f"--{key}={value}" for key, value in DEFAULT_INPUT_ARGS.items()]
    input_args.extend(['--output_dir', 'run_dir/list_args'])
    train_loop(input_args)

def test_train_loop_fft():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['output_dir'] = 'run_dir/fft'
    input_args['fft_num_filters'] = 8
    train_loop(input_args)

def test_train_loop_dwt():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['output_dir'] = 'run_dir/dwt'
    input_args['dwt_num_filters'] = 8
    train_loop(input_args)

def test_train_loop_dct():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['output_dir'] = 'run_dir/dct'
    input_args['dct_num_filters'] = 8
    train_loop(input_args)


if __name__=="__main__":
    sys.exit(pytest.main())