import sys
import pytest
import shutil

from elm_prediction.train import train_loop
from elm_prediction.analyze import do_analysis


RUN_DIR = 'run_dir'
DEFAULT_INPUT_ARGS = {
    'max_elms':10,
    'n_epochs':1,
    'fraction_valid':0.2,
    'fraction_test':0.2,
}


def test_train_loop_raw_only():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['output_dir'] = RUN_DIR + '/raw_only'
    train_loop(input_args)

def test_train_loop_save_onnx():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['output_dir'] = RUN_DIR + '/save_onnx'
    input_args['save_onnx'] = True
    train_loop(input_args)

def test_train_loop_list_args():
    input_args = [f"--{key}={value}" for key, value in DEFAULT_INPUT_ARGS.items()]
    input_args.extend(['--output_dir', RUN_DIR + '/list_args'])
    train_loop(input_args)

def test_train_loop_with_fft():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['output_dir'] = RUN_DIR + '/with_fft'
    input_args['fft_num_filters'] = 8
    train_loop(input_args)

def test_train_loop_with_dwt():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['output_dir'] = RUN_DIR + '/with_dwt'
    input_args['dwt_num_filters'] = 8
    train_loop(input_args)

def test_train_loop_with_dct():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['output_dir'] = RUN_DIR + '/with_dct'
    input_args['dct_num_filters'] = 8
    train_loop(input_args)

def test_train_loop_valid_indices():
    for valid_indices_method in range(4):
        input_args = DEFAULT_INPUT_ARGS.copy()
        input_args['output_dir'] = RUN_DIR + f'/valid_indices_method_{valid_indices_method:02d}'
        input_args['valid_indices_method'] = valid_indices_method
        train_loop(input_args)

def test_analyze():
    args_file = RUN_DIR + '/raw_only/args.pkl'
    do_analysis(args_file, interactive=False, click_through_pages=False, save=True)


if __name__=="__main__":
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    pytest_args = [
        '--verbose',
    ]
    sys.exit(
        pytest.main(pytest_args)
    )