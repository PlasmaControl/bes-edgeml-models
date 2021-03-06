import sys
import pytest
import shutil

from elm_prediction.train import train_loop
from elm_prediction.analyze import Analysis


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

def test_multifeatures_v2():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_ds_v2'
    input_args['output_dir'] = RUN_DIR + '/mfv2'
    input_args['signal_window_size'] = 128
    input_args['mf_time_slice_interval'] = 2
    input_args['subwindow_size'] = 32
    train_loop(input_args)

def test_mfv2_analyze():
    run_dir = RUN_DIR + '/mfv2'
    run = Analysis(run_dir=run_dir)
    run.plot_all()

def test_mfv2_do_analysis():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_ds_v2'
    input_args['output_dir'] = RUN_DIR + '/mfv2_analysis'
    input_args['do_analysis'] = True
    train_loop(input_args)

def test_mfv2_valid_indices():
    for valid_indices_method in range(4):
        input_args = DEFAULT_INPUT_ARGS.copy()
        input_args['model_name'] = 'multi_features_ds_v2'
        input_args['output_dir'] = RUN_DIR + f'/mfv2_valid_indices_method_{valid_indices_method:02d}'
        input_args['valid_indices_method'] = valid_indices_method
        train_loop(input_args)

def test_mfv2_with_fft():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_ds_v2'
    input_args['output_dir'] = RUN_DIR + '/mfv2_with_fft'
    input_args['signal_window_size'] = 128
    input_args['mf_time_slice_interval'] = 2
    input_args['subwindow_size'] = 32
    input_args['fft_num_filters'] = 8
    input_args['fft_nbins'] = 2
    train_loop(input_args)

def test_mfv2_with_dct():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_ds_v2'
    input_args['output_dir'] = RUN_DIR + '/mfv2_with_dct'
    input_args['signal_window_size'] = 128
    input_args['mf_time_slice_interval'] = 2
    input_args['subwindow_size'] = 32
    input_args['dct_num_filters'] = 8
    input_args['dct_nbins'] = 2
    train_loop(input_args)

def test_mfv2_with_dwt():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_ds_v2'
    input_args['output_dir'] = RUN_DIR + '/mfv2_with_dwt'
    input_args['signal_window_size'] = 128
    input_args['mf_time_slice_interval'] = 2
    input_args['subwindow_size'] = 32
    input_args['dwt_num_filters'] = 8
    train_loop(input_args)

def test_mfv2_sgd():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_ds_v2'
    input_args['output_dir'] = RUN_DIR + '/mfv2_sgd'
    input_args['optimizer'] = 'sgd'
    input_args['momentum'] = 0.1
    train_loop(input_args)

def test_multifeatures_v2_regression():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_ds_v2'
    input_args['output_dir'] = RUN_DIR + '/mf_v2_regression'
    input_args['regression'] = True
    train_loop(input_args)

def test_multifeatures_v2_regression_with_log():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_ds_v2'
    input_args['output_dir'] = RUN_DIR + '/mf_v2_regression_with_log'
    input_args['regression'] = 'log'
    train_loop(input_args)

def test_multifeatures_v2_cnn():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_ds_v2'
    input_args['output_dir'] = RUN_DIR + '/mf_v2_cnn'
    input_args['cnn_layer1_num_filters'] = 8
    input_args['cnn_layer2_num_filters'] = 8
    input_args['raw_num_filters'] = 0
    train_loop(input_args)

if __name__=="__main__":
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    sys.exit( pytest.main( ['--verbose', '--exitfirst'] ) )