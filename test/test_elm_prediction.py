import sys
import pytest
import shutil

from bes_edgeml_models.elm_classification.train import train_loop as train_elm_classification
from bes_edgeml_models.elm_regression.train import train_loop as train_elm_regression
from bes_edgeml_models.turbulence_regime_classification.train import train_loop as train_regime_classification
from bes_edgeml_models.velocimetry.train import train_loop as train_velocimetry
from bes_edgeml_models.elm_classification.analyze import Analysis


WORK_DIR = '.../bes-edgeml-work/'
RUN_DIR = WORK_DIR + '/tests/'
DEFAULT_INPUT_ARGS = {
    'max_elms':10,
    'n_epochs':1,
    'fraction_valid':0.2,
    'fraction_test':0.2,
    'batch_size': 16,
    'signal_window_size': 64,
}


def test_train_loop_raw_only():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_classification'
    input_args['output_dir'] = RUN_DIR + '/raw_only'
    train_elm_classification(input_args)

def test_train_loop_save_onnx():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_classification'
    input_args['output_dir'] = RUN_DIR + '/save_onnx'
    input_args['save_onnx'] = True
    train_elm_classification(input_args)

def test_train_loop_list_args():
    input_args = [f"--{key}={value}" for key, value in DEFAULT_INPUT_ARGS.items()]
    input_args.extend(['--output_dir', RUN_DIR + '/list_args'])
    input_args.extend(['--model_name=multi_features_classification'])
    train_elm_classification(input_args)

def test_train_loop_with_fft():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_classification'
    input_args['output_dir'] = RUN_DIR + '/with_fft'
    input_args['fft_num_filters'] = 8
    train_elm_classification(input_args)

def test_train_loop_with_dwt():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_classification'
    input_args['output_dir'] = RUN_DIR + '/with_dwt'
    input_args['dwt_num_filters'] = 8
    train_elm_classification(input_args)

def test_train_loop_with_dct():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_classification'
    input_args['output_dir'] = RUN_DIR + '/with_dct'
    input_args['dct_num_filters'] = 8
    train_elm_classification(input_args)

def test_regression_with_all():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_regression'
    input_args['output_dir'] = RUN_DIR + '/regression/with_all'
    input_args['dct_num_filters'] = 8
    input_args['dwt_num_filters'] = 8
    input_args['fft_num_filters'] = 8
    train_elm_regression(input_args)

def test_regime_classification_with_all():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_classification'
    input_args['output_dir'] = RUN_DIR + '/turbulence_regime_classification/with_all'
    input_args['dct_num_filters'] = 8
    input_args['dwt_num_filters'] = 8
    input_args['fft_num_filters'] = 8
    train_regime_classification(input_args)

def test_velocimetry_with_all():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_velocimetry'
    input_args['output_dir'] = RUN_DIR + '/turbulence_regime_classification/with_all'
    input_args['dct_num_filters'] = 8
    input_args['dwt_num_filters'] = 8
    input_args['fft_num_filters'] = 8
    train_velocimetry(input_args)

def test_multifeatures_v2():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_classification'
    input_args['output_dir'] = RUN_DIR + '/mfv2'
    input_args['signal_window_size'] = 128
    input_args['mf_time_slice_interval'] = 2
    input_args['subwindow_size'] = 32
    train_elm_classification(input_args)

def test_mfv2_analyze():
    run_dir = RUN_DIR + '/mfv2'
    run = Analysis(run_dir=run_dir)
    run.plot_all()

def test_mfv2_do_analysis():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_classification'
    input_args['output_dir'] = RUN_DIR + '/mfv2_analysis'
    input_args['do_analysis'] = True
    train_elm_classification(input_args)

def test_mfv2_valid_indices():
    for valid_indices_method in range(4):
        input_args = DEFAULT_INPUT_ARGS.copy()
        input_args['model_name'] = 'multi_features_classification'
        input_args['output_dir'] = RUN_DIR + f'/mfv2_valid_indices_method_{valid_indices_method:02d}'
        input_args['valid_indices_method'] = valid_indices_method
        train_elm_classification(input_args)

def test_mfv2_with_fft():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_classification'
    input_args['output_dir'] = RUN_DIR + '/mfv2_with_fft'
    input_args['signal_window_size'] = 128
    input_args['mf_time_slice_interval'] = 2
    input_args['subwindow_size'] = 32
    input_args['fft_num_filters'] = 8
    input_args['fft_nbins'] = 2
    train_elm_classification(input_args)

def test_mfv2_with_dct():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_classification'
    input_args['output_dir'] = RUN_DIR + '/mfv2_with_dct'
    input_args['signal_window_size'] = 128
    input_args['mf_time_slice_interval'] = 2
    input_args['subwindow_size'] = 32
    input_args['dct_num_filters'] = 8
    input_args['dct_nbins'] = 2
    train_elm_classification(input_args)

def test_mfv2_with_dwt():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_classification'
    input_args['output_dir'] = RUN_DIR + '/mfv2_with_dwt'
    input_args['signal_window_size'] = 128
    input_args['mf_time_slice_interval'] = 2
    input_args['subwindow_size'] = 32
    input_args['dwt_num_filters'] = 8
    train_elm_classification(input_args)

def test_mfv2_sgd():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_classification'
    input_args['output_dir'] = RUN_DIR + '/mfv2_sgd'
    input_args['optimizer'] = 'sgd'
    input_args['momentum'] = 0.1
    train_elm_classification(input_args)

def test_multifeatures_v2_regression():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_classification'
    input_args['output_dir'] = RUN_DIR + '/mf_v2_regression'
    input_args['regression'] = True
    train_elm_classification(input_args)

def test_multifeatures_v2_regression_with_log():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_classification'
    input_args['output_dir'] = RUN_DIR + '/mf_v2_regression_with_log'
    input_args['regression'] = 'log'
    train_elm_classification(input_args)

def test_multifeatures_v2_cnn():
    input_args = DEFAULT_INPUT_ARGS.copy()
    input_args['model_name'] = 'multi_features_classification'
    input_args['output_dir'] = RUN_DIR + '/mf_v2_cnn'
    input_args['cnn_layer1_num_filters'] = 8
    input_args['cnn_layer2_num_filters'] = 8
    input_args['raw_num_filters'] = 0
    train_elm_classification(input_args)

if __name__=="__main__":
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    sys.exit(pytest.main(['--verbose', '--exitfirst']))