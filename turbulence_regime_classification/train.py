# Python imports
import shutil
import pickle
import sys
from pathlib import Path

# Local Imports
from elm_prediction.src import utils
from turbulence_regime_classification.data_preprocessing.base_data import BaseData
from turbulence_regime_classification.options.train_arguments import TrainArguments

# ML imports
import torch


def train_loop(input_args: dict,
               trial=None,
               _rank: int | None = None,
               ) -> dict:

    # parse input args
    args_obj = TrainArguments()
    if input_args and isinstance(input_args, dict):
        # format dict into list
        arg_list = []
        for key, value in input_args.items():
            if isinstance(value, bool):
                if value is True:
                    arg_list.append(f'--{key}')
            else:
                arg_list.append(f'--{key}={value}')
        input_args = arg_list
    args = args_obj.parse(arg_list=input_args)

    # output directory and files
    output_dir = Path(args.output_dir).resolve()
    args.output_dir = output_dir.as_posix()
    shutil.rmtree(output_dir.as_posix(), ignore_errors=True)
    output_dir.mkdir(parents=True)

    output_file = output_dir / args.output_file
    log_file = output_dir / args.log_file
    args_file = output_dir / args.args_file
    test_data_file, checkpoint_file = utils.create_output_paths(args)

    # create LOGGER
    LOGGER = utils.get_logger(script_name=__name__, log_file=log_file)
    LOGGER.info(args_obj.make_args_summary_string())

    LOGGER.info(f"  Output directory: {output_dir.resolve().as_posix()}")

    # save args
    LOGGER.info(f"  Saving argument file: {args_file.as_posix()}")
    with args_file.open('wb') as f:
        pickle.dump(args, f)
    LOGGER.info(f"  File size: {args_file.stat().st_size / 1e3:.1f} kB")

    # setup device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if _rank is not None:
        # override args.device for multi-GPU distributed data parallel training
        args.device = f'cuda:{_rank}'
        LOGGER.info(f'  Distributed data parallel: process rank {_rank} on GPU {args.device}')
    device = torch.device(args.device)
    LOGGER.info(f'------>  Target device: {device}')

    data_obj = BaseData(args, LOGGER)
    train_data, valid_data, test_data = data_obj.get_data()



if __name__ == '__main__':
    if len(sys.argv) == 1:
        # input arguments if no command line arguments in `sys.argv`
        args = {'model_name': 'multi_features_ds_v2',
                'input_data_dir': Path(__file__).parent / 'data',
                'device': 'cuda',
                'batch_size': 64,
                'n_epochs': 20,
                'max_elms': -1,
                'fraction_test': 0.025,
                'fft_num_filters': 20,
                'dwt_num_filters': 20,
                'signal_window_size': 256,
                'output_dir': Path(__file__).parents[2] / 'bes-edgeml-work/regime_classification'
            }
    else:
        # use command line arguments in `sys.argv`
        args = None
    train_loop(input_args=args)

