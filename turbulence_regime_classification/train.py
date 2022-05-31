# Python imports
import shutil
import pickle
import sys
from pathlib import Path
import time
import io
import numpy as np

# Local Imports
from elm_prediction.src import utils, trainer
from turbulence_regime_classification.data_preprocessing.base_data import BaseData
from turbulence_regime_classification.options.train_arguments import TrainArguments
from turbulence_regime_classification.src.dataset import TurbulenceDataset
from turbulence_regime_classification.src.sampler import RandomBatchSampler
from turbulence_regime_classification.src.utils import make_labels
from turbulence_regime_classification.models.multi_features_model import MultiFeaturesClassificationModel

# ML imports
from sklearn.metrics import roc_auc_score, f1_score, r2_score
import torch
import torchinfo
from torch.utils.data import DataLoader, BatchSampler
from torch.nn.parallel import DistributedDataParallel as DDP


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

    LOGGER.info(f'Checking for labeled datasets.')
    make_labels(Path(__file__).parent, LOGGER)

    dataset = TurbulenceDataset(args, LOGGER)
    train_set, test_set = dataset.train_test_split(0.5, seed=42)
    with train_set as ts:
        if args.dataset_to_ram:
            ts.load_datasets()
        train_loader = DataLoader(ts,
                                  batch_size=None,  # must be disabled when using samplers
                                  sampler=BatchSampler(RandomBatchSampler(ts, args),
                                                       batch_size=args.batch_size + args.signal_window_size - 1,
                                                       drop_last=True)
                                  )

        model = MultiFeaturesClassificationModel(args).to(device)
        # distribute model for data-parallel training
        if _rank is not None:
            model = DDP(model, device_ids=[_rank])

        LOGGER.info(f"------>  Model: {args.model_name}       ")

        # display model details
        input_size = (
            args.batch_size,
            1,
            args.signal_window_size,
            8,
            8,
        )
        x = torch.rand(*input_size)
        x = x.to(device)
        if _rank is None:
            # skip torchinfo.summary if DistributedDataParallel
            tmp_io = io.StringIO()
            sys.stdout = tmp_io
            torchinfo.summary(model, input_size=input_size, device=device)
            sys.stdout = sys.__stdout__
            LOGGER.info("\t\t\t\tMODEL SUMMARY")
            LOGGER.info(tmp_io.getvalue())
        LOGGER.info(f'  Batched input size: {x.shape}')
        LOGGER.info(f"  Batched output size: {model(x).shape}")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        LOGGER.info(f"  Model contains {n_params} trainable parameters!")

        # optimizer
        if args.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        elif args.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
                dampening=args.dampening,
            )
        else:
            raise ValueError

        # get the lr scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            verbose=True,
        )

        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        # define variables for ROC and loss
        best_score = -np.inf

        # instantiate training object
        use_rnn = True if args.data_preproc == "rnn" else False
        engine = trainer.Run(
            model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            use_focal_loss=args.focal_loss,
            use_rnn=use_rnn,
            inverse_label_weight=args.inverse_label_weight,
        )

        # containers to hold train and validation losses
        train_loss = np.empty(0)
        valid_loss = np.empty(0)
        scores = np.empty(0)
        roc_scores = np.empty(0)

        outputs = {}

        training_start_time = time.time()
        # iterate through all the epochs
        LOGGER.info(f"  Begin training loop with {args.n_epochs} epochs")
        for epoch in range(args.n_epochs):
            start_time = time.time()

            # train over an epoch
            avg_loss = engine.train(
                train_loader,
                epoch,
                print_every=args.train_print_every,
            )


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
                'dataset_to_ram': False,
                'fft_num_filters': 20,
                'dwt_num_filters': 20,
                'signal_window_size': 256,
                'output_dir': Path(__file__).parents[2] / 'bes-edgeml-work/regime_classification'
            }
    else:
        # use command line arguments in `sys.argv`
        args = None

    print('From disk')
    args['dataset_to_ram'] = False
    start = time.time()
    train_loop(input_args=args)
    print("Time loaded from disk: ", time.time() - start)

