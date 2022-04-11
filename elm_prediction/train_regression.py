"""
Main module for training regression based ELM prediction models

To train an ELM prediction model, call `train_loop()` using
a recipe similar to `if __name__=="main"` block.
"""

import sys
import time
import pickle
import argparse
from typing import Union
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
import torchinfo

try:
    import optuna
except ImportError:
    optuna = None

try:
    from .options.train_arguments import TrainArguments
    from .src import utils, trainer, dataset
except ImportError:
    from elm_prediction.options.train_arguments import TrainArguments
    from elm_prediction.src import utils, trainer, dataset


def train_loop(input_args: Union[list, dict, None] = None,
               trial=None,  # optuna `trial` object
               _rank: Union[int, None] = None,  # process rank for data parallel dist. training; *must* be last arg
) -> dict:
    """Actual function to put the model to training. Use command line arg
    `--dry_run` to not create test data file and model checkpoint.

    Args:
    -----
        args (argparse.Namespace): Namespace object that stores all the command
            line arguments.
        data_obj (object): Data object that creates train, validation and test data.
        test_datafile_name (str): Name of the pickle file that stores the test data.
        fold (Union[int, None]): Integer index for the fold if using k-fold cross
        validation. Defaults to None.
        desc (bool): If true, prints the model architecture and details.
    """

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
    elif input_args and not isinstance(input_args, argparse.Namespace):
        raise TypeError(f'input_args must be a dict or Namespace, not {type(input_args)}')
    else:
        args = input_args

    # output directory and files
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir.as_posix()
    args.data_preproc = 'regression'
    args.label_look_ahead = 0
    args.truncate_buffer = 0

    file_suffix = ''
    file_suffix += '_regression' if args.regression else ''
    file_suffix += '_log' if args.regression == 'log' else ''

    output_file = output_dir / args.output_file
    log_file = output_dir / args.log_file
    args_file = output_dir / (Path(args.args_file).stem + file_suffix + Path(args.args_file).suffix)
    test_data_file, checkpoint_file = utils.create_output_paths(args)

    LOGGER = utils.get_logger(script_name=__name__, log_file=log_file)

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with args_file.open('wb') as f:
        pickle.dump(args, f)

    if _rank is not None:
        # override args.device for multi-GPU distributed data parallel training
        args.device = f'cuda:{_rank}'
        LOGGER.info(f'  Distributed data parallel: process rank {_rank} on GPU {args.device}')

    # create train, valid and test data
    data_cls = utils.create_data_class(args.data_preproc)
    data_obj = data_cls(args, LOGGER)
    train_data, valid_data, test_data = data_obj.get_data()

    # dump test data into a file
    if not args.dry_run:
        LOGGER.info(f"  Test data will be saved to: {test_data_file}")
        with open(test_data_file, "wb") as f:
            pickle.dump({"signals": test_data[0], "labels": test_data[1], "sample_indices": test_data[2],
                    "window_start": test_data[3], "elm_indices": test_data[4], }, f, )

    # create datasets
    train_dataset = dataset.ELMDataset(args, *train_data[0:4], logger=LOGGER)
    valid_dataset = dataset.ELMDataset(args, *valid_data[0:4], logger=LOGGER)

    # training and validation dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True, )

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True, )

    # device
    device = torch.device(args.device)
    LOGGER.info(f'------>  Target device: {device}')

    # model class and model instance
    model_class = utils.create_model_class(args.model_name)
    model = model_class(args)
    model = model.to(device)

    # distribute model for data-parallel training
    if _rank is not None:
        model = DistributedDataParallel(model, device_ids=[_rank])

    LOGGER.info(f"------>  Model: {args.model_name}       ")

    # display model details
    if args.model_name == "rnn":
        input_size = (args.batch_size, args.signal_window_size, 64)
    else:
        if args.data_preproc == "interpolate":
            input_size = (args.batch_size, 1, args.signal_window_size, args.interpolate_size, args.interpolate_size,)
        elif args.data_preproc == "gradient":
            input_size = (args.batch_size, 6, args.signal_window_size, 8, 8,)
        elif args.data_preproc == "rnn":
            input_size = (args.batch_size, args.signal_window_size, 64)
        else:
            input_size = (args.batch_size, 1, args.signal_window_size, 8, 8,)
    x = torch.rand(*input_size)
    x = x.to(device)
    LOGGER.info("\t\t\t\tMODEL SUMMARY")
    if _rank is None:
        # skip torchinfo.summary if DistributedDataParallel
        torchinfo.summary(model, input_size=input_size, device=device)
    LOGGER.info(f'  Batched input size: {x.shape}')
    LOGGER.info(f"  Batched output size: {model(x).shape}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"  Model contains {n_params} trainable parameters!")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # get the lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            verbose=True, )

    # loss function
    criterion = nn.MSELoss(reduction="none")

    # define variables for ROC and loss
    best_score = -np.inf

    # instantiate training object
    use_rnn = True if args.data_preproc == "rnn" else False
    engine = trainer.Run(model,
                         device=device,
                         criterion=criterion,
                         optimizer=optimizer,
                         use_focal_loss=args.focal_loss,
                         use_rnn=use_rnn,)

    # containers to hold train and validation losses
    train_loss = np.empty(0)
    valid_loss = np.empty(0)
    r2_scores = np.empty(0)

    outputs = {}

    training_start_time = time.time()
    # iterate through all the epochs
    for epoch in range(args.n_epochs):
        start_time = time.time()

        # train over an epoch
        avg_loss = engine.train(train_loader, epoch, print_every=args.train_print_every, )
        avg_loss = np.sqrt(avg_loss)
        train_loss = np.append(train_loss, avg_loss)

        # evaluate validation data
        avg_val_loss, preds, valid_labels = engine.evaluate(valid_loader, print_every=args.valid_print_every)
        avg_val_loss = np.sqrt(avg_val_loss)
        valid_loss = np.append(valid_loss, avg_val_loss)

        # step the learning rate scheduler
        scheduler.step(avg_val_loss)

        # ROC scoring
        r2 = r2_score(valid_labels, preds)
        r2_scores = np.append(r2_scores, r2)

        # F1 scoring
        # hard coding the threshold value for F1 score, smaller value will reduce
        # the number of false negatives while larger value reduces the number of
        # # false positives
        # thresh = 0.35
        # f1 = f1_score(valid_labels, (preds > thresh).astype(int))
        # f1_scores = np.append(f1_scores, f1)
        elapsed = time.time() - start_time

        LOGGER.info(f"Epoch: {epoch + 1:03d} \tavg train RMSE: {avg_loss:.3f} \tavg val. RMSE: {avg_val_loss:.3f}")
        LOGGER.info(f"Epoch: {epoch + 1:03d} \tR2: {r2:.3f}\ttime elapsed: {elapsed:.1f} s")

        # update and save outputs
        outputs['train_loss'] = train_loss
        outputs['valid_loss'] = valid_loss
        outputs['r2_scores'] = r2_scores
        # outputs['f1_scores'] = f1_scores

        with open(output_file.as_posix(), "w+b") as f:
            pickle.dump(outputs, f)

        # track best f1 score and save model
        if r2 > best_score:
            best_score = r2
            LOGGER.info(f"Epoch: {epoch + 1:03d} \tBest R2 Score: {best_score:.3f}")
            if not args.dry_run:
                LOGGER.info(f"Epoch: {epoch + 1:03d} \tSaving model to: {checkpoint_file}")
                model_data = {"model": model.state_dict(), "preds": preds, }
                torch.save(model_data, checkpoint_file)
                model_data['valid_labels'] = valid_labels


        # optuna hook to monitor training epochs
        if trial is not None and optuna is not None:
            trial.report(r2, epoch)
            # save outputs as lists in trial user attributes
            for key, item in outputs.items():
                trial.set_user_attr(key, item.tolist())
            if trial.should_prune():
                LOGGER.info("--------> Trial pruned by Optuna")
                for handler in LOGGER.handlers[:]:
                    handler.close()
                    LOGGER.removeHandler(handler)
                optuna.TrialPruned()

    # shut down logger handlers
    for handler in LOGGER.handlers[:]:
        handler.close()
        LOGGER.removeHandler(handler)

    total_elapsed = time.time() - training_start_time
    LOGGER.info(f'Training complete in {total_elapsed:0.1f}')

    return outputs, model_data


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # input arguments if no command line arguments in `sys.argv`
        arg_list = [# '--max_elms', '5',
                '--n_epochs', '3', '--fraction_valid', '0.2', '--fraction_test', '0.2']
    else:
        # use command line arguments in `sys.argv`
        arg_list = None
    args = TrainArguments().parse(arg_list=arg_list)
    outputs, model_data = train_loop(args)
    preds = model_data['preds']
    real = model_data['valid_labels']
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(len(preds)), preds, label='Prediction')
    ax.plot(range(len(real)), real, label='Real')
    fig.suptitle(f'Performance of {args.model_name} Model')
    ax.set_title(f'Epochs: {args.n_epochs}, SWS: {args.signal_window_size}')
    ax.text(0.05, 0.9, f'Best r2: {outputs["r2_scores"].max():0.2f}\n'
                       f'Best validation RMSE: {outputs["valid_loss"].min():0.2f}',
            transform=ax.transAxes)
    ax.set_xlabel('Time ($\mu$s)')
    ax.set_ylabel('Time to ELM ($\mu$s)')

    plt.savefig(f'./figures/{args.model_name}-regression_epochs-{args.n_epochs}_SWS-{args.signal_window_size}.png')
    plt.show()
