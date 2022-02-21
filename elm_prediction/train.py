'''
Main module for training ELM prediction models

To train an ELM prediction model, call `train_loop()` using
a recipe similar to `if __name__=="main"` block.
'''

import os
import sys
import time
import pickle
import argparse
from typing import Union
from pathlib import Path
import logging

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

import torch
import torch.nn as nn
import torch.distributed
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import optuna
except ImportError:
    optuna = None

try:
    from .options.train_arguments import TrainArguments
    from .src import utils, trainer, dataset
    from .models import multi_features_ds_model
except ImportError:
    from elm_prediction.options.train_arguments import TrainArguments
    from elm_prediction.src import utils, trainer, dataset
    from elm_prediction.models import multi_features_ds_model


def train_loop(
    args: argparse.Namespace,
    desc: bool = False,
    trial = None,  # optuna `trial` object
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
    # output directory and files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / args.output_file
    log_file = output_dir / args.log_file
    args_file = output_dir / args.args_file
    test_data_file, checkpoint_file = utils.create_output_paths(args)

    LOGGER = utils.get_logger(script_name=__name__, log_file=log_file)

    data_cls = utils.create_data(args.data_preproc)
    data_obj = data_cls(args, LOGGER)

    with args_file.open('wb') as f:
        pickle.dump(args, f)

    if _rank is not None:
        # override args.device for multi-GPU distributed data parallel training
        args.device = f'cuda:{_rank}'
        LOGGER.info(f'Distributed data parallel: process rank {_rank} on GPU {args.device}')

    LOGGER.info("-" * 30)

    if not args.dry_run:
        LOGGER.info(f"Test data will be saved to: {test_data_file}")
    LOGGER.info("-" * 30)

    # create train, valid and test data
    train_data, valid_data, _ = data_obj.get_data(
        shuffle_sample_indices=args.shuffle_sample_indices,
    )

    # dump test data into a file
    if not args.dry_run:
        with open(test_data_file, "wb") as f:
            pickle.dump(
                {
                    "signals": valid_data[0],
                    "labels": valid_data[1],
                    "sample_indices": valid_data[2],
                    "window_start": valid_data[3],
                },
                f,
            )

    # create datasets
    train_dataset = dataset.ELMDataset(
        args, *train_data, logger=LOGGER, phase="training"
    )
    valid_dataset = dataset.ELMDataset(
        args, *valid_data, logger=LOGGER, phase="validation"
    )

    # training and validation dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # model
    if args.model_name == 'multi_features_ds':
        raw_model = (
            multi_features_ds_model.RawFeatureModel(args)
            if args.raw_num_filters > 0
            else None
        )
        fft_model = (
            multi_features_ds_model.FFTFeatureModel(args)
            if args.fft_num_filters > 0
            else None
        )
        dwt_model = (
            multi_features_ds_model.DWTFeatureModel(args)
            if args.dwt_num_filters > 0
            else None
        )
        model_cls_args = (args, raw_model, fft_model, dwt_model)
    else:
        model_cls_args = (args, )

    model_cls = utils.create_model(args.model_name)

    device = torch.device(args.device)

    if _rank is None:
        # standard model training on CPU or single GPU
        model = model_cls(*model_cls_args)
        model = model.to(device)
    else:
        # model training on multiple GPUs with `distributed data parallel`
        original_model = model_cls(*model_cls_args)
        original_model.to(device)
        model = DDP(original_model, device_ids=[_rank])


    LOGGER.info("-" * 50)
    LOGGER.info(f"       Training with model: {args.model_name}       ")
    LOGGER.info("-" * 50)

    n_parameters = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            n_parameters += parameter.numel()

    LOGGER.info(f"  Trainable parameters: {n_parameters}       ")

    # display model details
    if desc:
        if args.model_name == "rnn":
            input_size = (args.batch_size, args.signal_window_size, 64)
        else:
            if args.data_preproc == "interpolate":
                input_size = (
                    args.batch_size,
                    1,
                    args.signal_window_size,
                    args.interpolate_size,
                    args.interpolate_size,
                )
            elif args.data_preproc == "gradient":
                input_size = (
                    args.batch_size,
                    6,
                    args.signal_window_size,
                    8,
                    8,
                )
            elif args.data_preproc == "rnn":
                input_size = (args.batch_size, args.signal_window_size, 64)
            else:
                input_size = (
                    args.batch_size,
                    1,
                    args.signal_window_size,
                    8,
                    8,
                )
        x = torch.rand(*input_size)
        x = x.to(device)
        utils.model_details(model, x, input_size)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # get the lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        verbose=True,
    )

    # loss function
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # define variables for ROC and loss
    best_score = 0

    # instantiate training object
    use_rnn = True if args.data_preproc == "rnn" else False
    engine = trainer.Run(
        model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        use_focal_loss=args.focal_loss,
        use_rnn=use_rnn,
    )

    # containers to hold train and validation losses
    train_loss = np.empty(0)
    valid_loss = np.empty(0)
    roc_scores = np.empty(0)
    f1_scores = np.empty(0)

    outputs = {}

    # iterate through all the epochs
    for epoch in range(args.n_epochs):
        start_time = time.time()

        # train over an epoch
        avg_loss = engine.train(
            train_loader, 
            epoch, 
            print_every=args.train_print_every,
        )
        train_loss = np.append(train_loss, avg_loss)

        # evaluate validation data
        avg_val_loss, preds, valid_labels = engine.evaluate(
            valid_loader, 
            print_every=args.valid_print_every
        )
        valid_loss = np.append(valid_loss, avg_val_loss)

        # step the learning rate scheduler
        scheduler.step(avg_val_loss)

        # ROC scoring
        roc_score = roc_auc_score(valid_labels, preds)
        roc_scores = np.append(roc_scores, roc_score)

        # F1 scoring
        # hard coding the threshold value for F1 score, smaller value will reduce
        # the number of false negatives while larger value reduces the number of
        # false positives
        thresh = 0.35
        f1 = f1_score(valid_labels, (preds > thresh).astype(int))
        f1_scores = np.append(f1_scores, f1)
        elapsed = time.time() - start_time

        LOGGER.info(f"Ep: {epoch+1:03d} \tavg train loss: {avg_loss:.3f} \tavg val. loss: {avg_val_loss:.3f}")
        LOGGER.info(f"Ep: {epoch+1:03d} \tROC-AUC: {roc_score:.3f} \tF1: {f1:.3f} \ttime elapsed: {elapsed:.1f} s")

        # update and save outputs
        outputs['train_loss'] = train_loss
        outputs['valid_loss'] = valid_loss
        outputs['roc_scores'] = roc_scores
        outputs['f1_scores'] = f1_scores

        with open(output_file.as_posix(), "wb") as f:
            pickle.dump(outputs, f)

        # track best f1 score and save model
        if f1 > best_score:
            best_score = f1
            LOGGER.info(f"Ep: {epoch+1:03d} \tBest Score: {best_score:.3f}")
            if not args.dry_run:
                LOGGER.info(f"Ep: {epoch+1:03d} \tSaving model to: {checkpoint_file}")
                model_data = {
                    "model": model.state_dict(), 
                    "preds": preds,
                }
                torch.save(model_data, checkpoint_file)

        # optuna hook to monitor training epochs
        if trial is not None and optuna is not None:
            trial.report(f1, epoch)
            # save outputs as lists in trial user attributes
            for key, item in outputs.items():
                trial.set_user_attr(key, item.tolist())
            if trial.should_prune():
                LOGGER.info("Trial pruned by Optuna")
                for handler in LOGGER.handlers[:]:
                    handler.close()
                    LOGGER.removeHandler(handler)
                optuna.TrialPruned()

    # shut down logger handlers
    for handler in LOGGER.handlers[:]:
        handler.close()
        LOGGER.removeHandler(handler)

    return outputs


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # input arguments if no command line arguments in `sys.argv`
        input_data_file = Path(__file__).parent / 'labeled-elm-events-10elms.hdf5'
        arg_list = [
            "--input_data_file", input_data_file.as_posix(),
            "--n_epochs",  "2",
        ]
    else:
        # use command line arguments in `sys.argv`
        arg_list = None
    args = TrainArguments().parse(verbose=True, arg_list=arg_list)
    train_loop(args)
    