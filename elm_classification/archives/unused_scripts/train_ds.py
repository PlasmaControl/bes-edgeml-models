import os
import time
import pickle
import argparse
import warnings
from typing import Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import pywt

try:
    import optuna
except ImportError:
    optuna = None

from data_preprocessing import *
from options.train_arguments import TrainArguments
from src import utils, trainer, dataset
from models import multi_features_model
from models import multi_features_ds_model


def get_multi_features(args, train_data, valid_data):
    print(f"Train data shape: {train_data[0].shape}")
    print(f"Valid data shape: {valid_data[0].shape}")
    train_data_cwt = list(train_data)
    valid_data_cwt = list(valid_data)
    max_scale = 1024
    num = int(np.log2(max_scale)) + 1
    widths = np.round(
        np.geomspace(1, max_scale, num=num, endpoint=True)
    ).astype(int)
    train_data_cwt[0], _ = pywt.cwt(
        train_data_cwt[0], scales=widths, wavelet="morl", axis=0
    )
    train_data_cwt[0] = np.transpose(train_data_cwt[0], (1, 0, 2, 3))
    valid_data_cwt[0], _ = pywt.cwt(
        valid_data_cwt[0], scales=widths, wavelet="morl", axis=0
    )
    valid_data_cwt[0] = np.transpose(valid_data_cwt[0], (1, 0, 2, 3))

    train_data_cwt = tuple(train_data_cwt)
    valid_data_cwt = tuple(valid_data_cwt)
    print(f"CWT Train data shape: {train_data_cwt[0].shape}")
    print(f"CWT Valid data shape: {valid_data_cwt[0].shape}")

    print(f"CWT Train data label shape: {train_data_cwt[1].shape}")
    print(f"CWT Valid data label shape: {valid_data_cwt[1].shape}")

    assert (
        train_data[0].shape[0] == train_data_cwt[0].shape[0]
    ), "CWT train data leading dimension does not match with the raw data!"
    assert (
        valid_data[0].shape[0] == valid_data_cwt[0].shape[0]
    ), "CWT valid data leading dimension does not match with the raw data!"

    return train_data_cwt, valid_data_cwt


def train_loop(
    args: argparse.Namespace,
    data_obj: object,
    test_datafile_name: str = 'test_data.pkl',
    fold: Union[int, None] = None,
    desc: bool = False,
    trial = None,  # optuna `trial` object
    rank: Union[int, None] = None,  # process rank for data parallel dist. training; *must* be last arg
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
    LOGGER = data_obj.logger  # define `LOGGER` inside function

    # containers to hold train and validation losses
    train_loss = np.empty(0)
    valid_loss = np.empty(0)
    roc_scores = np.empty(0)
    f1_scores = np.empty(0)

    if rank is not None:
        # override args.device for multi-GPU distributed data parallel training
        args.device = f'cuda:{rank}'
        LOGGER.info(f'Distributed data parallel: process rank {rank} on GPU {args.device}')

    if not args.dry_run:
        test_data_path, model_ckpt_path = utils.create_output_paths(
            args, infer_mode=False,
        )
        test_data_file = os.path.join(test_data_path, test_datafile_name)
        LOGGER.info(f"Test data will be saved to: {test_data_file}")

        if args.multi_features:
            test_data_file_cwt = os.path.join(
                test_data_path, "cwt_" + test_datafile_name
            )

    LOGGER.info("-" * 30)

    LOGGER.info("-" * 30)
    LOGGER.info(f"       Training fold: {fold}       ")
    LOGGER.info("-" * 30)

    # turn off model details for subsequent folds/epochs
    if fold is not None:
        if fold >= 1:
            desc = False

    # create train, valid and test data
    train_data, valid_data, _ = data_obj.get_data(
        shuffle_sample_indices=args.shuffle_sample_indices,
        fold=fold
    )

    if args.multi_features:
        train_data_cwt, valid_data_cwt = get_multi_features(
            args,
            train_data,
            valid_data
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
        if args.multi_features:
            with open(test_data_file_cwt, "wb") as f:
                pickle.dump(
                    {
                        "signals": valid_data_cwt[0],
                        "labels": valid_data_cwt[1],
                        "sample_indices": valid_data_cwt[2],
                        "window_start": valid_data_cwt[3],
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

    if args.multi_features:
        train_dataset_cwt = dataset.ELMDataset(
            args, *train_data_cwt, logger=LOGGER, phase="training (CWT)"
        )
        valid_dataset_cwt = dataset.ELMDataset(
            args, *valid_data_cwt, logger=LOGGER, phase="validation (CWT)"
        )

        # create a combined dataset
        train_dataset = dataset.ConcatDatasets(train_dataset, train_dataset_cwt)
        valid_dataset = dataset.ConcatDatasets(valid_dataset, valid_dataset_cwt)

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
        model_args = (args, raw_model, fft_model, dwt_model)
    elif args.multi_features:
        raw_model = multi_features_model.RawFeatureModel(args)
        fft_model = multi_features_model.FFTFeatureModel(args)
        cwt_model = multi_features_model.CWTFeatureModel(args)
        model_args = (args, raw_model, fft_model, cwt_model)
    else:
        model_args = (args, )

    model_cls = utils.create_model(args.model_name)

    device = torch.device(args.device)

    if rank is None:
        # model training on CPU or single GPU
        model = model_cls(*model_args)
        model = model.to(device)
    else:
        # model training on multiple GPUs
        original_model = model_cls(*model_args)
        original_model.to(device)
        model = DDP(original_model, device_ids=[rank])


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
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # get the lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        verbose=True,
    )

    # loss function
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # define variables for ROC and loss
    best_score = 0
    best_loss = np.inf

    # instantiate training object
    use_rnn = True if args.data_preproc == "rnn" else False
    engine = trainer.Run(
        model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        use_focal_loss=args.focal_loss,
        use_rnn=use_rnn,
        multi_features=args.multi_features,
    )

    # output file
    outputs_file = (
        Path(args.output_dir) / 'outputs.pkl'
        # / f"signal_window_{args.signal_window_size}"
        # / f"label_look_ahead_{args.label_look_ahead}"
        # / "training_metrics"
        # / f"{args.model_name}{args.filename_suffix}.pkl"
    )
    outputs_file.parent.mkdir(
        parents=True, exist_ok=True
    )  # make dir. for output file

    outputs = {}

    # iterate through all the epochs
    for epoch in range(args.n_epochs):
        start_time = time.time()
        # train
        avg_loss = engine.train(
            train_loader, epoch, print_every=args.train_print_every
        )
        train_loss = np.append(train_loss, avg_loss)

        # evaluate
        avg_val_loss, preds, valid_labels = engine.evaluate(
            valid_loader, print_every=args.valid_print_every
        )
        valid_loss = np.append(valid_loss, avg_val_loss)

        # step the scheduler
        scheduler.step(avg_val_loss)

        # scoring
        roc_score = roc_auc_score(valid_labels, preds)
        roc_scores = np.append(roc_scores, roc_score)
        thresh = 0.35
        f1 = f1_score(valid_labels, (preds > thresh).astype(int))
        f1_scores = np.append(f1_scores, f1)
        elapsed = time.time() - start_time

        LOGGER.info(
            f"Epoch: {epoch + 1}, \tavg train loss: {avg_loss:.4f}, \tavg validation loss: {avg_val_loss:.4f}"
        )
        LOGGER.info(
            f"Epoch: {epoch +1}, \tROC-AUC score: {roc_score:.4f}, \tF1-score: {f1:.4f}, \ttime elapsed: {elapsed:.1f} s"
        )

        if f1 > best_score:
            best_score = f1
            LOGGER.info(
                f"Epoch: {epoch+1}, \tSave Best Score: {best_score:.4f} Model"
            )
            if not args.dry_run:
                # save the model if best f1 score is found
                model_save_path = os.path.join(
                    model_ckpt_path,
                    f"{args.model_name}_lookahead_{args.label_look_ahead}_{args.data_preproc}{args.filename_suffix}.pth",
                )
                torch.save(
                    {"model": model.state_dict(), "preds": preds},
                    model_save_path,
                )
                LOGGER.info(f"Model saved to: {model_save_path}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(
                f"Epoch: {epoch+1}, \tSave Best Loss: {best_loss:.4f} Model"
            )

        outputs['train_loss'] = train_loss
        outputs['valid_loss'] = valid_loss
        outputs['roc_scores'] = roc_scores
        outputs['f1_scores'] = f1_scores

        with open(outputs_file.as_posix(), "wb") as f:
            pickle.dump(outputs, f)

        # optuna hook to monitor training epochs
        if trial is not None:
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

    for handler in LOGGER.handlers[:]:
        handler.close()
        LOGGER.removeHandler(handler)

    return outputs


def _distributed_train_loop(rank, world_size, *train_loop_args):
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)
    print(len(train_loop_args))
    train_loop(*train_loop_args, rank=rank)
    torch.distributed.destroy_process_group()


def run_distributed_train_loop(*train_loop_args):
    """
    train_loop_args: Must be ordered argument list for train_loop(), excluding `rank`
    """
    args = train_loop_args[0]

    assert args.distributed != 1
    assert args.device.startswith('cuda')
    assert torch.distributed.is_available()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    world_size = torch.cuda.device_count() if args.distributed == -1 else args.distributed
    mp.spawn(_distributed_train_loop,
             args=(world_size, *train_loop_args),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    args, parser = TrainArguments().parse(verbose=True)
    LOGGER = utils.get_logger(
        script_name=__name__,
        log_file=os.path.join(
            args.log_dir,
            f"output_logs_{args.model_name}{args.filename_suffix}.log",
        ),
    )
    data_cls = utils.create_data(args.data_preproc)
    data_obj = data_cls(args, LOGGER)
    train_loop(
        args,
        data_obj,
        test_datafile_name=f"test_data_lookahead_{args.label_look_ahead}_{args.data_preproc}{args.filename_suffix}.pkl",
    )
