import os
import time
import pickle
import argparse
from typing import Union
from pathlib import Path

import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import pywt

from data_preprocessing import *
from options.train_arguments import TrainArguments
from src import utils, trainer, dataset
from models import multi_features_model


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
    test_datafile_name: str,
    fold: Union[int, None] = None,
    desc: bool = False,
) -> None:
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
    # containers to hold train and validation losses
    train_loss = []
    valid_loss = []
    roc_scores = []
    f1_scores = []
    test_data_path, model_ckpt_path = utils.create_output_paths(
        args, infer_mode=False
    )
    test_data_file = os.path.join(test_data_path, test_datafile_name)
    if args.multi_features:
        test_data_file_cwt = os.path.join(
            test_data_path, "cwt_" + test_datafile_name
        )

    LOGGER = data_obj.logger  # define `LOGGER` inside function
    LOGGER.info("-" * 30)

    if not args.dry_run:
        LOGGER.info(f"Test data will be saved to: {test_data_file}")
    LOGGER.info("-" * 30)
    LOGGER.info(f"       Training fold: {fold}       ")
    LOGGER.info("-" * 30)

    # turn off model details for subsequent folds/epochs
    if fold is not None:
        if fold >= 1:
            desc = False

    # create train, valid and test data
    train_data, valid_data, _ = data_obj.get_data(
        shuffle_sample_indices=args.shuffle_sample_indices, fold=fold
    )

    if args.multi_features:
        train_data_cwt, valid_data_cwt = get_multi_features(
            args, train_data, valid_data
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
    if args.multi_features:
        raw_model = multi_features_model.RawFeatureModel(args)
        fft_model = multi_features_model.FFTFeatureModel(args)
        cwt_model = multi_features_model.CWTFeatureModel(args)
        model_cls = utils.create_model(args.model_name)
        model = model_cls(args, raw_model, fft_model, cwt_model)
    else:
        model_cls = utils.create_model(args.model_name)
        model = model_cls(args)

    device = torch.device(args.device)
    model = model.to(device)
    LOGGER.info("-" * 50)
    LOGGER.info(f"       Training with model: {args.model_name}       ")
    LOGGER.info("-" * 50)

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

    # iterate through all the epochs
    for epoch in range(args.n_epochs):
        start_time = time.time()
        # train
        avg_loss = engine.train(
            train_loader, epoch, print_every=args.train_print_every
        )
        train_loss.append(avg_loss)

        # evaluate
        avg_val_loss, preds, valid_labels = engine.evaluate(
            valid_loader, print_every=args.valid_print_every
        )
        valid_loss.append(avg_val_loss)

        # step the scheduler
        scheduler.step(avg_val_loss)

        # scoring
        roc_score = roc_auc_score(valid_labels, preds)
        roc_scores.append(roc_score)
        thresh = 0.35
        f1 = f1_score(valid_labels, (preds > thresh).astype(int))
        f1_scores.append(f1)
        elapsed = time.time() - start_time

        LOGGER.info(
            f"Epoch: {epoch + 1}, \tavg train loss: {avg_loss:.4f}, \tavg validation loss: {avg_val_loss:.4f}"
        )
        LOGGER.info(
            f"Epoch: {epoch +1}, \tROC-AUC score: {roc_score:.4f}, \tF1-score: {f1:.4f}, \ttime elapsed: {elapsed}"
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
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    roc_scores = np.array(roc_scores)
    f1_scores = np.array(f1_scores)

    outputs_file = (
        Path("outputs")
        / f"signal_window_{args.signal_window_size}"
        / f"label_look_ahead_{args.label_look_ahead}"
        / "training_metrics"
        / f"{args.model_name}{args.filename_suffix}.pkl"
    )
    outputs_file.parent.mkdir(
        parents=True, exist_ok=True
    )  # make dir. for output file

    with open(outputs_file.as_posix(), "wb") as f:
        pickle.dump(
            {
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "roc_scores": roc_scores,
                "f1_scores": f1_scores,
            },
            f,
        )


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
