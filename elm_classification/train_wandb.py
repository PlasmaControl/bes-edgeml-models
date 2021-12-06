import argparse
import os
import pickle
import time
from typing import Union

import wandb

wandb.login()

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from options.train_arguments import TrainArguments
from src import utils, run, dataset
from models.feature_gradients_model import SpatialFeatures, TemporalFeatures


def make(
    args: argparse.Namespace, train_data: tuple, valid_data: tuple
) -> tuple:
    # create datasets
    train_dataset = dataset.ELMDataset(
        args, *train_data, logger=LOGGER, phase="training"
    )

    valid_dataset = dataset.ELMDataset(
        args,
        *valid_data,
        logger=LOGGER,
        phase="validation",
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
    model = get_model(args)
    device = torch.device(args.device)
    model = model.to(device)

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

    return train_loader, valid_loader, model, optimizer, scheduler, criterion


def get_model(args: argparse.Namespace) -> object:
    model_cls = utils.create_model(args.model_name)
    if args.model_name == "feature_gradients":
        spatial = SpatialFeatures(args)
        temporal = TemporalFeatures()
        model = model_cls(args, spatial, temporal)
    elif args.model_name == "mts_cnn":
        model = model_cls(
            args,
            in_channels=1,
            out_channels=[4, 8],
            diag_fc_units=[16, 16, 64],
            detect_fc_units=[64, 32, 1],
        )
    else:
        model = model_cls(args)

    return model


def display_model_details(args: argparse.Namespace, model: object) -> None:
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
    x = x.to(args.device)
    utils.model_details(model, x, input_size)


def training_log(
    loss: float,
    epoch: int,
    roc: float = None,
    f1: float = None,
    mode: str = "Training",
):
    if mode == "Validation":
        if roc is not None and f1 is not None:
            wandb.log(
                {"epoch": epoch, "validation loss": loss, "roc": roc, "f1": f1},
                step=epoch,
            )
        else:
            raise ValueError("ROC and f1 values are not passed.")
    else:
        wandb.log({"epoch": epoch, "training loss": loss}, step=epoch)
    print(f"{mode} loss after epoch {epoch}: {loss:.4f}")


def train_loop(
    args: argparse.Namespace,
    data_obj: object,
    test_datafile_name: str,
    config: dict,
    fold: Union[int, None] = None,
    desc: bool = True,
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
    # if (not args.kfold) and (fold is not None):
    #     LOGGER.info(
    #         f"K-fold is set to {args.kfold} but fold index is passed!"
    #         " Proceeding without using K-fold."
    #     )
    #     fold = None
    with wandb.init(
        project=f"{args.model_name}_{time.strftime('%m%d%Y')}",
        config=config,
    ):
        wandb.run.name = f"{args.model_name}_sws_{args.signal_window_size}_la_{args.label_look_ahead}"
        # containers to hold train and validation losses
        train_loss = []
        valid_loss = []
        test_data_path, model_ckpt_path = utils.create_output_paths(
            args, infer_mode=False
        )
        test_data_file = os.path.join(test_data_path, test_datafile_name)

        # add loss values to tensorboard
        if args.add_tensorboard:
            writer = SummaryWriter(
                log_dir=os.path.join(
                    args.log_dir,
                    "tensorboard",
                    f"{args.model_name}{args.filename_suffix}",
                )
            )

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

        # dump test data into to a file
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

        # create image transforms
        if (
            (args.model_name.startswith("feature"))
            or (args.model_name.startswith("cnn"))
            or (args.model_name.startswith("rnn"))
            # or (args.model_name.startswith("mts_cnn"))
        ):
            transforms = None
        else:
            transforms = dataset.get_transforms(args)

        LOGGER.info("-" * 50)
        LOGGER.info(f"       Training with model: {args.model_name}       ")
        LOGGER.info("-" * 50)

        # get dataloaders, model, optimizers and loss function
        (
            train_loader,
            valid_loader,
            model,
            optimizer,
            scheduler,
            criterion,
        ) = make(args, train_data, valid_data)

        # display model details
        if desc:
            display_model_details(args, model)

        # define variables for ROC and loss
        best_score = 0
        best_loss = np.inf

        # instantiate training object
        use_rnn = True if args.data_preproc == "rnn" else False
        trainer = run.Run(
            model,
            device=args.device,
            criterion=criterion,
            optimizer=optimizer,
            use_focal_loss=args.focal_loss,
            use_rnn=use_rnn,
        )

        # dummy input, need it for saving the model to ONNX format
        input_size = (
            args.batch_size,
            1,
            args.signal_window_size,
            8,
            8,
        )
        dummy_input = torch.rand(*input_size)
        dummy_input = dummy_input.to(args.device)

        # tell wandb to watch what the model gets up to: gradients, weights, etc.
        wandb.watch(model, criterion, log="all", log_freq=10)
        # iterate through all the epochs
        for epoch in range(args.n_epochs):
            start_time = time.time()
            # train
            avg_loss = trainer.train(
                train_loader, epoch, print_every=args.train_print_every
            )
            train_loss.append(avg_loss)
            training_log(avg_loss, epoch, mode="Training")

            # evaluate
            avg_val_loss, preds, valid_labels = trainer.evaluate(
                valid_loader, print_every=args.valid_print_every
            )
            valid_loss.append(avg_val_loss)

            # step the scheduler
            scheduler.step(avg_val_loss)
            # print(f"Train losses: {train_loss}")
            # print(f"Valid losses: {valid_loss}")
            if args.add_tensorboard:
                writer.add_scalars(
                    f"{args.model_name}_signal_window_{args.signal_window_size}_lookahead_{args.label_look_ahead}",
                    {
                        "train_loss": avg_loss,
                        "valid_loss": avg_val_loss,
                    },
                    epoch + 1,
                )
                writer.close()
            # scoring
            roc_score = roc_auc_score(valid_labels, preds)
            thresh = 0.35
            f1 = f1_score(valid_labels, (preds > thresh).astype(int))

            # log the parameters in wandb
            training_log(avg_val_loss, epoch, roc_score, f1, mode="Validation")
            elapsed = time.time() - start_time

            LOGGER.info(
                f"Epoch: {epoch + 1}, \tavg train loss: {avg_loss:.4f}, \tavg validation loss: {avg_val_loss:.4f}"
            )
            LOGGER.info(
                f"Epoch: {epoch + 1}, \tROC-AUC score: {roc_score:.4f}, \ttime elapsed: {elapsed}"
            )

            if roc_score > best_score:
                best_score = roc_score
                LOGGER.info(
                    f"Epoch: {epoch + 1}, \tSave Best Score: {best_score:.4f} Model"
                )
                if not args.dry_run:
                    # save the model if best ROC is found
                    model_save_path = os.path.join(
                        model_ckpt_path,
                        f"{args.model_name}_lookahead_{args.label_look_ahead}_{args.data_preproc}.pth",
                    )
                    torch.save(
                        {"model": model.state_dict(), "preds": preds},
                        model_save_path,
                    )
                    LOGGER.info(f"Model saved to: {model_save_path}")

                    # export the model to ONNX to visualize the graph in netron
                    torch.onnx.export(
                        model,
                        dummy_input,
                        f"{args.model_name}_sws_{args.signal_window_size}_la_{args.label_look_ahead}.onnx",
                    )
                    wandb.save(
                        f"{args.model_name}_sws_{args.signal_window_size}_la_{args.label_look_ahead}.onnx"
                    )

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                LOGGER.info(
                    f"Epoch: {epoch + 1}, \tSave Best Loss: {best_loss:.4f} Model"
                )


if __name__ == "__main__":
    args, parser = TrainArguments().parse(verbose=True)

    config = dict(
        epochs=args.n_epochs,
        classes=2,
        signal_window_size=args.signal_window_size,
        label_look_ahead=args.label_look_ahead,
        data_preproc=args.data_preproc,
        architecture=args.model_name,
        learning_rate=args.lr,
        l2_reg=args.weight_decay,
    )
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
        test_datafile_name=f"test_data_lookahead_{args.label_look_ahead}_{args.data_preproc}.pkl",
        config=config,
    )
