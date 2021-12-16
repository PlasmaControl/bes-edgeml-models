import os
import time
import pickle
import argparse
from typing import Union

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import roc_auc_score

from data_preprocessing import *
from options.train_arguments import TrainArguments
from src import utils, run, dataset


# TODO: Take care of K-fold cross-validation and `kfold` and `n_folds` args.
def train_loop(args: argparse.Namespace, data_obj: object, test_datafile_name: str, fold: Union[int, None] = None,
        desc: bool = True, ) -> None:
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
    # containers to hold train and validation losses
    train_loss = []
    valid_loss = []
    test_data_path, model_ckpt_path = utils.create_output_paths(args, infer_mode=False)
    test_data_file = os.path.join(test_data_path, test_datafile_name)

    # add loss values to tensorboard
    if args.add_tensorboard:
        writer = SummaryWriter(
                log_dir=os.path.join(args.log_dir, "tensorboard", f"{args.model_name}{args.filename_suffix}", ))

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
    train_data, valid_data, _ = data_obj.get_data(shuffle_sample_indices=args.shuffle_sample_indices, fold=fold)

    # dump test data into to a file
    if not args.dry_run:
        with open(test_data_file, "w+b") as f:
            pickle.dump({"signals": valid_data[0], "labels": valid_data[1], "sample_indices": valid_data[2],
                    "window_start": valid_data[3], }, f, )

    # create image transforms
    if ((args.model_name.startswith("feature")) or (args.model_name.startswith("cnn")) or (
    args.model_name.startswith("rnn"))):
        transforms = None
    else:
        transforms = dataset.get_transforms(args)

    # create datasets
    train_dataset = dataset.ELMDataset(args, *train_data, logger=LOGGER, phase="training")

    valid_dataset = dataset.ELMDataset(args, *valid_data, logger=LOGGER, phase="validation", )

    # training and validation dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True, )

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=True, )

    # model
    model_cls = utils.create_model(args.model_name)
    model = model_cls(args)

    device = torch.device(args.device)  # "cuda" if torch.cuda.is_available() else "cpu")
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
                        args.batch_size, 1, args.signal_window_size, args.interpolate_size, args.interpolate_size,)
            elif args.data_preproc == "gradient":
                input_size = (args.batch_size, 6, args.signal_window_size, 8, 8,)
            elif args.data_preproc == "rnn":
                input_size = (args.batch_size, args.signal_window_size, 64)
            else:
                input_size = (args.batch_size, 1, args.signal_window_size, 8, 8,)
        x = torch.randn(*input_size)
        x = x.to(device)
        utils.model_details(model, x, input_size)
        # make torchviz visualisation of model.
        if args.viz == "show_autograd":
            utils.model_viz(model, x, show_autograd=True)
        elif args.viz:
            utils.model_viz(model, x, show_autograd=False)
        if args.add_tensorboard:
            writer.add_graph(model, x)

    # optimizer
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # get the lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2,
            verbose=True, )

    # loss function
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # define variables for ROC and loss
    best_score = 0
    best_loss = np.inf

    # instantiate training object
    use_rnn = True if args.data_preproc == "rnn" else False
    engine = run.Run(model, device=device, criterion=criterion, optimizer=optimizer, use_focal_loss=args.focal_loss,
            use_rnn=use_rnn, )

    # iterate through all the epochs
    for epoch in range(args.n_epochs):
        start_time = time.time()
        # train
        avg_loss = engine.train(train_loader, epoch, print_every=args.train_print_every)
        train_loss.append(avg_loss)

        # evaluate
        avg_val_loss, preds, valid_labels = engine.evaluate(valid_loader, print_every=args.valid_print_every)
        valid_loss.append(avg_val_loss)

        # step the scheduler
        scheduler.step(avg_val_loss)
        # print(f"Train losses: {train_loss}")
        # print(f"Valid losses: {valid_loss}")
        if args.add_tensorboard:
            writer.add_scalars(
                    f"{args.model_name}_signal_window_{args.signal_window_size}_lookahead_{args.label_look_ahead}",
                    {"train_loss": avg_loss, "valid_loss": avg_val_loss, }, epoch + 1, )
            writer.close()
        # scoring
        roc_score = roc_auc_score(valid_labels, preds)
        elapsed = time.time() - start_time

        LOGGER.info(f"Epoch: {epoch + 1}, \tavg train loss: {avg_loss:.4f}, \tavg validation loss: {avg_val_loss:.4f}")
        LOGGER.info(f"Epoch: {epoch + 1}, \tROC-AUC score: {roc_score:.4f}, \ttime elapsed: {elapsed}")

        if roc_score > best_score:
            best_score = roc_score
            LOGGER.info(f"Epoch: {epoch + 1}, \tSave Best Score: {best_score:.4f} Model")
            if not args.dry_run:
                # save the model if best ROC is found
                model_save_path = os.path.join(model_ckpt_path,
                        f"{args.model_name}_lookahead_{args.label_look_ahead}_{args.data_preproc}.pth", )
                torch.save({"model": model.state_dict(), "preds": preds}, model_save_path, )
                LOGGER.info(f"Model saved to: {model_save_path}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f"Epoch: {epoch + 1}, \tSave Best Loss: {best_loss:.4f} Model")

    # # save the predictions in the valid dataframe  # valid_folds["preds"] = torch.load(  #     os.path.join(  #         args.model_dir, f"{args.model_name}_fold{fold}_best_roc.pth"  #     ),  #     map_location=torch.device("cpu"),  # )["preds"]

    # return valid_folds


if __name__ == "__main__":
    args, parser = TrainArguments().parse(verbose=True)
    LOGGER = utils.get_logger(script_name=__name__,
            log_file=os.path.join(args.log_dir, f"output_logs_{args.model_name}{args.filename_suffix}.log", ), )
    data_cls = utils.create_data(args.data_preproc)
    data_obj = data_cls(args, LOGGER)
    train_loop(args, data_obj,
            test_datafile_name=f"test_data_lookahead_{args.label_look_ahead}_{args.data_preproc}.pkl", )
