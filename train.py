import os
import time
import pickle
import argparse
from typing import Union

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

from options.train_arguments import TrainArguments
from src import config, data, utils, run, cnn_feature_model, model


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    dataloader: torch.utils.data.DataLoader,
):
    # learning rate scheduler
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            verbose=True,
            eps=1e-6,
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=0, verbose=False
        )
    elif scheduler_name == "CyclicLR":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1e-4,
            max_lr=1e-3,
            mode="triangular",
            cycle_momentum=False,
            verbose=False,
        )
    elif scheduler_name == "CyclicLR2":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1e-4,
            max_lr=1e-3,
            mode="triangular2",
            cycle_momentum=False,
            verbose=False,
        )
    elif scheduler_name == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            epochs=args.epochs,
            steps_per_epoch=len(dataloader),
            max_lr=1e-3,
            pct_start=0.2,
            anneal_strategy="cos",
        )
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.5, verbose=True
        )

    return scheduler


def train_loop(
    data_obj: data.Data,
    test_datafile_name: str,
    model_class=cnn_feature_model.CNNModel,
    kfold: bool = False,
    fold: Union[int, None] = None,
    desc: bool = True,
):
    # TODO: Implement K-fold cross-validation
    if kfold and (fold is None):
        raise Exception(
            f"K-fold cross validation is passed but fold index in range [0, {args.folds}) is not specified."
        )
    if (not kfold) and (fold is not None):
        LOGGER.info(
            f"K-fold is set to {kfold} but fold index is passed!"
            " Proceeding without using K-fold."
        )
        fold = None

    # test data file path
    test_data_file = os.path.join(args.data_dir, test_datafile_name)

    LOGGER.info("-" * 60)
    if args.balance_classes:
        LOGGER.info("Training with balanced classes.")
    else:
        LOGGER.info("Training using unbalanced (original) classes.")

    LOGGER.info(f"Test data will be saved to: {test_data_file}")
    LOGGER.info("-" * 30)
    LOGGER.info(f"       Training fold: {fold}       ")
    LOGGER.info("-" * 30)

    # turn off model details for subsequent folds/epochs
    if fold is not None:
        if fold >= 1:
            desc = False

    # create train, valid and test data
    train_data, valid_data, test_data = data_obj.get_data(
        shuffle_sample_indices=True, fold=fold
    )

    # dump test data into to a file
    with open(test_data_file, "wb") as f:
        pickle.dump(
            {
                "signals": test_data[0],
                "labels": test_data[1],
                "sample_indices": test_data[2],
                "window_start": test_data[3],
            },
            f,
        )

    # create image transforms
    if type(model_class).__name__ in ["FeatureModel", "CNNModel"]:
        transforms = None
    else:
        transforms = data.get_transforms()

    # create datasets
    train_dataset = data.ELMDataset(
        *train_data,
        args.signal_window_size,
        args.label_look_ahead,
        stack_elm_events=args.stack_elm_events,
        add_noise=args.add_noise,
        transform=transforms,
    )

    valid_dataset = data.ELMDataset(
        *valid_data,
        args.signal_window_size,
        args.label_look_ahead,
        stack_elm_events=args.stack_elm_events,
        transform=transforms,
    )

    # training and validation dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # model
    model = model_class()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_name = type(model).__name__
    LOGGER.info("-" * 50)
    LOGGER.info(f"       Training with model: {model_name}       ")
    LOGGER.info("-" * 50)

    # display model details
    if desc:
        if config.stack_elm_events and model_name == "StackedELMModel":
            input_size = (config.batch_size, 1, config.size, config.size)
        else:
            input_size = (
                config.batch_size,
                1,
                config.signal_window_size,
                8,
                8,
            )
        x = torch.rand(*input_size)
        x = x.to(device)
        cnn_feature_model.model_details(model, x, input_size)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        amsgrad=False,
    )

    # get the lr scheduler
    scheduler = get_lr_scheduler(
        optimizer, scheduler_name=config.scheduler, dataloader=train_loader
    )

    # loss function
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # define variables for ROC and loss
    best_score = 0
    best_loss = np.inf

    # instantiate training object
    engine = run.Run(
        model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        use_focal_loss=True,
    )

    # iterate through all the epochs
    for epoch in range(config.epochs):
        start_time = time.time()

        if config.scheduler in [
            "CosineAnnealingLR",
            "CyclicLR",
            "CyclicLR2",
            "OneCycleLR",
        ]:
            # train
            avg_loss = engine.train(
                train_loader, epoch, scheduler=scheduler, print_every=5000
            )

            # evaluate
            avg_val_loss, preds, valid_labels = engine.evaluate(
                valid_loader, print_every=2000
            )
            scheduler = get_lr_scheduler(
                optimizer,
                scheduler_name=config.scheduler,
                dataloader=train_loader,
            )
        else:
            # train
            avg_loss = engine.train(train_loader, epoch, print_every=5000)

            # evaluate
            avg_val_loss, preds, valid_labels = engine.evaluate(
                valid_loader, print_every=2000
            )

            # step the scheduler
            if config.scheduler == "ReduceLROnPlateau":
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        # scoring
        roc_score = roc_auc_score(valid_labels, preds)
        elapsed = time.time() - start_time

        LOGGER.info(
            f"Epoch: {epoch + 1}, \tavg train loss: {avg_loss:.4f}, \tavg validation loss: {avg_val_loss:.4f}"
        )
        LOGGER.info(
            f"Epoch: {epoch +1}, \tROC-AUC score: {roc_score:.4f}, \ttime elapsed: {elapsed}"
        )

        # save the model if best ROC is found
        model_save_path = os.path.join(
            config.model_dir,
            f"{model_name}_fold{fold}_best_roc_{config.data_mode}_lookahead_{config.label_look_ahead}_noise_{config.stdev}.pth",
        )
        if roc_score > best_score:
            best_score = roc_score
            LOGGER.info(
                f"Epoch: {epoch+1}, \tSave Best Score: {best_score:.4f} Model"
            )
            torch.save(
                {"model": model.state_dict(), "preds": preds},
                model_save_path,
            )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(
                f"Epoch: {epoch+1}, \tSave Best Loss: {best_loss:.4f} Model"
            )
        LOGGER.info(f"Model saved to: {model_save_path}")
    # # # save the predictions in the valid dataframe
    # # valid_folds["preds"] = torch.load(
    # #     os.path.join(
    # #         config.model_dir, f"{config.model_name}_fold{fold}_best_roc.pth"
    # #     ),
    # #     map_location=torch.device("cpu"),
    # # )["preds"]

    # # return valid_folds


if __name__ == "__main__":
    args, parser = TrainArguments().parse()
    utils.test_args_compat(args, parser)
    LOGGER = utils.get_logger(
        script_name=__name__,
        log_file=f"output_logs_{args.data_mode}_noise_{args.sigma}.log",
    )
    data_obj = data.Data(args)
    print(data_obj)
    # train_loop(
    #     data_obj,
    #     model_class=cnn_feature_model.FeatureModel,
    #     test_datafile_name=f"test_data_{config.data_mode}_lookahead_{config.label_look_ahead}.pkl",
    # )
