import os
import time
from typing import Union

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

import config, data, utils, run, cnn_feature_model


LOGGER = utils.get_logger(script_name=__name__, log_file="output_logs.log")


def train_loop(data_obj: data.Data, fold: Union[int, None], desc: bool = True):
    LOGGER.info("-" * 30)
    LOGGER.info(f"       Training fold: {fold}       ")
    LOGGER.info("-" * 30)

    # turn off model details for subsequent folds/epochs
    if fold is not None:
        if fold >= 1:
            desc = False

    # create train, valid and test data
    train_data, valid_data, _ = data_obj.get_data(
        shuffle_sample_indices=True, fold=fold
    )

    # create datasets
    train_dataset = data.ELMDataset(
        *train_data, config.signal_window_size, config.label_look_ahead
    )

    valid_dataset = data.ELMDataset(
        *valid_data, config.signal_window_size, config.label_look_ahead
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
    model = cnn_feature_model.FeatureModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # display model details
    if desc:
        input_size = (config.batch_size, 1, config.signal_window_size, 8, 8)
        x = torch.rand(*input_size)
        x = x.to(device)
        cnn_feature_model.model_details(model, x, input_size)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        amsgrad=False,
    )

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True, eps=1e-6
    )

    # loss function
    criterion = nn.BCEWithLogitsLoss()

    # define variables for ROC and loss
    best_score = 0
    best_loss = np.inf

    # instantiate training object
    engine = run.Run(
        model, device=device, criterion=criterion, optimizer=optimizer
    )

    # iterate through all the epochs
    for epoch in range(config.epochs):
        start_time = time.time()

        # train
        avg_loss = engine.train(train_loader, epoch, print_every=5000)

        # evaluate
        avg_val_loss, preds, valid_labels = engine.evaluate(
            valid_loader, print_every=2000
        )

        # step the scheduler
        scheduler.step(avg_val_loss)

        # scoring
        roc_score = roc_auc_score(valid_labels, preds)
        elapsed = time.time() - start_time

        LOGGER.info(
            f"Epoch: {epoch + 1}, \tavg train loss: {avg_loss:.4f}, \tavg validation loss: {avg_val_loss:.4f}"
        )
        LOGGER.info(
            f"Epoch: {epoch +1}, \tROC-AUC score: {roc_score:.4f}, \ttime elapsed: {elapsed}"
        )

        if roc_score > best_score:
            best_score = roc_score
            LOGGER.info(
                f"Epoch: {epoch+1}, \tSave Best Score: {best_score:.4f} Model"
            )
            torch.save(
                {"model": model.state_dict(), "preds": preds},
                os.path.join(
                    config.model_dir,
                    f"{config.model_name}_fold{fold}_best_roc.pth",
                ),
            )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(
                f"Epoch: {epoch+1}, \tSave Best Loss: {best_loss:.4f} Model"
            )
            # torch.save(
            #     {"model": model.state_dict(), "preds": preds},
            #     os.path.join(
            #         config.model_dir,
            #         f"{config.model_name}_fold{fold}_best_loss.pth",
            #     ),
            # )
    # # save the predictions in the valid dataframe
    # valid_folds["preds"] = torch.load(
    #     os.path.join(
    #         config.model_dir, f"{config.model_name}_fold{fold}_best_roc.pth"
    #     ),
    #     map_location=torch.device("cpu"),
    # )["preds"]

    # return valid_folds


if __name__ == "__main__":
    data_obj = data.Data(kfold=False)
    train_loop(data_obj, fold=None)
