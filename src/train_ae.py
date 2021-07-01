import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import pickle
import os
from matplotlib import pyplot as plt
from autoencoder import AE_simple, device
import numpy as np

from itertools import product
from collections import namedtuple
from collections import OrderedDict

import config, data

# ------------------------------------------------------------------------------------------------

class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())

        runs = []

        for v in product(*params.values()):
            run = Run(*v)
            # print(run)
            runs.append(run)

        return runs

# Train function - this trains the passed in model by cycling 
# between the train_loop() and validation_loop()
def train(model: AE_simple, 
    train_dataloader: DataLoader, 
    test_dataloader: DataLoader, 
    optimizer,
    scheduler, 
    loss_fn,
    folder_name: str, 
    epochs: int = config.epochs,
    print_output: bool = True):
    
    run_name = model.latent_dim

    tb = SummaryWriter(log_dir = f'runs/{folder_name}/{run_name}')

    epoch_avg_losses = []
    all_losses = None

    for t in range(epochs):
        if(print_output):
            print(f"Epoch {t+1}\n-------------------------------")

        train_loop(model, train_dataloader, optimizer, loss_fn)

        if(t == epochs - 1): # Last epoch
            epoch_avg_loss, all_losses = validation_loop(model, test_dataloader, loss_fn, all_losses = True)
        else:
            epoch_avg_loss = validation_loop(model, test_dataloader, loss_fn)

        epoch_avg_losses.append(epoch_avg_loss)
        tb.add_scalar('Epoch Avg Loss', epoch_avg_loss, t + 1)

        # Change optimizer learning rate
        scheduler.step(epoch_avg_loss)
    
    if(print_output):
        print("Done Training!")

    return epoch_avg_losses, all_losses
    
def train_loop(model, dataloader: DataLoader, optimizer, loss_fn, print_output: bool = True):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y) # Average loss for the given batch

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # For every 1000th batch:
        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print (name, param.data)
            #         break
            # param = model.parameters()[0][0,0]
            if(print_output):
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def validation_loop(model, dataloader: DataLoader, loss_fn, all_losses: bool = False, print_output: bool = True):
    size = len(dataloader.dataset)
    test_loss = 0

    if(all_losses):
        all_epoch_losses = []

    model.eval()
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            cur_avg_batch_loss = loss_fn(pred, y).item()
            test_loss += cur_avg_batch_loss 

            if(all_losses):
                all_epoch_losses.append(cur_avg_batch_loss)

    test_loss /= size

    if(print_output):
        print(f"Validation Dataset Avg loss: {test_loss:>8f} \n")

    if(all_losses):
        return test_loss, all_epoch_losses
    else:
        return test_loss

def save_test_dataset(test_data, folder, filename):
    parent_path = f'test_datasets/{folder}'
    os.makedirs(parent_path, exist_ok = True)
    filepath = parent_path + f'/{filename}'

    # dump test data into to a file
    with open(filepath, "wb") as f:
        pickle.dump(
            {
                "signals": test_data[0],
                "labels": test_data[1],
                "sample_indices": test_data[2],
                "window_start": test_data[3],
            },
            f,
        )
    return

def save_model(model, folder, model_name):
    parent_path = f'trained_models/{folder}'
    os.makedirs(parent_path, exist_ok = True)
    model_save_path = parent_path + f'/{model_name}'

    torch.save(model, model_save_path)

def run_training(params: OrderedDict, save: bool = True):
    runs = RunBuilder.get_runs(params)
    folder = 'one_hidden_layer'

    for run in runs:
        model = AE_simple(run.latent)
        model = model.to(device)

        loss_fn = torch.nn.MSELoss()

        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config.learning_rate, 
            momentum=0.9, 
            weight_decay=config.l2_factor)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=2,
                verbose=True,
                eps=1e-6,
            )    

        # Get datasets and form dataloaders
        data_ = data.Data(kfold=False, balance_classes=config.balance_classes)
        train_data, valid_data, test_data = data_.get_data(shuffle_sample_indices=True) 

        save_test_dataset(test_data, folder, f'latent_{run.latent}')

        train_dataset = data.ELMDataset(
            *train_data,
            config.signal_window_size,
            config.label_look_ahead,
            stack_elm_events=False,
            transform=None,
            for_autoencoder = True
        )

        valid_dataset = data.ELMDataset(
            *valid_data,
            config.signal_window_size,
            config.label_look_ahead,
            stack_elm_events=False,
            transform=None,
            for_autoencoder = True
        )

        batch_size = config.batch_size
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        epoch_avg_losses, all_losses = train(model, 
            train_dataloader, 
            valid_dataloader, 
            optimizer, 
            scheduler, 
            loss_fn,
            folder_name = folder)

        # Save the model - weights and structure
        if(save):
            save_model(model, folder, f'simple_ae_latent_{run.latent}')

if __name__ == '__main__':
    params = OrderedDict(
        latent = [500, 400, 300, 200, 100, 50]
        )

    run_training(params)
