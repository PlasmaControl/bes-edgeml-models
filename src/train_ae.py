import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import pickle
import os, math
from matplotlib import pyplot as plt
from autoencoder import Autoencoder, device
import numpy as np

from itertools import product
from collections import namedtuple
from collections import OrderedDict

import config, data

# ------------------------------------------------------------------------------------------------

class RunBuilder():
    @staticmethod
    def get_runs(params: OrderedDict):
        Run = namedtuple('Run', params.keys())

        runs = []

        for v in product(*params.values()):
            run = Run(*v)
            # print(run)
            runs.append(run)

        return runs

# Train function - this trains the passed in model by cycling 
# between the train_loop() and validation_loop()
def train(model: Autoencoder, 
    train_dataloader: DataLoader, 
    valid_dataloader: DataLoader, 
    optimizer,
    scheduler, 
    loss_fn,
    tb: SummaryWriter,  
    epochs: int = config.epochs,
    print_output: bool = True):

    avg_training_losses = []
    avg_validation_losses = []

    for t in range(epochs):
        if(print_output):
            print(f"Epoch {t+1}\n-------------------------------")

        avg_train_loss = train_loop(model, train_dataloader, optimizer, loss_fn)

        avg_validation_loss = validation_loop(model, valid_dataloader, loss_fn)

        avg_training_losses.append(avg_train_loss)
        avg_validation_losses.append(avg_validation_loss)

        tb.add_scalar('Training: Average Sample Loss vs. Epochs', avg_train_loss, t + 1)
        tb.add_scalar('Validation: Average Sample Loss vs. Epochs', avg_validation_loss, t + 1)

        # Change optimizer learning rate
        # scheduler.step(epoch_avg_loss)
    
    # Add model graph
    # sample,label = next(iter(train_dataloader))
    # tb.add_graph(model, sample)

    if(print_output):
        print("Done Training!")

    return avg_training_losses, avg_validation_losses
    
def train_loop(model, dataloader: DataLoader, optimizer, loss_fn, print_output: bool = True):
    model.train()
    total_loss = 0

    # Sample windows in dataloader = batch_size * len(dataloader)
    samples_in_dataset = len(dataloader.dataset)
    batches_in_dataloader = len(dataloader)
    batch_size =  math.ceil(samples_in_dataset / batches_in_dataloader)

    if(print_output):
        print('Training batch size:', batch_size)
        print('# of samples in Train Dataset:', samples_in_dataset)
        print('# of batches in Train Dataloader:', batches_in_dataloader)

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y) # Average loss for the given batch
        total_loss += loss.item() * len(X)

        # if(len(X) < batch_size):
        #     print(batch, batch * batch_size, len(X))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # For every 1000th batch:
        if (batch + 1) % 1000 == 0:
            loss, current = loss.item(), (batch + 1) * batch_size
            if(print_output):
                print(f"loss: {loss:>7f}  [{current:>5d}/{samples_in_dataset:>5d}]")

    avg_sample_loss = total_loss / samples_in_dataset

    if(print_output):
        print(f"Training Avg. Sample loss: {avg_sample_loss:>8f}")

    return avg_sample_loss

def validation_loop(model, dataloader: DataLoader, loss_fn, all_losses: bool = False, print_output: bool = True):
    batches_in_dataloader = len(dataloader)
    samples_in_dataset = len(dataloader.dataset)

    total_validation_loss = 0

    model.eval()
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            avg_batch_loss = loss_fn(pred, y).item()
            total_validation_loss += (avg_batch_loss) * len(X)

    avg_sample_loss = total_validation_loss / samples_in_dataset

    if(print_output):
        print(f"Validation Avg. Sample loss: {avg_sample_loss:>8f} \n")
    
    # Return the average sample loss 
    return avg_sample_loss

def save_test_dataset(test_dataset, run_category, filename, folder = config.ae_test_datasets_dir):
    # folder = 'outputs/test_datasets'
    parent_path = folder + '/' + run_category # 'outputs/test_datasets/three_hidden'
    os.makedirs(parent_path, exist_ok = True)
    filepath = parent_path + f'/{filename}' # 'outputs/test_datasets/three_hidden/Autoencoder_400_300_400'

    # dump test data into to a file
    with open(filepath, "wb") as f:
        pickle.dump(
            {
                "signals": test_dataset[0],
                "labels": test_dataset[1],
                "sample_indices": test_dataset[2],
                "window_start": test_dataset[3],
            },
            f,
        )
    return

def save_model(model, run_category, folder = config.ae_trained_models_dir):
    # folder is the run name
    parent_path = folder + '/' + run_category 
    os.makedirs(parent_path, exist_ok = True)
    model_save_path = parent_path + f'/{model.name}' + '.pth'

    torch.save(model, model_save_path)

def run_training(params: OrderedDict, run_category: str = 'normalized_three_hidden_batch_32_100_elms', save: bool = True):
    # Get the runs
    runs = RunBuilder.get_runs(params)

    # For each run, create, train, and save model and metrics
    for run in runs:
        print(run)
        model = Autoencoder(
            run.latent, 
            run.encoder_hidden_layers, 
            run.decoder_hidden_layers)
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

        save_test_dataset(test_data, run_category, model.name)

        train_dataset = data.ELMDataset(
            *train_data,
            config.signal_window_size,
            config.label_look_ahead,
            stack_elm_events=False,
            transform=None,
            for_autoencoder = True,
            normalize = True
        )

        valid_dataset = data.ELMDataset(
            *valid_data,
            config.signal_window_size,
            config.label_look_ahead,
            stack_elm_events=False,
            transform=None,
            for_autoencoder = True,
            normalize = True
        )

        batch_size = config.batch_size
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        tb = SummaryWriter(log_dir = config.ae_tensorboard_dir + '/' + run_category + '/' + model.name)

        train_avg_losses, validation_avg_losses = train(model, 
            train_dataloader, 
            valid_dataloader, 
            optimizer, 
            scheduler, 
            loss_fn,
            tb)

        # analyze()

        # Save the model - weights and structure
        if(save):
            save_model(model, run_category)

if __name__ == '__main__':
    params = OrderedDict(
        latent = [400, 300, 200, 100, 64, 32, 16, 8, 4],
        encoder_hidden_layers = [[400]],
        decoder_hidden_layers = [[400]]
        )

    # print(RunBuilder.get_runs(params))
    run_training(params)
