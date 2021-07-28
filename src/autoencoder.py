import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import torch.nn.functional as F

import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Tuple
from collections import OrderedDict

import data, config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using {} device'.format(device))


# Flexible Autoencoder class
class Autoencoder(torch.nn.Module):
    # Constructor - sets up encoder and decoder layers
    def __init__(self,
                 latent_dim: int,
                 encoder_hidden_layers: list,
                 decoder_hidden_layers: list,
                 batch_size: int = config.batch_size,
                 num_channels: int = 1,
                 frames_per_window: int = config.signal_window_size,
                 relu_negative_slope: float = 0.1,
                 name: str = None):

        super(Autoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.encoder_hidden_layers = encoder_hidden_layers
        self.decoder_hidden_layers = decoder_hidden_layers

        # (channels, signal window size, height, width)
        self.input_shape = (batch_size, num_channels, frames_per_window, 8, 8)
        self.frames_per_window = frames_per_window  # Initialized to 8 frames

        self.relu_negative_slope = relu_negative_slope

        # batch x 1 x 8 x 8 x 8 = 512 input features per item in batch 
        self.num_input_features = int(torch.numel(torch.randn(self.input_shape)) / batch_size)

        self.flatten = torch.nn.Flatten()
        self.layers = self.create_layers()

        self.model = torch.nn.Sequential(self.layers)

        if (name is None):
            self.name = self._get_name()
        else:
            self.name = name

        return

    def create_layers(self):
        # all layers
        layers = OrderedDict()

        # ENCODER -----------------------------------------------------------------------------------------
        for i, layer_size in enumerate(self.encoder_hidden_layers):
            if i == 0:
                d_layer = torch.nn.Linear(self.num_input_features, self.encoder_hidden_layers[i])
            else:
                d_layer = torch.nn.Linear(self.encoder_hidden_layers[i - 1], self.encoder_hidden_layers[i])

            # Add fully connected, then dropout, then relu layers to encoder
            layers[f'Encoder Linear Layer {i + 1}'] = d_layer
            # layers[f'Encoder Dropout Layer {i+1}'] = torch.nn.Dropout(p=self.dropout_rate))
            layers[f'Encoder ReLU Layer {i + 1}'] = torch.nn.LeakyReLU(negative_slope=self.relu_negative_slope)

        # Add latent dim layer after encoder hidden layers
        latent = torch.nn.Linear(self.encoder_hidden_layers[-1], self.latent_dim)
        layers[f'Latent Linear Layer'] = latent
        # layers[f'Latent ReLU Layer'] = torch.nn.LeakyReLU(negative_slope=self.relu_negative_slope)

        # DECODER -----------------------------------------------------------------------------------------
        for i, layer_size in enumerate(self.decoder_hidden_layers):
            if i == 0:
                d_layer = torch.nn.Linear(self.latent_dim, self.decoder_hidden_layers[i])
            else:
                d_layer = torch.nn.Linear(self.decoder_hidden_layers[i - 1], self.decoder_hidden_layers[i])

            # Add fully connected, then dropout, then relu layers to encoder
            layers[f'Decoder Linear Layer {i + 1}'] = d_layer
            # layers[f'Encoder Dropout Layer {i+1}'] = torch.nn.Dropout(p=self.dropout_rate))
            layers[f'Decoder ReLU Layer {i + 1}'] = torch.nn.LeakyReLU(negative_slope=self.relu_negative_slope)

        # Add last layer after decoder hidden layers
        last = torch.nn.Linear(self.decoder_hidden_layers[-1], self.num_input_features)
        layers[f'Last Linear Layer'] = last
        # layers[f'Latent ReLU Layer'] = torch.nn.LeakyReLU(negative_slope=self.relu_negative_slope)

        return layers

    # Forward pass of the autoencoder - returns the reshaped output of net
    def forward(self, x):
        shape = x.shape
        # print(f'Before flatten: {x.shape}')
        x = self.flatten(x)
        # print(f'After flatten: {x.shape}')

        reconstructed = self.model(x)
        reconstructed = reconstructed.view(*shape)
        # print(reconstructed.shape)

        return reconstructed

    def _get_name(self):
        # n = type(self).__name__ + f'_{2 * len(self.encoder_hidden_layers) + 1}_hidden_{self.latent_dim}_latent'
        s = type(self).__name__ + '_'
        for i in self.encoder_hidden_layers:
            s += str(i) + '_'

        s += str(self.latent_dim) + '_'

        for i in range(len(self.decoder_hidden_layers)):
            if i == len(self.decoder_hidden_layers) - 1:
                s += str(self.decoder_hidden_layers[i])
            else:
                s += str(self.decoder_hidden_layers[i]) + '_'

        return s


# Simple/easy Autoencoder class
class Conv_AE(torch.nn.Module):
    # Constructor - sets up encoder and decoder layers
    def __init__(self,
                 latent_dim: int,
                 input_channels: int = 1,
                 kernel_size=2,
                 num_filters=4,
                 frames_per_window: int = config.signal_window_size,
                 batch_size=config.batch_size,
                 relu_negative_slope: float = 0.1,
                 name: str = None
                 ):
        super(Conv_AE, self).__init__()

        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.num_filters = num_filters

        # (channels, signal window size, height, width)
        self.input_shape = (batch_size, input_channels, frames_per_window, 8, 8)

        self.frames_per_window = frames_per_window  # Initialized to 8 frames
        self.relu_negative_slope = relu_negative_slope

        self.conv1 = torch.nn.Conv3d(input_channels,
                                     num_filters,
                                     kernel_size=(frames_per_window, kernel_size, kernel_size))
        self.flatten = torch.nn.Flatten()

        # Calculate the input size for linear layer
        temp_in = torch.rand(*self.input_shape)
        temp_out = self.flatten(self.conv1(temp_in))
        self.linear_in = temp_out.shape[1]

        self.fc1 = torch.nn.Linear(self.linear_in, self.latent_dim)
        self.fc2 = torch.nn.Linear(self.latent_dim, self.linear_in)

        self.t_conv1 = torch.nn.ConvTranspose3d(num_filters,
                                                input_channels,
                                                kernel_size=(frames_per_window, kernel_size, kernel_size))

        if name is None:
            self.name = self._get_name()
        else:
            self.name = name

    # Forward pass of the autoencoder - returns the reshaped output of net
    def forward(self, x):
        input_shape = x.shape
        print(f'Input shape {input_shape}')

        x = F.leaky_relu(self.conv1(x), negative_slope=self.relu_negative_slope)
        temp_shape = x.shape
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)

        x = F.leaky_relu(self.fc1(x), negative_slope=self.relu_negative_slope)
        print(x.shape)

        x = F.leaky_relu(self.fc2(x), negative_slope=self.relu_negative_slope)
        print(x.shape)

        x = torch.reshape(x, temp_shape)
        print(x.shape)

        x = self.t_conv1(x)
        print(f'Output shape {x.shape}')
        return x

    def _get_name(self):
        # n = type(self).__name__ + f'_{2 * len(self.encoder_hidden_layers) + 1}_hidden_{self.latent_dim}_latent'
        s = type(self).__name__ + '_'
        s += f'latent_{self.latent_dim}_'
        s += f'filters_{self.num_filters}_'
        s += f'kernel_{self.kernel_size}'

        return s


# This train function is just for quick debugging - actual train function is in train_ae.py
def train(model,
          train_dataloader: DataLoader,
          valid_dataloader: DataLoader,
          optimizer,
          scheduler,
          loss_fn,
          epochs: int = config.epochs,
          print_output: bool = True):
    tb = SummaryWriter(log_dir=f'outputs/tensorboard/conv')

    avg_training_losses = []
    avg_validation_losses = []

    for t in range(epochs):
        if print_output:
            print(f"Epoch {t + 1}\n-------------------------------")

        avg_train_loss = train_loop(model, train_dataloader, optimizer, loss_fn)

        avg_validation_loss = validation_loop(model, valid_dataloader, loss_fn)

        avg_training_losses.append(avg_train_loss)
        avg_validation_losses.append(avg_validation_loss)

        tb.add_scalar('Training: Average Sample Loss vs. Epochs', avg_train_loss, t + 1)
        tb.add_scalar('Validation: Average Sample Loss vs. Epochs', avg_validation_loss, t + 1)

        # Change optimizer learning rate
        # scheduler.step(epoch_avg_loss)

    if print_output:
        print("Done Training!")

    return avg_training_losses, avg_validation_losses


# Train loop for quick debugging
def train_loop(model, dataloader: DataLoader, optimizer, loss_fn, print_output: bool = True):
    model.train()
    total_loss = 0

    # Sample windows in dataloader = batch_size * len(dataloader)
    samples_in_dataset = len(dataloader.dataset)
    batches_in_dataloader = len(dataloader)
    batch_size = math.ceil(samples_in_dataset / batches_in_dataloader)

    if print_output:
        print('Batch size:', batch_size)
        print('Number of samples in Train Dataset:', samples_in_dataset)
        print('Number of batches in Train Dataloader:', batches_in_dataloader)

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)  # Average loss for the given batch
        total_loss += loss.item()

        # if(len(X) < batch_size):
        #     print(batch, batch * batch_size, len(X))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # For every 1000th batch:
        if (batch + 1) % 1000 == 0:
            loss, current = loss.item(), (batch + 1) * batch_size
            if print_output:
                print(f"loss: {loss:>7f}  [{current:>5d}/{samples_in_dataset:>5d}]")

    avg_sample_loss = total_loss / samples_in_dataset

    if print_output:
        print(f"Training Avg. Sample loss: {avg_sample_loss:>8f}")

    # Return the average sample loss 
    return avg_sample_loss


# Validation loop for quick debugging
def validation_loop(model, dataloader: DataLoader, loss_fn, print_output: bool = True):
    batches_in_dataloader = len(dataloader)
    samples_in_dataset = len(dataloader.dataset)

    validation_loss = 0

    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            avg_batch_loss = loss_fn(pred, y).item()
            validation_loss += avg_batch_loss

    avg_sample_loss = validation_loss / samples_in_dataset

    if print_output:
        print(f"Validation Avg. Sample loss: {avg_sample_loss:>8f} \n")

    # Return the average sample loss 
    return avg_sample_loss


if __name__ == '__main__':
    model = Conv_AE(latent_dim=32,
                    kernel_size=3,
                    num_filters=10)

    model = model.to(device)
    # print(model.name)
    loss_fn = torch.nn.MSELoss(reduction='mean')

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
    data_ = data.Data(kfold=False, balance_classes=config.balance_classes, normalize=True)
    train_data, valid_data, test_data = data_.get_data(shuffle_sample_indices=True)

    train_dataset = data.ELMDataset(
        *train_data,
        config.signal_window_size,
        config.label_look_ahead,
        stack_elm_events=False,
        transform=None,
        for_autoencoder=True
    )

    valid_dataset = data.ELMDataset(
        *valid_data,
        config.signal_window_size,
        config.label_look_ahead,
        stack_elm_events=False,
        transform=None,
        for_autoencoder=True
    )

    batch_size = config.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # x,y = next(iter(train_dataloader))
    # pred = model(x)

    # train_avg_losses, validation_avg_losses = train(model,
    #                                                 train_dataloader,
    #                                                 valid_dataloader,
    #                                                 optimizer,
    #                                                 scheduler,
    #                                                 loss_fn)
    #
    # plt.plot(train_avg_losses)
    # plt.show()
    #
    # torch.save(model, './outputs/trained_models/conv/test_ae.pth')
