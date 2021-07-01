import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from collections import OrderedDict

import data, config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Flexible Autoencoder class
class Autoencoder(torch.nn.Module):
    # Constructor - sets up encoder and decoder layers
    def __init__(self,
        latent_dim: int, 
        encoder_hidden_layers: list,
        decoder_hidden_layers: list,
        batch_size: int = 4,
        num_channels: int = 1,
        frames_per_window: int = 8,
        relu_negative_slope: float = 0.1,
        learning_rate: float = .0001,
        l2_factor: float = 5e-3,
        dropout_rate: float = 0.3):

        super(Autoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.encoder_hidden_layers = encoder_hidden_layers
        self.decoder_hidden_layers = decoder_hidden_layers

        # (channels, signal window size, height, width) - (4,1,8,8,8)
        self.signal_window_shape = (batch_size, num_channels, frames_per_window, 8, 8)
        self.frames_per_window = frames_per_window # Initialized to 8 frames

        self.relu_negative_slope = relu_negative_slope
        self.dropout_rate = dropout_rate

        # batch x 1 x 8 x 8 x 8 = 512 input features per item in batch 
        self.num_input_features = int(torch.numel(torch.randn(self.signal_window_shape)) / batch_size)

        self.flatten = torch.nn.Flatten()
        self.layers = self.create_layers()

        self.model = torch.nn.Sequential(self.layers)
        
        return

    def create_layers(self):
        # all layers
        layers = OrderedDict()

        # ENCODER -----------------------------------------------------------------------------------------
        for i, layer_size in enumerate(self.encoder_hidden_layers):
            if i == 0:
                d_layer = torch.nn.Linear(self.num_input_features , self.encoder_hidden_layers[i])    
            else:
                d_layer = torch.nn.Linear(self.encoder_hidden_layers[i-1], self.encoder_hidden_layers[i])

            # Add fully connected, then dropout, then relu layers to encoder
            layers[f'Encoder Linear Layer {i+1}'] = d_layer
            # layers[f'Encoder Dropout Layer {i+1}'] = torch.nn.Dropout(p=self.dropout_rate))
            layers[f'Encoder ReLU Layer {i+1}'] = torch.nn.LeakyReLU(negative_slope=self.relu_negative_slope)

        # Add latent dim layer after encoder hidden layers
        latent = torch.nn.Linear(self.encoder_hidden_layers[-1], self.latent_dim)
        layers[f'Latent Linear Layer'] = latent
        # layers[f'Latent ReLU Layer'] = torch.nn.LeakyReLU(negative_slope=self.relu_negative_slope)

        # DECODER -----------------------------------------------------------------------------------------
        for i, layer_size in enumerate(self.decoder_hidden_layers):
            if i == 0:
                d_layer = torch.nn.Linear(self.latent_dim, self.decoder_hidden_layers[i])    
            else:
                d_layer = torch.nn.Linear(self.decoder_hidden_layers[i-1], self.encoder_hidden_layers[i])

            # Add fully connected, then dropout, then relu layers to encoder
            layers[f'Decoder Linear Layer {i+1}'] = d_layer
            # layers[f'Encoder Dropout Layer {i+1}'] = torch.nn.Dropout(p=self.dropout_rate))
            layers[f'Decoder ReLU Layer {i+1}'] = torch.nn.LeakyReLU(negative_slope=self.relu_negative_slope)

        # Add last layer after decoder hidden layers
        last = torch.nn.Linear(self.decoder_hidden_layers[-1], self.num_input_features)
        layers[f'Last Linear Layer'] = last
        # layers[f'Latent ReLU Layer'] = torch.nn.LeakyReLU(negative_slope=self.relu_negative_slope)

        return layers

    # Forward pass of the autoencoder - returns the reshaped output of net
    def forward(self, x):
        shape = x.shape
        # print(shape)
        x = self.flatten(x)
        # print(x.shape)

        reconstructed = self.model(x)
        reconstructed = reconstructed.view(*shape)
        # print(reconstructed.shape)

        return reconstructed

# Simple/easy Autoencoder class
class AE_simple(torch.nn.Module):
    # Constructor - sets up encoder and decoder layers
    def __init__(self,
        latent_dim: int, 
        relu_negative_slope: float = 0.1,
        signal_window_shape: Tuple = (1,8,8,8),
        signal_window_size: int = 8,
        learning_rate: float = .0001,
        l2_factor: float = 5e-3,
        dropout_rate: float = 0.3):

        super(AE_simple, self).__init__()

        self.latent_dim = latent_dim

        self.signal_window_shape = signal_window_shape # (channels, signal window size, height, width)
        self.signal_window_size = signal_window_size # Initialized to 8 frames 
        self.relu_negative_slope = relu_negative_slope
        self.dropout_rate = dropout_rate

        # 1x8x8x8 = 512 input features
        self.num_input_features = self.signal_window_shape[0]
        for i in range(1, len(self.signal_window_shape)):
            self.num_input_features *= self.signal_window_shape[i]
        # print(f'total number of features: {self.num_input_features}')

        self.flatten = torch.nn.Flatten()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(512, self.latent_dim),
            torch.nn.LeakyReLU(negative_slope = self.relu_negative_slope)
            )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 512)
            )

    # Forward pass of the autoencoder - returns the reshaped output of net
    def forward(self, x):
        input_shape = x.shape
        x = self.flatten(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        reconstructed = decoded.view(*input_shape)
        # print(reconstructed.shape)
        return reconstructed


if __name__ == '__main__':
    model = Autoencoder(100, [400], [400])
    summary(model)


    
        

