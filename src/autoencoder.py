import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

import data, config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Flexible Autoencoder class
class Autoencoder(torch.nn.Module):
    # Constructor - sets up encoder and decoder layers
    def __init__(self,
        latent_dim: int, 
        encoder_hidden_layers: Tuple,
        decoder_hidden_layers: Tuple, 
        relu_negative_slope: float = 0.1,
        signal_window_shape: Tuple = (1,8,8,8),
        signal_window_size: int = 8,
        learning_rate: float = .0001,
        l2_factor: float = 5e-3,
        dropout_rate: float = 0.3):

        super(Autoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.encoder_hidden_layers = encoder_hidden_layers
        self.decoder_hidden_layers = decoder_hidden_layers
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
        self.encoder = torch.nn.Sequential()
        self.create_encoder()
        self.decoder = torch.nn.Sequential()
        self.create_decoder()
        
        return

    def create_encoder(self):
        # Add the requested number of encoder hidden dense layers + relu layers
        for i, layer_size in enumerate(self.encoder_hidden_layers):
            if i == 0:
                d_layer = torch.nn.Linear(self.num_input_features, self.encoder_hidden_layers[i])    
            else:
                d_layer = torch.nn.Linear(self.encoder_hidden_layers[i-1], self.encoder_hidden_layers[i])

            # Add fully connected, then dropout, then relu layers
            self.encoder.add_module(f'Encoder Dense Layer {i+1}', d_layer)
            self.encoder.add_module(f'Encoder Dropout Layer {i+1}', torch.nn.Dropout(p=self.dropout_rate))
            self.encoder.add_module(f'Encoder ReLU Layer {i+1}', torch.nn.LeakyReLU(negative_slope=self.relu_negative_slope))

        # Add latent dim layer after encoder hidden layers
        latent = torch.nn.Linear(self.encoder_hidden_layers[i], self.latent_dim)
        self.encoder.add_module(f'Latent Layer', latent)
        # self.encoder.add_module(f'Latent Dropout Layer {i+1}', torch.nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module(f'Latent ReLU Layer', torch.nn.LeakyReLU(negative_slope=self.relu_negative_slope))

        return

    def create_decoder(self):
        # Add the requested number of decoder hidden dense layers
        for i, layer_size in enumerate(self.decoder_hidden_layers):
            if i == 0:
                d_layer = torch.nn.Linear(self.latent_dim, self.decoder_hidden_layers[i])    
            else:
                d_layer = torch.nn.Linear(self.decoder_hidden_layers[i-1], self.decoder_hidden_layers[i])

            self.decoder.add_module(f'Decoder Dense Layer {i+1}', d_layer)
            self.decoder.add_module(f'Decoder Dropout Layer {i+1}', torch.nn.Dropout(p=self.dropout_rate))
            self.decoder.add_module(f'Decoder ReLU Layer {i+1}', torch.nn.LeakyReLU(negative_slope=self.relu_negative_slope))

        # Add last layer after decoder hidden layers
        last = torch.nn.Linear(self.decoder_hidden_layers[i], self.num_input_features)
        self.decoder.add_module(f'Last Layer', last)
        # self.decoder.add_module(f'Last Dropout Layer {i+1}', torch.nn.Dropout(p=self.dropout_rate))
        # self.decoder.add_module(f'Last ReLU Layer', torch.nn.ReLU())

        return

    # Forward pass of the autoencoder - returns the reshaped output of net
    def forward(self, x):
        shape = x.shape
        # print(shape)
        x = self.flatten(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        reconstructed = decoded.view(*shape)
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

# if __name__== '__main__':
#     batch_size = 4
#     learning_rate = .0001
#     l2_factor = 5e-3

#     loss_fn = torch.nn.MSELoss()

#     optimizer = torch.optim.SGD(
#         model.parameters(), 
#         lr=learning_rate, 
#         momentum=0.9, 
#         weight_decay=l2_factor)

#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             mode="min",
#             factor=0.5,
#             patience=2,
#             verbose=True,
#             eps=1e-6,
#         )

#     # Print out the model architecture
#     input_size = (4,1,8,8,8)
#     summary(model, input_size)

#     # Get datasets and form dataloaders
#     data_ = data.Data(kfold=False, balance_classes=config.balance_classes)
#     train_data, test_data, _ = data_.get_data(shuffle_sample_indices=True)
    
#     train_dataset = data.ELMDataset(
#         *train_data,
#         config.signal_window_size,
#         config.label_look_ahead,
#         stack_elm_events=False,
#         transform=None,
#         for_autoencoder = True
#     )

#     test_dataset = data.ELMDataset(
#         *test_data,
#         config.signal_window_size,
#         config.label_look_ahead,
#         stack_elm_events=False,
#         transform=None,
#         for_autoencoder = True
#     )

#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

#     # Train the model and plot loss
#     avg_losses, all_losses = train_model(model, 
#         train_dataloader, 
#         test_dataloader, 
#         optimizer, 
#         scheduler, 
#         loss_fn, 
#         epochs  = 1)

#     # print(all_losses)

#     plot(avg_losses, all_losses)

#     # Save the model - weights and structure
#     # model_save_path = './trained_models/simple_ae.pth'
#     # torch.save(model, model_save_path)
    
        

