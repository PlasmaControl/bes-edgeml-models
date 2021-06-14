import torch
from typing import Tuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Autoencoder class
class Autoencoder_PT(torch.nn.Module):
    # Constructor - sets up encoder and decoder layers
    def __init__(self,
        latent_dim: int, 
        encoder_hidden_layers: Tuple,
        decoder_hidden_layers: Tuple, 
        relu_negative_slope: float = 0.0,
        model_input_shape: Tuple = (1,8,8,8),
        signal_window_size: int = 8):

        super(Autoencoder_PT, self).__init__()

        self.latent_dim = latent_dim
        self.encoder_hidden_layers = encoder_hidden_layers
        self.decoder_hidden_layers = decoder_hidden_layers
        self.model_input_shape = model_input_shape # (batch size, signal window size, height, width)
        self.signal_window_size = signal_window_size # Initialized to 8 frames 
        self.relu_negative_slope = relu_negative_slope

        # 8x8x8 = 512 input features
        self.num_input_features = self.model_input_shape[0]
        for i in range(1, len(self.model_input_shape)):
            self.num_input_features *= self.model_input_shape[i]
        # print(f'total number of features: {num_features}')

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
            self.encoder.add_module(f'Encoder Dense Layer {i+1}', d_layer)
            self.encoder.add_module(f'Encoder ReLU Layer {i+1}', torch.nn.ReLU())

        # Add latent dim layer after encoder hidden layers
        latent = torch.nn.Linear(self.encoder_hidden_layers[i], self.latent_dim)
        self.encoder.add_module(f'Latent Layer', latent)
        self.encoder.add_module(f'Latent ReLU Layer', torch.nn.ReLU())

        return

    def create_decoder(self):
        # Add the requested number of decoder hidden dense layers
        for i, layer_size in enumerate(self.decoder_hidden_layers):
            if i == 0:
                d_layer = torch.nn.Linear(self.latent_dim, self.decoder_hidden_layers[i])    
            else:
                d_layer = torch.nn.Linear(self.decoder_hidden_layers[i-1], self.decoder_hidden_layers[i])
            self.decoder.add_module(f'Decoder Dense Layer {i+1}', d_layer)
            self.decoder.add_module(f'Decoder ReLU Layer {i+1}', torch.nn.ReLU())

        # Add last layer after decoder hidden layers
        last = torch.nn.Linear(self.decoder_hidden_layers[i], self.num_input_features)
        self.decoder.add_module(f'Last Layer', last)
        self.decoder.add_module(f'Last ReLU Layer', torch.nn.ReLU())

        return

    # Forward pass of the autoencoder - returns the reshaped output of net
    def forward(self, x):
        x = self.flatten(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(self.model_input_shape)



if __name__== '__main__':
    model = Autoencoder_PT(32, 
        encoder_hidden_layers = (250,100), 
        decoder_hidden_layers = (100,250))
    
    print()
    print(model)
        

