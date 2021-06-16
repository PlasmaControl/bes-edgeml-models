import torch
from typing import Tuple
from torchinfo import summary
import data, config
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

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
        model_input_shape: Tuple = (4,1,8,8,8),
        signal_window_size: int = 8,
        learning_rate: float = .01):

        super(Autoencoder_PT, self).__init__()

        self.latent_dim = latent_dim
        self.encoder_hidden_layers = encoder_hidden_layers
        self.decoder_hidden_layers = decoder_hidden_layers
        self.model_input_shape = model_input_shape # (batch size, channels, signal window size, height, width)
        self.signal_window_size = signal_window_size # Initialized to 8 frames 
        self.relu_negative_slope = relu_negative_slope
        self.batch_size = model_input_shape[0]

        # 8x8x8 = 512 input features
        self.num_input_features = self.model_input_shape[1]
        for i in range(2, len(self.model_input_shape)):
            self.num_input_features *= self.model_input_shape[i]
        # print(f'total number of features: {num_features}')

        self.flatten = torch.nn.Flatten()
        self.encoder = torch.nn.Sequential()
        self.create_encoder()
        self.decoder = torch.nn.Sequential()
        self.create_decoder()

        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)

        return

    def create_encoder(self): #(250,100)
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

    # Train this autoencoder model
    def train_model(self, dataloader: DataLoader, epochs: int = 3, print_output: bool = True):
        if print_output:
            print('Beginning Training Model')

        # loop over the dataset multiple times
        for epoch in range(epochs):  
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()

                if(print_output):
                    # print every 2000 mini-batches
                    if i % 2000 == 1999:    
                        print('Epoch %d, loss after sample #%5d: %.3f' %
                              (epoch + 1, i + 1, running_loss / 2000))
                        running_loss = 0.0

        
        if(print_output):
            print('Finished Training')
        return

if __name__== '__main__':
    model = Autoencoder_PT(32, 
        encoder_hidden_layers = (250,100), 
        decoder_hidden_layers = (100,250))

    model = model.to(device)

    input_size = (4,1,8,8,8)
    summary(model, input_size)

    fold = 1
    data_ = data.Data(kfold=True, balance_classes=config.balance_classes)
    train_data, test_data, _ = data_.get_data(shuffle_sample_indices=True, fold=fold)
    
    train_dataset = data.ELMDataset(
        *train_data,
        config.signal_window_size,
        config.label_look_ahead,
        stack_elm_events=False,
        transform=None,
        for_autoencoder = True
    )

    # test_dataset = data.ELMDataset(
    #     *test_data,
    #     config.signal_window_size,
    #     config.label_look_ahead,
    #     stack_elm_events=False,
    #     transform=None,
    #     for_autoencoder = True
    # )

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Train the model
    model.train_model(train_dataloader, epochs = 5)

    # Save the model
    model_save_path = './models/trained_model.pth'
    torch.save(model.state_dict(), model_save_path)
    
        

