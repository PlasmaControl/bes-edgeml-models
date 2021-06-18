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
        signal_window_shape: Tuple = (1,8,8,8),
        signal_window_size: int = 8,
        learning_rate: float = .01,
        l2_factor: float = 5e-3,
        dropout_rate: float = 0.3):

        super(Autoencoder_PT, self).__init__()

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

        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=l2_factor)

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
            self.encoder.add_module(f'Encoder ReLU Layer {i+1}', torch.nn.ReLU())

        # Add latent dim layer after encoder hidden layers
        latent = torch.nn.Linear(self.encoder_hidden_layers[i], self.latent_dim)
        self.encoder.add_module(f'Latent Layer', latent)
        self.encoder.add_module(f'Latent Dropout Layer {i+1}', torch.nn.Dropout(p=self.dropout_rate))
        self.encoder.add_module(f'Latent ReLU Layer', torch.nn.ReLU())

        return

    def create_decoder(self):
        # Add the requested number of decoder hidden dense layers
        for i, layer_size in enumerate(self.decoder_hidden_layers):
            if i == 0:
                d_layer = torch.nn.Linear(self.latent_dim, self.decoder_hidden_layers[i])    
            else:
                d_layer = torch.nn.Linear(self.decoder_hidden_layers[i-1], self.decoder_hidden_layers[i])

            # Add latent dim layer after encoder hidden layers
            self.decoder.add_module(f'Decoder Dense Layer {i+1}', d_layer)
            self.decoder.add_module(f'Decoder Dropout Layer {i+1}', torch.nn.Dropout(p=self.dropout_rate))
            self.decoder.add_module(f'Decoder ReLU Layer {i+1}', torch.nn.ReLU())

        # Add last layer after decoder hidden layers
        last = torch.nn.Linear(self.decoder_hidden_layers[i], self.num_input_features)
        self.decoder.add_module(f'Last Layer', last)
        self.decoder.add_module(f'Last Dropout Layer {i+1}', torch.nn.Dropout(p=self.dropout_rate))
        self.decoder.add_module(f'Last ReLU Layer', torch.nn.ReLU())

        return

    # Forward pass of the autoencoder - returns the reshaped output of net
    def forward(self, x):
        shape = x.shape
        print(shape)
        x = self.flatten(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(*shape)

    # Train the passed in autoencoder model
    @staticmethod
    def train_model(model, dataloader: DataLoader, epochs: int, print_output: bool = True):
        if print_output:
            print('Beginning Training Model')

        model.train()
        losses = []

        # loop over the dataset multiple times
        for epoch in range(epochs):
            epoch_total_loss = 0.0  
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                # print(i)
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                model.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = model.loss(outputs, labels)
                loss.backward()
                model.optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_total_loss += loss.item()

                # if(print_output):    
                    #print('Epoch %d, Average loss: %.3f' % (epoch + 1, epoch_total_loss / (i + 1)))
                    # print(loss)

                

                # print every 500 mini-batches
                mini_batch_size = 1000
                if i % mini_batch_size == mini_batch_size - 1:
                    losses.append(running_loss)
                    # if(print_output):    
                        # print('Epoch %d, loss after sample #%5d: %.3f' % (epoch + 1, i + 1, running_loss / mini_batch_size))
                    # running_loss = 0.0


            # After each epoch, calculate avg loss:
            #if(print_output):    
                #print('Epoch %d, Average loss: %.3f' % (epoch + 1, epoch_total_loss / (i + 1)))
                # print(loss)
        
        if(print_output):
            print('Finished Training')
        return losses

if __name__== '__main__':
    model = Autoencoder_PT(64, 
        encoder_hidden_layers = (250,100), 
        decoder_hidden_layers = (100,250))

    model = model.to(device)
    batch_size = 4

    # input_size = (1,1,8,8,8)
    # summary(model, input_size)

    fold = 1
    data_ = data.Data(kfold=True, balance_classes=config.balance_classes)
    train_data, _, _ = data_.get_data(shuffle_sample_indices=True, fold=fold)
    
    train_dataset = data.ELMDataset(
        *train_data,
        config.signal_window_size,
        config.label_look_ahead,
        stack_elm_events=False,
        transform=None,
        for_autoencoder = True
    )

    print(f'Length of train dataset: {train_dataset.__len__()}')

    # test_dataset = data.ELMDataset(
    #     *test_data,
    #     config.signal_window_size,
    #     config.label_look_ahead,
    #     stack_elm_events=False,
    #     transform=None,
    #     for_autoencoder = True
    # )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Train the model
    Autoencoder_PT.train_model(model, train_dataloader, epochs = 1, print_output = True)

    # Save the model - weights and structure
    model_save_path = './models/trained_model.pth'
    torch.save(model, model_save_path)
    
        

