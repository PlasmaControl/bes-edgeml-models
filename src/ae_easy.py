import torch
from typing import Tuple
from torchinfo import summary
import data, config
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Autoencoder class
class Autoencoder_easy(torch.nn.Module):
    # Constructor - sets up encoder and decoder layers
    def __init__(self,
        latent_dim: int,  
        signal_window_shape: Tuple = (1,8,8,8),
        signal_window_size: int = 8,
        relu_negative_slope: float = .01,
        learning_rate: float = .0001,
        ):

        super(Autoencoder_easy, self).__init__()

        self.latent_dim = latent_dim
        self.signal_window_shape = signal_window_shape # (channels, signal window size, height, width)
        self.signal_window_size = signal_window_size # Initialized to 8 frames 
        self.relu_negative_slope = relu_negative_slope


        self.flatten = torch.nn.Flatten()
        self.layer = torch.nn.Linear(512, self.latent_dim)
        # self.relu = torch.nn.LeakyReLU(negative_slope=self.relu_negative_slope)
        

        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=learning_rate, 
            momentum=0.9
            )
        

    
    # Forward pass of the autoencoder - returns the reshaped output of net
    def forward(self, x):
        shape = x.shape
        # print(shape)
        x = self.flatten(x)
        x = self.layer(x)
        x = self.relu(x)
        reconstructed = x.view(*shape)
        # print(reconstructed.shape)
        return reconstructed

    @staticmethod
    def train_loop(model, dataloader: DataLoader, print_output: bool = True):
        model.train()
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = model.loss(pred, y)

            # Backpropagation
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            if batch % 1000 == 0:
                loss, current = loss.item(), batch * len(X)
                if(print_output):
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    @staticmethod
    def test_loop(model, dataloader: DataLoader, print_output: bool = True):
        size = len(dataloader.dataset)
        test_loss, correct = 0, 0

        model.eval()
        
        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                test_loss += model.loss(pred, y).item()

        test_loss /= size

        if(print_output):
            print(f"Test Error:\n Avg loss: {test_loss:>8f} \n")

        return test_loss


    @staticmethod
    def train_model(
        model,  
        train_dataloader: DataLoader, 
        test_dataloader: DataLoader,
        epochs: int = 10, 
        print_output: bool = True):

        all_losses = []

        for t in range(epochs):
            if(print_output):
                print(f"Epoch {t+1}\n-------------------------------")
            model.train_loop(model, train_dataloader,)
            epoch_loss = model.test_loop(model, test_dataloader)

            all_losses.append(epoch_loss)

            # Change optimizer learning rate
            # model.scheduler.step(epoch_loss)
        
        if(print_output):
            print("Done Training!")

        return all_losses


def plot_loss(losses):
    plt.plot(losses)
    plt.title('Training Loss')
    plt.ylabel('Avg Loss')
    plt.xlabel('epochs')
    # plt.show()
    plt.savefig('loss_plot.png')


if __name__== '__main__':
    model = Autoencoder_easy(512)

    model = model.to(device)
    batch_size = 4

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

    # print(f'Length of train dataset: {train_dataset.__len__()}')

    test_dataset = data.ELMDataset(
        *test_data,
        config.signal_window_size,
        config.label_look_ahead,
        stack_elm_events=False,
        transform=None,
        for_autoencoder = True
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Train the model
    losses = Autoencoder_easy.train_model(model, train_dataloader, test_dataloader, epochs  = 10, print_output = True)
    plot_loss(losses)

    # Save the model - weights and structure
    model_save_path = './easy_model.pth'
    torch.save(model, model_save_path)
    
        

