import torch
from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np
import os

from autoencoder import Autoencoder, AE_simple, device
import data, config

# Get data and form dataset
data_ = data.Data(kfold=False, balance_classes=config.balance_classes)
train_data, test_data, _ = data_.get_data(shuffle_sample_indices=False)

train_dataset = data.ELMDataset(
        *train_data,
        config.signal_window_size,
        config.label_look_ahead,
        stack_elm_events=False,
        transform=None,
        for_autoencoder = True
    )

train_data = data.ELMDataset(
        *train_data,
        config.signal_window_size,
        config.label_look_ahead,
        stack_elm_events=False,
        transform=None,
        for_autoencoder = False
    )

# Load model
# PATH = './trained_models/one_hidden_layer/simple_ae_latent_50'
PATH = 'outputs/trained_models/three_hidden/Autoencoder_400_200_400.pth'
model = torch.load(PATH, map_location=device)
model = model.to(device)
model.eval()

loss_fn = torch.nn.MSELoss()

def plot(index):
    actual_window = train_dataset[index][0].to(device)
    pred_window = train_dataset[index][1]
    model_window = model(actual_window)

    loss = loss_fn(model_window, actual_window)
    print(loss.item())
    
    number_frames = 4
    number_rows = 2
    fig, ax = plt.subplots(nrows = number_rows, ncols = number_frames)

    # Plot the actual frames 0,2,4,6
    actual = actual_window.cpu().detach().numpy()[0] # (1,8,8,8)
    actual_min = np.amin(actual)
    actual_max = np.amax(actual)
    for i in range(number_frames):
        cur_ax = ax[0][i]
        cur_ax.imshow(actual[2*i], cmap = 'hot', vmin = actual_min, vmax = actual_max)
        cur_ax.set_title(f'A {2*i}')
        cur_ax.axis('off')

    # Plot the prediction frames 0,2,4,6
    pred = model_window.cpu().detach().numpy()[0]
    for i in range(number_frames):
        cur_ax = ax[1][i]
        cur_ax.imshow(pred[2*i], cmap = 'hot', vmin = actual_min, vmax = actual_max)
        cur_ax.set_title(f'P {2*i}')
        cur_ax.axis('off')

    fig.tight_layout()
    # fig.savefig('plot.png') 
    plt.show()

if __name__ == '__main__':
    # Plot 10 model predictions

    for i in range(len(train_dataset)):
        if train_data[i][1].item() == 1:
            print(i, (train_data[i][1]).item())

    for i in range(0, 11000, 1000):
        # print(i)
        plot(i)
   
