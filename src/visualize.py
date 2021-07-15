import torch
from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np
import os

from autoencoder import Autoencoder, Conv_AE, device
import data, config

# Get data and form dataset
data_ = data.Data(kfold=False, balance_classes=config.balance_classes, normalize = True)
train_data, test_data, _ = data_.get_data(shuffle_sample_indices=False)

train_dataset = data.ELMDataset(
        *train_data,
        config.signal_window_size,
        config.label_look_ahead,
        stack_elm_events=False,
        transform=None,
        for_autoencoder = True,
    )

test_dataset = data.ELMDataset(
        *test_data,
        config.signal_window_size,
        config.label_look_ahead,
        stack_elm_events=False,
        transform=None,
        for_autoencoder = True,
    )

# train_dataset.__getitem__(15500)[0]

# Load model
# PATH = './untrained_autoencoder.pth'
PATH = './outputs/trained_models/normalized_three_hidden_batch_32_100_elms/Autoencoder_600_32_600.pth'
model = torch.load(PATH, map_location=device)
model = model.to(device)
model.eval()

print()
print(model)

loss_fn = torch.nn.MSELoss()

def plot(index: int, n: int):
    actual_window = train_dataset[index][0].to(device)
    model_window = model(actual_window)

    loss = loss_fn(model_window, actual_window)
    print(loss.item())
    
    number_frames = 4
    number_rows = 2
    fig, ax = plt.subplots(nrows = number_rows, ncols = number_frames)

    # Plot the actual frames 0,2,4,6
    actual = actual_window.cpu().detach().numpy()[0] # (1,8,8,8)
    # actual_min = np.amin(actual)
    # actual_max = np.amax(actual)
    actual_min = -1
    actual_max = 1
    for i in range(number_frames):
        cur_ax = ax[0][i]
        cur_ax.imshow(actual[n*i], cmap = 'RdBu', vmin = actual_min, vmax = actual_max)
        cur_ax.set_title(f'A {n*i}')
        cur_ax.axis('off')

    # Plot the prediction frames 0,2,4,6
    pred = model_window.cpu().detach().numpy()[0]
    for i in range(number_frames):
        cur_ax = ax[1][i]
        cur_ax.imshow(pred[n*i], cmap = 'RdBu', vmin = actual_min, vmax = actual_max)
        cur_ax.set_title(f'P {n*i}')
        cur_ax.axis('off')

    fig.tight_layout()
    # fig.savefig('plot.png') 
    plt.show()


def main():
    # Plot 10 model predictions
    # for i in range(len(train_dataset)):
    #     if train_data[i][1].item() == 1:
    #         print(i, (train_data[i][1]).item())

    for i in range(20000, 31000, 1000):
        # print(i)
        plot(i, n = 4)

if __name__ == '__main__':
    main()
    pass
   
