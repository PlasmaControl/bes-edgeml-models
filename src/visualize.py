import torch
from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np
import os

from autoencoder import Autoencoder, Conv_AE, device
import data, config

# Get data and form dataset
data_ = data.Data(kfold=False, balance_classes=config.balance_classes, normalize=True)
train_data, test_data, _ = data_.get_data(shuffle_sample_indices=False)

train_dataset = data.ELMDataset(
    *train_data,
    config.signal_window_size,
    config.label_look_ahead,
    stack_elm_events=False,
    transform=None,
    for_autoencoder=True,
)

train_labeled_dataset = data.ELMDataset(
    *train_data,
    config.signal_window_size,
    config.label_look_ahead,
    stack_elm_events=False,
    transform=None,
    for_autoencoder=False,
)

test_dataset = data.ELMDataset(
    *test_data,
    config.signal_window_size,
    config.label_look_ahead,
    stack_elm_events=False,
    transform=None,
    for_autoencoder=True,
)

# Load model
PATH = './outputs/trained_models/normalized_8_frames_batch_32_100_epochs_conv/Conv_AE_latent_64_filters_20_kernel_2.pth'
# PATH = './archived_ouputs/trained_models/normalized_32_frames_three_hidden_batch_32_100_elms/Autoencoder_1000_64_1000.pth'
model = torch.load(PATH, map_location=device)
model = model.to(device)
model.eval()

print('Visualizing output of', model.name)

loss_fn = torch.nn.MSELoss()


def plot(index: int, n: int):
    actual_window = train_dataset[index][0].to(device)
    actual_window = torch.unsqueeze(actual_window, 0)

    model_window = model(actual_window)

    loss = loss_fn(model_window, actual_window)
    print(loss.item())

    number_frames = 4
    number_rows = 2
    fig, ax = plt.subplots(nrows=number_rows, ncols=number_frames)

    # Plot the actual frames
    actual = actual_window.squeeze().cpu().detach().numpy()  # (8,8,8)

    print(f'actual_min: {np.amin(actual)}')
    print(f'actual_max: {np.amax(actual)}')

    actual_min = -.5
    actual_max = .5
    for i in range(number_frames):
        cur_ax = ax[0][i]
        cur_ax.imshow(actual[n * i], cmap='RdBu', vmin=actual_min, vmax=actual_max)
        cur_ax.set_title(f'A {n * i}')
        cur_ax.axis('off')

    # Plot the prediction frames
    pred = model_window.squeeze().cpu().detach().numpy()

    print(f'pred min: {np.amin(pred)}')
    print(f'pred max: {np.amax(pred)}')
    
    for i in range(number_frames):
        cur_ax = ax[1][i]
        cur_ax.imshow(pred[n * i], cmap='RdBu', vmin=actual_min, vmax=actual_max)
        cur_ax.set_title(f'P {n * i}')
        cur_ax.axis('off')

    fig.tight_layout()
    # fig.savefig('plot.png') 
    plt.show()


def main():
    n = config.signal_window_size // 4
    for i in range(38000, 39000, 100):
        plot(i, n)


if __name__ == '__main__':
    main()
    # for i in range(len(train_dataset)):
    #     if train_labeled_dataset[i][1] == 1:
    #         print(i)
    pass
