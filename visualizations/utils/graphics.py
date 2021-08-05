import torch
import numpy as np
import matplotlib.pyplot as plt


def plot(index: int, n: int):
    actual_window = train_dataset[index][0].to(device)
    actual_window = torch.unsqueeze(actual_window, 0)

    model_window = self.model(actual_window)

    loss_fn = torch.nn.MSELoss()
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
