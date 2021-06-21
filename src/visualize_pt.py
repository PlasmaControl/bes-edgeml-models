import torch
# from autoencoder_pt import Autoencoder_PT
from ae_easy import Autoencoder_easy
# from ae_one_piece import Autoencoder_OP
from ae_easy import device
import data, config
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import seaborn as sb
import numpy as np
from scipy.ndimage.filters import gaussian_filter

fold = 1
data_ = data.Data(kfold=True, balance_classes=config.balance_classes)
train_data, test_data, _ = data_.get_data(shuffle_sample_indices=False, fold=fold)

train_dataset = data.ELMDataset(
        *train_data,
        config.signal_window_size,
        config.label_look_ahead,
        stack_elm_events=False,
        transform=None,
        for_autoencoder = True
    )

# Load model
PATH = './easy_model.pth'
model = torch.load(PATH)
model = model.to(device)
model.eval()



def plot(index):
    actual_window = train_dataset[index][0].to(device)
    pred_window = train_dataset[index][1]
    model_window = model(actual_window)
    
    number_frames = 4
    number_rows = 2
    fig, ax = plt.subplots(nrows = number_rows, ncols = number_frames)

    # Plot the actual frames 0,2,4,6
    actual = actual_window.cpu().detach().numpy()[0]
    for i in range(number_frames):
        cur_ax = ax[0][i]
        cur_ax.imshow(actual[2*i], cmap = 'hot')
        cur_ax.set_title(f'A {2*i}')
        cur_ax.axis('off')

    # Plot the prediction frames 0,2,4,6
    pred = model_window.cpu().detach().numpy()[0]
    for i in range(number_frames):
        cur_ax = ax[1][i]
        cur_ax.imshow(pred[2*i], cmap = 'hot')
        cur_ax.set_title(f'P {2*i}')
        cur_ax.axis('off')

    fig.tight_layout()
    # fig.savefig('plot.png') 
    plt.show()

# train_data[0] is a numpy array of many 8x8 signal frames
signal_frames = train_data[0]
signal_labels = train_data[1]


s = signal_frames[15000:15300,...]
l = signal_labels[15000:15300,...]

# fig = plt.figure()

def init():
    sb.heatmap(np.zeros((8, 8)), square=True, cbar=True, xticklabels=False, yticklabels=False, cmap = 'hot')

def animate(i):
    plt.clf()
    # print(f'Label at frame {i}: {l[i]}')
    data = s[i]
    label = l[i]
    # print(i)
    plt.title(f'Label: {label}')
    sb.heatmap(data, vmin = -1, vmax = 1, square=True, cbar=True, xticklabels=False, yticklabels=False, cmap = 'hot')


def show_animation():
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(s), interval = 1, repeat = False)
    plt.show()


# def save_animation(save_path):
#     anim = FuncAnimation(fig, animate, init_func=init, frames=len(s), interval = 1, repeat = False)
#     writer = FFMpegWriter(fps=60, metadata=dict(artist='PA'), bitrate=1800)
#     anim.save(save_path)
#     plt.close()


if __name__ == '__main__':
    for i in range(0, 11000, 1000):
        # print(i)
        plot(i)
    
    # show_animation()
    # save_animation('elm_event_animation')