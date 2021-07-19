from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sb
import numpy as np
import data, config

data_ = data.Data(kfold=False, balance_classes=config.balance_classes, normalize = True)
train_data, test_data, _ = data_.get_data(shuffle_sample_indices=False)

train_dataset = data.ELMDataset(
        *train_data,
        config.signal_window_size,
        config.label_look_ahead,
        stack_elm_events=False,
        transform=None,
        for_autoencoder = True
    )

# train_data[0] is a numpy array of many 8x8 signal frames
signal_frames = train_data[0]
signal_labels = train_data[1]

start_idx = 1950
end_idx = 2400

s = signal_frames[start_idx:end_idx,...] #s is the frames displayed
l = signal_labels[start_idx:end_idx,...] #l is the corresponding labels

def init():
    sb.heatmap(np.zeros((8, 8)), square=True, cbar=True, xticklabels=False, yticklabels=False, cmap = 'hot')

def animate(i):
    plt.clf()
    # print(f'Label at frame {i}: {l[i]}')
    data = s[i]
    label = l[i]
    print(i)
    plt.title(f'Label: {label}')
    sb.heatmap(data, vmin = -1, vmax = 1, square=True, cbar=True, xticklabels=False, yticklabels=False, cmap = 'hot')


def show_animation():
    fig = plt.figure()
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(s), interval = 1, repeat = False)
    plt.show()

if __name__ == '__main__':
    show_animation()
    # for i, label in enumerate(signal_labels):
    #     if label == 1.0:
    #         print(i, label)