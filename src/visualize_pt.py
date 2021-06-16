import data, config
from matplotlib import pyplot as plt
from matplotlib import animation
import seaborn as sb
import numpy as np

fold = 1

data_ = data.Data(kfold=True, balance_classes=config.balance_classes)
train_data, _, _ = data_.get_data(shuffle_sample_indices=False, fold=fold)


# train_data[0] is a numpy array of many 8x8 signal frames
signal_frames = train_data[0]
signal_labels = train_data[1]


s = signal_frames[15000:15300,...]
l = signal_labels[15000:15300,...]

s = signal_frames[20610:21000,...]
l = signal_labels[20610:21000,...]

fig = plt.figure()
data = np.random.rand(8, 8)
sb.heatmap(data, square=True)
# print(signal_frames[0])

def init():
    sb.heatmap(np.zeros((8, 8)), square=True, cbar=True, xticklabels=False, yticklabels=False, cmap = 'hot')

def animate(i):
    plt.clf()
    # print(f'Label at frame {i}: {l[i]}')
    data = s[i]
    label = l[i]
    plt.title(f'Label: {label}')
    sb.heatmap(data, vmin = -1, vmax = 1, square=True, cbar=True, xticklabels=False, yticklabels=False, cmap = 'hot')


def show_animation():
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(s), interval = 10, repeat = False)
    plt.show()


if __name__ == '__main__':
    show_animation()

    for i, label in enumerate(signal_labels):
        if label == 1.0:
            print(i, label)

    max_val = np.amax(signal_frames)
    min_val = np.amin(signal_frames)

    # print(f'Max: {max_val}, Min: {min_val}')