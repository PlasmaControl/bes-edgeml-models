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

fig = plt.figure()
# data = np.random.rand(8, 8)
# sb.heatmap(data, square=True)
# print(signal_frames[0])

def init():
    sb.heatmap(np.zeros((8, 8)), vmin = -1, vmax = 1, square=True, cbar=False, xticklabels=False, yticklabels=False, cmap = 'hot')

def animate(i):
    plt.clf()
    data = signal_frames[i]
    sb.heatmap(data, square=True, vmin = -1, vmax = 1, cbar=True, xticklabels=False, yticklabels=False, cmap = 'hot')


# def init():
#     plt.imshow(np.zeros((8, 8)), cmap = 'hot')

# def animate(i):
#     plt.clf()
#     print(f'Label at frame {i}: {signal_labels[i]}')
#     data = signal_frames[i]
#     plt.imshow(data, cmap = 'hot')


# anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(signal_frames), interval = 1, repeat = False)
# plt.show()


# for i, label in enumerate(signal_labels):
#     if label == 1.0:
#         print(i, label)