import logging
import numpy as np
from src.utils import make_logger
from options.test_arguments import TestArguments
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex



class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(3, 6, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2)

        self.conv_trans1 = nn.ConvTranspose2d(6, 3, 4, 2, 1)
        self.conv_trans2 = nn.ConvTranspose2d(3, 1, 4, 2, 1)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.conv_trans1(x))
        x = self.conv_trans2(x)
        return x

    dataset = datasets.MNIST(
        root='PATH',
        transform=transforms.ToTensor(),
        download=True
    )
    loader = DataLoader(
        dataset,
        num_workers=2,
        batch_size=8,
        shuffle=True
    )


def train():
    model = MyModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 1
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()

            print('Epoch {}, Batch idx {}, loss {}'.format(
                epoch, batch_idx, loss.item()))

    def normalize_output(img):
        img = img - img.min()
        img = img / img.max()
        return img

    # Plot some images
    idx = torch.randint(0, output.size(0), ())
    pred = normalize_output(output[idx, 0])
    img = data[idx, 0]

    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img.detach().numpy())
    axarr[1].imshow(pred.detach().numpy())

    # Visualize feature maps
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    model.conv1.register_forward_hook(get_activation('conv1'))
    data, _ = dataset[0]
    data.unsqueeze_(0)
    output = model(data)

    act = activation['conv1'].squeeze()
    fig, axarr = plt.subplots(act.size(0))
    for idx in range(act.size(0)):
        axarr[idx].imshow(act[idx])

    plt.show()


def plot_voxels(n_voxels):
    import matplotlib.pyplot as plt
    import numpy as np
    from visualization import PCA, Visualizations


    def init_data(init_height):
        def explode(data):
            size = np.array(data.shape) * 2
            data_e = np.zeros(size - 1, dtype=data.dtype)
            data_e[::2, ::2, ::2] = data
            return data_e

        # build up the numpy logo
        # n_voxels[:, :, -2:] = n_voxels[:, :, -2:] * 5

        cmap = get_cmap('jet')
        rgb_arr = cmap(n_voxels).reshape(-1, 4)
        rgb_arr[:, -1] = 0.5 #FOR TRANSPARENT ONLY
        facecolors = np.array([to_hex(rgb, keep_alpha=True) for rgb in rgb_arr]).reshape(n_voxels.shape)
        edgecolors = np.where(n_voxels, '#BFAB6E', '#BFAB6E')
        filled = np.ones(n_voxels.shape)

        # upscale the above voxel image, leaving gaps
        filled_2 = explode(filled)
        fcolors_2 = explode(facecolors)
        ecolors_2 = explode(edgecolors)


        # Shrink the gaps
        x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
        x[0::2, :, :] += 0.05
        y[:, 0::2, :] += 0.05
        z[:, :, 0::2] += 0.05
        x[1::2, :, :] += 0.95
        y[:, 1::2, :] += 0.95
        z[:, :, 1::2] += 0.95

        # z[:, 0:16, :] += init_height

        return (x, y, z, filled_2, fcolors_2, ecolors_2)


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d',
                         xlim=n_voxels.shape[0],
                         ylim=n_voxels.shape[1],
                         zlim=n_voxels.shape[2])

    ax.invert_zaxis()
    #Even aspect ratio
    ax.set_box_aspect((n_voxels.shape[0], n_voxels.shape[1], n_voxels.shape[2]))
    # Hide grid lines
    ax.grid(False)
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    #Remove background plane
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # Set viewing angle
    ax.view_init(elev=0, azim=0)


    init_height = 8
    dat = init_data(init_height)
    # data = np.array(list(gen_data(dat, init_height, N)))
    vox = ax.voxels(dat[0], dat[1], dat[2], dat[3], facecolors=dat[4], edgecolors=dat[5])


    plt.show()


if __name__ == "__main__":
    args, parser = TestArguments().parse(verbose=True)
    LOGGER = make_logger(script_name=__name__, log_file=None)

    plot_voxels(args, LOGGER)