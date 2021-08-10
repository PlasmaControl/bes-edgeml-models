import argparse
import logging
import os
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from options.base_arguments import BaseArguments
from src.utils import get_logger

from visualizations.utils.utils import get_dataloader, get_model

'''
Visualize the activation of classification models via the SHAP library.
'''


class Visualizations:

    def __init__(self, args: argparse.Namespace, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.train_set = get_dataloader(self.args, self.logger, use_saved=True)
        self.model = get_model(self.args, self.logger)
        self.device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.fm_signal_window, labels = next(iter(self.train_set))

    def feature_map(self, layer: str):

        # Visualize feature maps
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        self.model.conv1.register_forward_hook(get_activation(layer))
        data = self.fm_signal_window[0]
        data.unsqueeze_(0)
        output = self.model(data)

        act = activation[layer].squeeze()

        n = self.args.signal_window_size // 4
        num_filters = act.size(0)

        self.plot_feature_map(layer, n, num_filters, act)

        # for idx in range(act.size(0)):
        #     axarr[idx].imshow(act[idx])
        #
        # plt.show()

        # for i in range (38000, 39000, 100):
        #     self.plot_(i, n)

    def shap(self):
        batch = next(iter(self.train_set))
        signal_windows, labels = batch

        background = signal_windows[:30]
        to_explain = signal_windows[30:]

        e = shap.DeepExplainer(self.model, background)
        shap_values = e.shap_values(to_explain)

        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(to_explain.numpy(), 1, -1), 1, 2)

        shap.image_plot(shap_numpy, -test_numpy)

    def plot_feature_map(self, layer: str, n: int, num_filters: int, activation):

        actual_window = self.fm_signal_window
        actual_window = torch.unsqueeze(actual_window[0], 0)

        number_cols = 4
        number_rows = num_filters + 1

        # set seaborn plotting mode
        sns.set()
        fig, ax = plt.subplots(nrows=number_rows, ncols=number_cols, figsize=(15, 15))
        fig.suptitle(f'{self.args.model_name} activations for layer {layer}')

        # Plot the actual frames
        actual = actual_window.squeeze().cpu().detach().numpy()  # (16,8,8)

        print(f'actual_min: {np.amin(actual)}')
        print(f'actual_max: {np.amax(actual)}')

        actual_min = np.amin(actual)
        actual_max = np.amax(actual)

        for i in range(number_cols):
            cur_ax = ax[0][i]
            cur_ax.imshow(actual[n * i], cmap='RdBu', vmin=actual_min, vmax=actual_max)
            cur_ax.set_title(f'A {n * i}')
            cur_ax.axis('off')

        for i in range(1, number_rows):
            for j in range(number_cols):
                cur_ax = ax[i][j]
                act = activation[i - 1][n * j]
                cur_ax.imshow(act, cmap='RdBu', vmin=actual_min, vmax=actual_max)
                cur_ax.set_title(f'F{i} {n * j}')
                cur_ax.axis('off')

        fig.tight_layout()
        # fig.savefig('plot.png')
        plt.show()

    def plot_weights(self, layer_name, single_channel=True, collated=False):

        model = self.model
        # extracting the model features at the particular layer number
        layer_dict = {'conv1': model.conv1, 'conv2': model.conv2, 'conv3': model.conv3}
        layer = layer_dict[layer_name]

        # checking whether the layer is convolution layer or not
        if isinstance(layer, torch.nn.Conv3d):
            # getting the weight tensor data
            weight_tensor = layer.weight.data

            if single_channel:
                if collated:
                    self.plot_filters_single_channel_big(weight_tensor)
                else:
                    self.plot_filters_single_channel(weight_tensor)

            else:
                if weight_tensor.shape[1] == 3:
                    self.plot_filters_multi_channel(weight_tensor)
                else:
                    print("Can only plot weights with three channels with single channel = False")

        else:
            print("Can only visualize layers which are convolutional")

    @staticmethod
    def plot_filters_single_channel_big(t):

        # setting the rows and columns
        nrows = t.shape[0] * t.shape[2]
        ncols = t.shape[1] * t.shape[3]

        npimg = np.array(t.numpy(), np.float32)
        npimg = npimg.transpose((0, 2, 1, 3))
        npimg = npimg.ravel().reshape(nrows, ncols)

        npimg = npimg.T

        fig, ax = plt.subplots(figsize=(ncols / 10, nrows / 200))
        imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='gray', ax=ax, cbar=False)

    @staticmethod
    def plot_filters_single_channel(t):

        # kernels depth * number of kernels
        nplots = t.shape[0] * t.shape[1]
        ncols = 12

        nrows = 1 + nplots // ncols
        # convert tensor to numpy image
        npimg = np.array(t.numpy(), np.float32)

        count = 0
        fig = plt.figure(figsize=(ncols, nrows))

        # looping through all the kernels in each channel
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                count += 1
                ax1 = fig.add_subplot(nrows, ncols, count)
                npimg = np.array(t[i, j].numpy(), np.float32)
                npimg = (npimg - np.mean(npimg)) / np.std(npimg)
                npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
                ax1.imshow(npimg)
                ax1.set_title(str(i) + ',' + str(j))
                ax1.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_filters_multi_channel(t):

        # get the number of kernals
        num_kernels = t.shape[0]

        # define number of columns for subplots
        num_cols = 12
        # rows = num of kernels
        num_rows = num_kernels

        # set the figure size
        fig = plt.figure(figsize=(num_cols, num_rows))

        # looping through all the kernels
        for i in range(t.shape[0]):
            ax1 = fig.add_subplot(num_rows, num_cols, i + 1)

            # for each kernel, we convert the tensor to numpy
            npimg = np.array(t[i].numpy(), np.float32)
            # standardize the numpy image
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            npimg = npimg.transpose((1, 2, 0))
            ax1.imshow(npimg)
            ax1.axis('off')
            ax1.set_title(str(i))
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

        plt.savefig('myimage.png', dpi=100)
        plt.tight_layout()
        plt.show()

    def imshow(self, img, title):

        """Custom function to display the image using matplotlib"""

        # define std correction to be made
        std_correction = np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        # define mean correction to be made
        mean_correction = np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1)

        # convert the tensor img to numpy img and de normalize
        npimg = np.multiply(img.numpy(), std_correction) + mean_correction

        # plot the numpy image
        plt.figure(figsize=(self.args.batch_size * 4, 4))
        plt.axis("off")
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    args, parser = BaseArguments().parse(verbose=True)
    LOGGER = get_logger(
        script_name=__name__,
        log_file=os.path.join(
            args.log_dir,
            f"output_logs_{args.model_name}_{args.data_mode}{args.filename_suffix}.log",
        ),
    )

    viz = Visualizations(args=args, logger=LOGGER)
    viz.feature_map('conv1')
    # viz.plot_weights('conv3')
