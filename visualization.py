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
    viz.feature_map('conv3')

    # TODO: add labels (target data 0 or 1) for plots
