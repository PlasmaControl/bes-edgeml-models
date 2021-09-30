"""
Created by Jeffrey Zimmerman with funding from the University of Wisconsin and DOE.
Under supervision of Dr. David Smith, U. Wisconsin for the ELM prediction and classification using ML group.
2021
"""

import argparse
import logging
import os
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition as comp
import torch
from torch import device

from options.test_arguments import TestArguments
from src.utils import get_logger, create_output_paths
from analyze import *

from visualizations.utils.utils import get_dataloader, get_model


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

        pred_dict = predict_v2(args=self.args,
                               model=self.model,
                               test_data=(signal_windows, labels),
                               hook_layer='conv')

        activations = torch.tensor(pred_dict['activations'])

        background = activations[:30]
        to_explain = activations[30:]

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


class PCA():

    def __init__(self, viz: Visualizations,
                 layer: str,
                 elm_index: int or slice = None
                 ):

        self.args = viz.args
        self.logger = viz.logger
        self.model = viz.model
        self.device = viz.device
        self.layer = layer
        self.elm_index = elm_index

        self.n_components = 5
        self.batch_num = 1

        self.elm_predictions = self.get_predictions()
        self.num_elms = len(self.elm_predictions)
        self.elm_id = list(self.elm_predictions.keys())
        self.pca_dict = self.perform_PCA()

    def get_predictions(self) -> dict:

        # load the model checkpoint and other paths
        # noinspection PyTupleAssignmentBalance
        (
            test_data_dir,
            model_ckpt_dir,
            clf_report_dir,
            plot_dir,
            roc_dir,
        ) = create_output_paths(self.args, infer_mode=True)

        model_ckpt_path = os.path.join(
            model_ckpt_dir,
            f"{args.model_name}_{args.data_mode}_lookahead_{args.label_look_ahead}{args.filename_suffix}.pth",
        )
        print(f"Using elm_model checkpoint: {model_ckpt_path}")
        self.model.load_state_dict(
            torch.load(
                model_ckpt_path,
                map_location=self.device,
            )["model"]
        )

        # get the test data and dataloader
        test_fname = os.path.join(
            test_data_dir,
            f"test_data_{self.args.data_mode}_lookahead_{self.args.label_look_ahead}{self.args.filename_suffix}.pkl",
        )

        print(f"Using test data file: {test_fname}")
        test_data, test_dataset = get_test_dataset(
            args, file_name=test_fname, logger=self.logger
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
        )
        inputs, _ = next(iter(test_loader))

        print(f"Input size: {inputs.shape}")

        if self.args.test_data_info:
            show_details(test_data)

        model_predict(self.model, self.device, test_loader)

        # get prediction dictionary containing truncated signals, labels,
        # micro-/macro-predictions and elm_time
        pred_dict = predict_v2(args=self.args,
                               test_data=test_data,
                               model=self.model,
                               device=self.device,
                               hook_layer=self.layer)

        return pred_dict

    def perform_PCA(self) -> dict:
        """
        Use scikit learn's pca analysis tools to reduce dimensionality of hidden layer
        output.

        :return: dict

        {'pca': array of transformed primary components. pca[0] is the corresponding time index.

        'components': weights of transformation vectors for each primary component.

        'evr': explained variance ratio of each primary component.}
        """

        if not np.array(self.elm_index).any():
            self.elm_index = np.arange(len(self.elm_id))

        elm_dict = {self.elm_id[i]: self.elm_predictions[self.elm_id[i]] for i in self.elm_index}
        elm_arrays = self.make_arrays(elm_dict, 'activations')
        activations = elm_arrays['activations']
        # weights = make_array(self.elm_predictions, 'weights')
        t_dim = np.arange(activations.shape[0])

        # weighted = []
        # for i, act in enumerate(activations):
        #     weighted.append(weights[i] * act)

        standard = StandardScaler().fit_transform(activations)
        # standard_t = np.append([t_dim], standard.T, axis=0).T

        pca = comp.PCA(n_components=self.n_components)
        pca.fit(standard)
        decomposed = pca.transform(standard)
        decomposed_t = np.append([t_dim], decomposed.T, axis=0).T
        print(f'Explained variance ratio of first {self.n_components} components')
        for i, x in enumerate(pca.explained_variance_ratio_[:self.n_components]):
            print(f'\tComponent {i}: {x * 100:0.3f}')
        for i, component in enumerate(pca.components_[pca.explained_variance_ratio_ > 0.01]):
            print(f'\nComponent {i} ({pca.explained_variance_ratio_[i] * 100:0.2f}% E.V.):\n{component}')

        self.pca_dict = {
            'pca': decomposed_t,
            'components': pca.components_,
            'evr': pca.explained_variance_ratio_
        }

        return self.pca_dict

    def plot_pca(self, plot_type: str) -> None:

        """
        Plot primary components of input data at specified layer.

        :param plot_type: ['3d' | 'grid']
        """

        elm_arrays = self.make_arrays(self.elm_predictions, 'activations')
        start_end = np.column_stack((elm_arrays['elm_start_idx'], elm_arrays['elm_end_idx']))[self.elm_index]
        elm_labels = elm_arrays['labels']
        decomposed_t = self.pca_dict['pca']

        if plot_type == '3d':
            fig = plt.figure()
            for start, end in start_end[:1]:
                labels = elm_labels[start:end]
                fig = px.scatter_3d(x=decomposed_t[:, 0][start:end],
                                    y=decomposed_t[:, 1][start:end],
                                    z=decomposed_t[:, 2][start:end],
                                    color=labels)

            fig.update_layout(
                title="PC_1 and PC_2 vs Time",
                scene=dict(
                    xaxis_title="Time (microseconds)",
                    yaxis_title="PC 1",
                    zaxis_title="PC 2"),
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="RebeccaPurple"
                ))
            fig.show()

        if plot_type == 'grid':
            fig, axs = plt.subplots(nrows=self.n_components, ncols=self.n_components, figsize=(16, 9))
            for ax in axs.flat:
                ax.set_axis_off()
            for pc_col in range(0, self.n_components):
                for pc_row in range(0, pc_col + 1):
                    ax = axs[pc_row][pc_col]
                    ax.set_axis_on()
                    pc_x = decomposed_t[:, pc_col - pc_row]
                    pc_y = decomposed_t[:, pc_col + 1]
                    for start, end in start_end:
                        if pc_row == pc_col:
                            x = np.arange(end - start)
                            y = pc_y[start:end]
                        else:
                            x = pc_x[start:end]
                            y = pc_y[start:end]
                        ax.scatter(x=x,
                                   y=y,
                                   edgecolor='none',
                                   c=matplotlib.cm.gist_rainbow(np.linspace(0, 1, end - start)),
                                   s=4)
                    ax.set_ylabel(f'PC_{pc_col + 1}')
                    ax.set_xlabel('Time' if pc_row == pc_col else f'PC_{pc_col - pc_row}')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.suptitle(f'PCA of Layer {self.layer} in feature model')
            plt.show()

    def correlate_pca(self, plot_type: str) -> None:
        """
        Plot correlations of input channels with PCA.

        - plot_type = 'box' for box plot of correlations along radial dimension
        - plot_type = 'hist' for histogram of correlation all 64 channels
        - plot_type = 'line' for line graph of correlation mean along radial dimension.

        :param plot_type: Type of matplotlib plot to show. ['box' | 'hist' | 'line']
        :type plot_type: str
        :return: None
        :rtype: None
        """
        ixslice = pd.IndexSlice
        pca_t = self.pca_dict['pca'].T[1:]
        elm_list = list(self.elm_predictions.keys())
        arrs = self.make_arrays(self.elm_predictions, 'signals')
        signals = arrs['signals']
        start_end = np.column_stack((arrs['elm_start_idx'], arrs['elm_end_idx']))
        reshaped = np.empty((64, signals.shape[0]))
        df_index_a = np.empty(signals.shape[0])
        df_index_b = np.empty(signals.shape[0])
        for i, (start_idx, end_idx) in enumerate(start_end):
            df_index_a[start_idx:end_idx] = elm_list[i]
            df_index_b[start_idx:end_idx] = np.arange(end_idx - start_idx)

        df_index = pd.MultiIndex.from_arrays([df_index_a, df_index_b], names=['ELM_ID', 'Time'])

        for channel_y in range(8):
            for channel_x in range(8):
                reshaped[8 * channel_x + channel_y] = signals[:, channel_y, channel_x]

        data_arr = np.append(reshaped, pca_t, axis=0).T
        data_labels = [f'Channel_{x + 1}' for x in range(64)]
        pc_labels = [f'PC_{x + 1}' for x in range(len(pca_t))]
        labels_all = np.append(data_labels, pc_labels)
        df_all = pd.DataFrame(data_arr, columns=labels_all, index=df_index)
        df_all = df_all.groupby(level='ELM_ID').corr().loc[ixslice[:, 'PC_1':'PC_5'], 'Channel_1':'Channel_64']

        if plot_type.lower() == 'hist':
            pc1_corr_df = df_all.xs('PC_1', level=1)
            # pc2_corr_df = df_all.xs('PC_2', level=1)
            for x in range(4):
                fig, axs = plt.subplots(4, 4)
                for i, ax in enumerate(axs.flat, start=1):
                    i = x * 16 + i
                    sns.histplot(data=pc1_corr_df, x=f'Channel_{i}', ax=ax, color=sns.color_palette()[1])
                    sns.kdeplot(data=pc1_corr_df, x=f'Channel_{i}', ax=ax, color=sns.color_palette()[1])
                    ax.set_title(f'Channel {i}', loc='left', y=0.8)
                fig.suptitle('Distribution of Correlations Between PC1 and Input Channels')
                plt.show()

        if plot_type.lower() == 'box':
            df_box = df_all.loc[:, 'Channel_17':'Channel_24'].reset_index(level=1)
            # must make df categorical to work on seaborn box plot.
            all_vals = []
            pc_vals = []
            channel_vals = []
            for col in df_box.loc[:, 'Channel_17':'Channel_24']:
                this_column_values = df_box[col].tolist()
                this_column_pc = df_box['level_1'].tolist()
                all_vals += this_column_values
                pc_vals += this_column_pc
                channel_vals += [col] * len(this_column_values)
            df_box = pd.DataFrame(data=np.transpose([pc_vals, channel_vals, all_vals]),
                                  columns=['Component', 'Channel', 'Correlation'])
            df_box['Correlation'] = df_box['Correlation'].astype(float)
            fig, ax = plt.subplots(1, 1)
            sns.boxplot(data=df_box, x='Channel', y='Correlation', hue='Component', ax=ax)
            fig.suptitle(f'Distribution of Correlation Along Radial Axis')
            ax.set_title(f'layer: {self.layer}')
            plt.show()

        if plot_type.lower() == 'line':
            df_line = df_all.loc[:, 'Channel_17':'Channel_24']
            fig, ax = plt.subplots(1, 1)
            df_line.groupby(level=1).mean().transpose().plot(xlabel='Channel',
                                                             ylabel='Correlation',
                                                             ax=ax)

            fig.suptitle('Mean Correlation Along Radial Axis')
            ax.set_title(f'Layer: {self.layer}')
            ax.legend()
            plt.show()

    def FFT(self, show_plot: bool = False):

        kernels = self.elm_predictions[self.elm_id[0]]['weights']
        pca_weights = self.pca_dict['components']

        K_w = np.sum(
            np.multiply(
                kernels,
                pca_weights[0].repeat(np.prod(kernels.shape[1:])).reshape(kernels.shape)
            ),
            axis=0)

        fft = np.fft.rfftn(K_w, axes=(1, 2, 0))
        if show_plot:
            fig, axs = plt.subplots(int(np.ceil(fft.shape[0] / 2)), 2)
            for ax, frame in zip(axs.flat, fft):
                ax.imshow(np.log(np.abs(np.fft.fftshift(frame)) ** 2))
            plt.show()

        return fft

    def random_sample(self, s_sample: tuple):

        (n_samples, s_sample) = s_sample
        samples = np.empty((n_samples, 5))

        for x in range(n_samples):
            sample = np.random.choice(self.num_elms, s_sample, replace=False)
            self.elm_index = sample
            self.perform_PCA()
            fft = self.FFT()
            fft1d = fft.sum(axis=(1, 2))
            samples[x] = fft1d

        fft_df = pd.DataFrame(samples, columns=[f'Freq {i}' for i in range(5)])

        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle('Distribution of Frequencies in Randomly Sampled PCA')
        ax1.set_title(f's={s_sample}, n={n_samples}')

        fft_df.boxplot(ax=ax1)
        fft_df.plot.kde(ax=ax2)

        plt.show()

    def plot_boxes(self) -> None:

        elm_ids = list(self.elm_predictions.keys())
        elm_id = elm_ids[0]
        activations = self.elm_predictions[elm_id]['activations'].squeeze().T

        ncols = 4
        nrows = ceil(activations.shape[0] / (4 * ncols))

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
        for i, ax in enumerate(axs.flat):
            vals = activations[4 * i: 4 * i + 4]
            labels = np.chararray(vals.shape, itemsize=7, unicode=True)
            for j in range(4):
                labels[j, :] = f'Node {4 * i + j}'
            activations_df = pd.DataFrame(data=list(zip(labels.flat, vals.flat)),
                                          columns=['Node', 'Activation']
                                          )
            sns.violinplot(x='Node', y='Activation', data=activations_df, ax=ax)

        fig.suptitle(f'Violin Plots of Node Activations in Layer {self.layer}')
        plt.show()

    @staticmethod
    def make_arrays(dic: dict, key: str):

        """
        Helper function to return value from predict_v2 elm_prediction dict key
        as numpy array along with start and stop indices of ELM.
        ----------------------------------------------------
        :return:   {
                        'key': np.array,        (dimensions: N-nodes x len_ELM)
                        'elm_start': np.array,  (dimensions: 1 x len_ELM)
                        'elm_end': np.array     (dimensions: 1 x len_ELM)
                        }
        :rtype: dict
        """

        ends = np.cumsum([dic[elm_idx]['elm_time'].shape[0] for elm_idx in list(dic.keys())])
        starts = np.pad(ends, (1, 0), mode='constant')[:-1]

        arr = np.empty((ends[-1], *list(dic.values())[0][key].squeeze().shape[1:]))
        label_arr = np.empty(ends[-1])

        for i_start, i_end, elm_dic in zip(starts, ends, list(dic.values())):
            arr[i_start:i_end] = elm_dic[key].squeeze()
            label_arr[i_start:i_end] = elm_dic['labels'].squeeze()

        return {key: arr, 'labels': label_arr, 'elm_start_idx': starts, 'elm_end_idx': ends}


def make_animation(viz: Visualizations, elm_id: int = 0) -> None:
    layers = list(viz.model.layers.keys())[:-1]
    d_lst = []
    l_lst = []
    for layer in layers:
        pca = PCA(viz, layer=layer)
        labels = pca.elm_predictions[elm_id]['labels']
        time_dict = PCA.make_arrays({elm_id: pca.elm_predictions[elm_id]}, 'activations')
        start = time_dict['elm_start_idx'].item()
        end = time_dict['elm_end_idx'].item()
        elm = pca.perform_PCA()['pca'][start:end, 1:3]

        d_lst.append(elm)
        l_lst += [layer] * len(labels)

    df_arr = np.array(d_lst).reshape((-1, 2))
    df = pd.DataFrame({'Layer': l_lst,
                       'Label': np.tile(labels, len(layers)),
                       'PC_1': df_arr[:, 0],
                       'PC_2': df_arr[:, 1]})
    fig = px.scatter(df, x='PC_1', y='PC_2', animation_frame='Layer', color='Label')
    fig.show()


if __name__ == "__main__":
    args, parser = TestArguments().parse(verbose=True)
    LOGGER = get_logger(
        script_name=__name__,
        log_file=os.path.join(
            args.log_dir,
            f"output_logs_{args.model_name}_{args.data_mode}{args.filename_suffix}.log",
        ),
    )

    viz = Visualizations(args=args, logger=LOGGER)
    make_animation(viz)
    # pca = PCA(viz, layer='conv')
    # pca.plot_pca('grid')
