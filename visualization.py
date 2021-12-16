"""
Created by Jeffrey Zimmerman with funding from the University of Wisconsin and DOE.
Under supervision of Dr. David Smith, U. Wisconsin for the ELM prediction and classification using ML group.
2021
"""

import argparse
import logging
import os
import re

from matplotlib.cm import Set1
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import decomposition as comp
import torch
from torch import device, nn
from torch.optim import SGD
from flashtorch.activmax import GradientAscent

from src.utils import get_logger, create_output_paths
from options.test_arguments import TestArguments
from analyze import *
from visualizations.utils.utils import get_dataloader, get_model


class GradientAscent3D(GradientAscent):

    def __init__(self, model, cl_args):
        super().__init__(model, img_size=8)
        self.args = cl_args

    def optimize3D(self, layer, filter_idx, input_=None, num_iter=30):

        # if type(layer) != nn.modules.conv.Conv3d:
        #     raise TypeError('The layer must be nn.modules.conv.Conv3d.')
        #
        # num_total_filters = layer.out_channels
        # self._validate_filter_idx(num_total_filters, filter_idx)

        if input_ is None:
            input_ = np.random.uniform(0, 10, size=(self.args.signal_window_size, 8, 8))
            input_[:, 2:, :] /= 2
            input_ = torch.tensor(input_, requires_grad=True, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # remove previous hooks
        while len(self.handlers) > 0:
            self.handlers.pop().remove()

        # register new hooks for activations and gradients
        self.handlers.append(self._register_forward_hooks(layer, filter_idx))
        self.handlers.append(self._register_backward_hooks())

        self.gradients = torch.zeros(input_.shape)

        return np.array([ascent.squeeze().detach().numpy() for ascent in self._ascent(input_, num_iter)])

    def _register_backward_hooks(self):
        def _record_gradients(module, grad_in, grad_out):
            if self.gradients.shape == grad_in[0].shape:
                self.gradients = grad_in[0]

        for _, module in self.model.named_modules():
            if isinstance(module, nn.modules.conv.Conv3d):
                return module.register_backward_hook(_record_gradients)


class Visualizations:

    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> object:
        self.args = args
        self.logger = logger
        self.gen_suffix_type = '_' + re.split('[_.]', args.input_file)[-2] if 'generated' in args.input_file else ''
        self.filename_suffix = args.filename_suffix + self.gen_suffix_type
        self.test_data, self.test_set = self._get_test_dataset()
        self.model = get_model(self.args, self.logger)
        self.device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.fm_signal_window, labels = next(iter(self.test_set))

    def _get_test_dataset(self):

        (test_data_dir, model_ckpt_dir, clf_report_dir, plot_dir, roc_dir,) = create_output_paths(self.args,
                                                                                                  infer_mode=True)

        # get the test data and dataloader
        accepted_preproc = ['wavelet']
        test_fname = os.path.join(test_data_dir,
                                  f'test_data_lookahead_{self.args.label_look_ahead}{self.filename_suffix}'
                                  f'{"_" + self.args.data_preproc if self.args.data_preproc in accepted_preproc else ""}.pkl', )

        print(f"Using test data file: {test_fname}")
        test_data, test_dataset = get_test_dataset(self.args, file_name=test_fname, logger=self.logger)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False,
                                                  drop_last=True)

        return test_data, test_loader

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

        # for idx in range(act.size(0)):  #     axarr[idx].imshow(act[idx])  #  # plt.show()

        # for i in range (38000, 39000, 100):  #     self.plot_(i, n)

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

    def max_activation(self, layer, filter_idx, num_iter=30):

        layer = self.model.layers[layer]
        g_ascent = GradientAscent3D(self.model, self.args)
        output = g_ascent.optimize3D(layer, filter_idx=filter_idx, num_iter=num_iter)
        return output

    def generate_csi(self, target_class, iterations=150, initial_lr=6, random_state=None, blur=None,
                     l2_weight: float = 0):
        """

        @param initial_lr: Learning rate of optimizer
        @param random_state: State for seed input
        @param blur: Apply Gaussian Blur after each iteration with sigma specified by blur
        @param target_class: Options: ['ELM' | 'pre-ELM']
        @param iterations: number of times to update input.
        @return: np.array of every 10th iteration of optimizer
        """
        # set up keys from arguments
        target_class = str(target_class)
        try:
            label = {'ELM': torch.tensor([1], device=self.device, dtype=torch.float32),
                     'PRE-ELM': torch.tensor([0], device=self.device, dtype=torch.float32),
                     '1': torch.tensor([1], device=self.device, dtype=torch.float32),
                     '0': torch.tensor([0], device=self.device, dtype=torch.float32)}[target_class.upper()]
            print(f'Target class: {target_class}; Label: {label.item()}')
        except KeyError:
            raise 'Target classes are "ELM" or "pre-ELM"'

        self.model.eval()

        # generate initial random input
        np.random.seed(random_state)
        # generated = np.random.uniform(0, 10, size=(self.args.signal_window_size, 8, 8))
        # # Top two rows of BES are capped at 10, bottom 6 are capped at 5
        # generated[:, 4:, :] /= 2
        # generated = torch.tensor(generated, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)\
        #     .requires_grad_()
        # generated_i = generated

        # get test batch with elm or pre elm

        for batch in iter(self.test_set):
            if label in batch[1]:

                offset = np.random.randint(0, len(batch[1]))
                g_idx = np.argmax(batch[1] == label)
                if batch[1][g_idx - offset] != label:
                    continue

                generated = batch[0][g_idx - offset].unsqueeze(0)
                generated_i = generated

                # check if model prediction is true positive/negative
                model_output = self.model(generated)
                if model_output >= self.args.threshold and label == 1:
                    break
                elif model_output < self.args.threshold and label == 1:
                    continue
                elif model_output < self.args.threshold and label == 0:
                    break
                elif model_output >= self.args.threshold and label == 0:
                    continue

        generated_inputs = []
        model_outs = []
        for i in range(1, iterations):
            # Define optimizer for the image
            optimizer = SGD([generated], lr=initial_lr, weight_decay=l2_weight)
            # Forward pass
            output = self.model(generated)

            # Target specific class
            loss_fn = nn.BCEWithLogitsLoss()
            class_loss = loss_fn(output.view(-1), label)

            if i % 10 == 0 or i == iterations - 1:
                print(f'Iteration: {i}, Loss {class_loss.item():.2f}')
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()

            if i % 10 == 0 or i == (iterations - 1):
                # append image to list
                generated_inputs.append(generated)
                model_outs.append(output)

            if blur:
                generated = torch.tensor(gaussian_filter(generated.squeeze().detach().numpy(), blur)).unsqueeze(
                    0).unsqueeze(0).requires_grad_()

        return tuple((im.detach().numpy().squeeze(), torch.sigmoid(model_out_).item(),
                      generated_i.detach().numpy().squeeze()) for im, model_out_ in zip(generated_inputs, model_outs))


class PCA():

    def __init__(self, viz: Visualizations,
                 layer: str,
                 elm_index: int or np.ndarray = None
                 ):

        self.args = viz.args
        self.logger = viz.logger
        self.model = viz.model
        self.device = viz.device
        self.test_data = viz.test_data
        self.test_set = viz.test_set
        self.gen_suffix_type = viz.gen_suffix_type
        self.layer = layer
        self.usr_elm_index = np.array([elm_index]).reshape(-1, ) if elm_index is not None else elm_index
        self.filename_suffix = self.args.filename_suffix + self.gen_suffix_type

        self.n_components = 5
        self.batch_num = 1

        self.elm_predictions = self.get_predictions()
        self.num_elms = len(self.elm_predictions)
        self.elm_id = list(self.elm_predictions.keys())
        self.pca_dict = None

    def get_predictions(self) -> dict:

        inputs, _ = next(iter(self.test_set))

        print(f"Input size: {inputs.shape}")

        if self.args.test_data_info:
            show_details(self.test_data)

        model_predict(self.model, self.device, self.test_set)

        # get prediction dictionary containing truncated signals, labels,
        # micro-/macro-predictions and elm_time
        pred_dict = predict(args=self.args, test_data=self.test_data, model=self.model, device=self.device,
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

        if self.usr_elm_index is None:
            self.usr_elm_index = np.arange(len(self.elm_id))

        elm_dict = {self.elm_id[i]: self.elm_predictions[self.elm_id[i]] for i in self.usr_elm_index}
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
        start_end = np.column_stack((elm_arrays['elm_start_idx'], elm_arrays['elm_end_idx']))[self.usr_elm_index]
        elm_labels = elm_arrays['labels']
        decomposed_t = self.pca_dict['pca']

        labels_all = np.array([(1 - int(i)) * 'pre-' + 'ELM' for i in elm_labels])

        if plot_type == '3d':
            for start, end in start_end[:1]:
                labels = labels_all[start:end]
                fig = px.scatter_3d(x=decomposed_t[:, 0][start:end],
                                    y=decomposed_t[:, 1][start:end],
                                    z=decomposed_t[:, 2][start:end],
                                    color=labels)

            fig.update_traces(marker=dict(size=2,
                                          line=dict(width=1)),
                              selector=dict(mode='markers'))

            fig.update_layout(
                title="PC_1 and PC_2 vs Time",
                scene=dict(
                    xaxis_title="Time (microseconds)",
                    yaxis_title="PC 1",
                    zaxis_title="PC 2"),
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="RebeccaPurple"))
            fig.show()

        if plot_type == 'grid':

            from matplotlib.lines import Line2D

            fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(16, 9))
            for ax in axs.flat:
                ax.set_axis_off()
            for pc_col in range(0, 3):
                for pc_row in range(0, pc_col + 1):
                    ax = axs[pc_row][pc_col]
                    ax.set_axis_on()
                    pc_x = decomposed_t[:, pc_col - pc_row]
                    pc_y = decomposed_t[:, pc_col + 1]
                    for start, end in start_end:
                        colors = Set1((1 - elm_labels[start:end]) / 9)
                        if pc_row == pc_col:
                            x = np.arange(end - start)
                            y = pc_y[start:end]
                        else:
                            x = pc_x[start:end]
                            y = pc_y[start:end]
                        ax.scatter(x=x,
                                   y=y,
                                   edgecolor='none',
                                   c=colors,
                                   s=4)
                    ax.set_ylabel(f'PC_{pc_col + 1}', fontsize='large')
                    ax.set_xlabel('Time' if pc_row == pc_col else f'PC_{pc_col - pc_row}', fontsize='large')
            legend_elements = [Line2D([0], [0], marker='o', color=colors[0], label='pre-ELM'),
                               Line2D([0], [0], marker='o', color=colors[-1], label='ELM')]

            fig.legend(handles=legend_elements)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.suptitle(f'PCA of Layer {self.layer} in {self.args.model_name} model', fontsize='x-large')
            plt.show()

        if plot_type == 'compare':

            elm_predictions = self.make_arrays(self.elm_predictions, 'micro_predictions')['micro_predictions']
            ch_22 = self.make_arrays(self.elm_predictions, 'signals')['signals'][:, 2, 5]

            # make lists for dataframe
            ch_22_lst = []
            l_lst = []
            p_lst = []
            for start, end in start_end:
                labels = elm_labels[start:end]
                pred = elm_predictions[start:end]
                sig = ch_22[start:end]

                l_lst.append(labels.tolist())
                p_lst.extend(pred.tolist())
                ch_22_lst.extend(sig.tolist())

            # convert everything to appropriately sized arrays for pandas
            pc_arr = decomposed_t[:, :3]
            l_arr = np.array([(1 - int(j)) * 'pre-' + 'ELM' for i in l_lst for j in i])
            id_arr = np.array([self.usr_elm_index[i] for i, k in enumerate(l_lst) for _ in k])
            p_arr = np.array(p_lst)
            ch_22_arr = np.array(ch_22_lst)

            # normalize arrays to 1
            # pc_arr /= pc_arr.max()
            # p_arr /= p_arr.max()
            ch_22_arr /= ch_22_arr.max()

            df = pd.DataFrame(
                    {'ELM_ID': id_arr, 'Time': pc_arr[:, 0], 'Label': l_arr.astype(str), 'BES_CH_22': ch_22_arr,
                     'Predictions': p_arr, 'PC_1': pc_arr[:, 1], 'PC_2': pc_arr[:, 2]})

            df_melt = pd.melt(df, id_vars=['Time', 'Label'], value_vars=['PC_1', 'PC_2'],
                              var_name='PC', value_name='Activation')

            df_melt['Activation'] = MinMaxScaler().fit_transform(df_melt[['Activation']])

            fig = px.scatter(df_melt, x='Time', y='Activation',
                             color='PC',
                             title=f'Primary Components at Layer {self.layer} and Model Output')

            fig.add_trace(go.Scatter(x=df['Time'], y=df['Predictions'],
                                     mode='lines',
                                     name='Model Prediction'))

            fig.add_trace(go.Scatter(x=df['Time'], y=df['BES_CH_22'],
                                     mode='lines',
                                     name='BES Channel 22'))

            last = df[df['Label'] == 'ELM'].groupby(by='ELM_ID').last()['Time'].tolist()
            first = df[df['Label'] == 'ELM'].groupby(by='ELM_ID').first()['Time'].tolist()
            for x0, x1 in zip(first, last):
                fig.add_vrect(x0=x0, x1=x1,
                              annotation_text='Active ELM Region', annotation_position='top left',
                              fillcolor='red',
                              opacity=0.2,
                              line_width=0)

            fig.update_layout(hovermode='x unified',
                              font=dict(size=18)
                              )

            fig.show()

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
            df_new = df_box.loc[df_box['Component'] != 'PC_4']
            df_new = df_new.loc[df_new['Component'] != 'PC_5']
            df_new = df_new.loc[df_new['Component'] != 'PC_3']
            fig, ax = plt.subplots(1, 1)
            sns.boxplot(data=df_new, x='Channel', y='Correlation', hue='Component', ax=ax)
            fig.suptitle(f'Distribution of Correlation Along Radial Axis')
            ax.set_title(f'layer: {self.layer}')
            plt.show()

        if plot_type.lower() == 'line':
            df_line = df_all.loc[:, 'Channel_17':'Channel_24']
            # fig, ax = plt.subplots(1, 1)
            # df_line.groupby(level=1).mean().transpose().plot(xlabel='Channel',
            #                                                  ylabel='Correlation',
            #                                                  ax=ax)

            # fig.suptitle('Mean Correlation Along Radial Axis')
            # ax.set_title(f'Layer: {self.layer}')
            # ax.legend()
            # plt.show()

            return df_line.groupby(level=1).mean().transpose()

    def decompose_kernel(self):
        kernels = self.elm_predictions[self.elm_id[0]]['weights']
        pca_weights = self.pca_dict['components']

        K_w = np.multiply(kernels, pca_weights[0].repeat(np.prod(kernels.shape[1:])).reshape(kernels.shape))

        K_ws = np.sum(K_w, axis=0)

        return K_w

    @staticmethod
    def FFT(kernel, show_plot: bool = False):

        fft = np.fft.rfftn(kernel, axes=(1, 2, 0))
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
            self.usr_elm_index = sample
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
        Helper function to return value from predict elm_prediction dict key
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
                       'Time': np.tile(np.arange(len(labels)), len(layers)),
                       'PC_1': df_arr[:, 0],
                       'PC_2': df_arr[:, 1]})
    fig1 = px.scatter(df, x='PC_1', y='PC_2', animation_frame='Layer', color='Label',
                      title='PC 1 vs PC 2 Over Hidden Neural Network Layers')
    fig2 = px.scatter(df, x='Time', y='PC_1', animation_frame='Layer', color='Label',
                      title='PC 1 vs Time Over Hidden Neural Network Layers')
    fig3 = px.scatter(df, x='Time', y='PC_2', animation_frame='Layer', color='Label',
                      title='PC 2 vs Time Over Hidden Neural Network Layers')
    fig1.show()
    fig2.show()
    fig3.show()

def plot_signal(pca: PCA):
    import matplotlib.transforms as mtransforms

    signal = pca.elm_predictions[0]['signals'][:, 0, 0]
    time = np.arange(len(signal))

    fig, ax = plt.subplots(1, 1)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(time, signal)
    ax.fill_between(time, 0, 1, where=pca.elm_predictions[0]['labels'], facecolor='red', alpha=0.5, transform=trans)
    ax.set_ylabel('Channel Activation (V)')
    ax.set_xlabel('Time (ns)')
    plt.show()

def plot_corr_layers(viz_obj):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for layer in ['conv', 'fc1', 'fc2']:
        pca = PCA(viz_obj, layer=layer)
        pca.perform_PCA()
        df = pca.correlate_pca(plot_type='line')
        ax1.plot(df['PC_1'], label=layer)
        ax2.plot(df['PC_2'], label=layer)

    ax1.set_title('PC 1', y=0.85, fontsize='x-large')
    ax2.set_title('PC 2', y=0.85, fontsize='x-large')
    fig.suptitle('Mean Correlation of PCs with BES Channels Across NN Layers', fontsize='xx-large')
    ax1.set_ylabel('Correlation Coef.', fontsize='medium')
    ax1.set_xlabel(None)
    ax2.set_ylabel('Correlation Coef.', fontsize='large')
    ax2.set_xlabel('Channel', fontsize='large')

    xticks = [str(i) for i in range(17, 25)]
    ax1.set_xticklabels([])
    ax2.set_xticklabels(xticks)
    ax1.legend(fontsize="large")
    plt.show()
    return None


def make_fft(in_block):
    pca = PCA(viz, layer='conv')
    pca.perform_PCA()
    kernels = pca.decompose_kernel()

    fft = []
    for i in range(3):
        fft_i = pca.FFT(kernels[i])
        fft_i = np.log(np.abs(np.fft.fftshift(fft_i)) ** 2)
        fft.append(fft_i)

    fig, axs = plt.subplots(3, 1)
    for i, ax in enumerate(axs):
        kernel = fft[i]
        ax.imshow(kernel.reshape(8, -1), cmap='jet')
        ax.set_title(f'kernel {i}')
        ax.set_ylabel(r'$\theta$')

        ax.set_xticks([])

    xticks = np.arange(0, np.shape(kernel.reshape(8, -1))[-1], 2)
    xlabels = [str((t + 1, r + 1)) for t in range(kernel.shape[0]) for r in range(8)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels[::2], rotation=45, ha='right')
    ax.set_xlabel('Tuple of (Freq, Wave Number)')

    fig.suptitle('3D FFT PCA Weighted Convolutional Kernel', fontsize='x-large')

    plt.tight_layout()
    plt.show()


def make_max_act(viz: Visualizations, num_filters=32, plots_per_page=4):
    layer = 'conv'
    # TODO: make work for fc layers
    for x in range(num_filters // plots_per_page):
        start = x * plots_per_page
        end = x * plots_per_page + plots_per_page
        filters = np.arange(start, end)
        fig, axs = plt.subplots(plots_per_page, 1)
        for filt, ax in zip(filters, axs):
            out = viz.max_activation(layer, filt, num_iter=40)[-1]
            ax.imshow(out.reshape(8, 8 * out.shape[0]))
            ax.set_ylabel(f'Filter {filt}')
            ax.set_xticks(np.arange(-.5, 8 * out.shape[0] + 0.5, 8), minor=True)
            ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
            ax.grid(visible=None, which='major', axis='both')

        fig.suptitle(f'Iterations of Activation Maximizing Inputs for {args.model_name} Model')
        plt.savefig(f'/home/jazimmerman/PycharmProjects/bes-edgeml-models/bes-edgeml-models/'
                    f'visualizations/max_act/all_filters_conv/filt{filters[0]}-{filters[-1]}.png')
    plt.show()


def make_gen_csi(viz: Visualizations, num_iterations: int = 25, use_fft: bool = False, blur: float = None,
                 l2_weight: float = 0):
    cls_lst = ['pre-ELM', 'ELM']
    out_lst = []
    model_out_lst = []
    for cls in cls_lst:
        gen_outs = []
        gen_model_outs = []
        if use_fft:
            for x in range(num_iterations):
                out, model_out, gen_i = \
                viz.generate_csi(target_class=cls, iterations=150, random_state=x, blur=blur, l2_weight=l2_weight)[-1]
                frame = np.fft.rfftn(out, axes=(1, 2, 0))
                gen_outs.append((np.log(np.abs(np.fft.fftshift(frame)) ** 2)))
                gen_model_outs.append(model_out)
        else:
            for x in range(num_iterations):
                out, model_out, gen_i = \
                viz.generate_csi(target_class=cls, iterations=150, random_state=x, blur=blur, l2_weight=l2_weight)[-1]
                gen_outs.append(out)
                gen_model_outs.append(model_out)

        out = np.array(gen_outs).mean(axis=0)
        model_out = np.array(gen_model_outs).mean(axis=0)

        # out = gen_i
        # model_out_lst.append(torch.sigmoid(viz.model(torch.tensor(gen_i).unsqueeze(0).unsqueeze(0))).item())

        out_lst.append(out)
        model_out_lst.append(model_out)

    _min, _max = np.amin(out_lst), np.amax(out_lst)
    fig, axs = plt.subplots(2, 1)
    for i in range(2):
        ax = axs[i]
        # model_output = torch.sigmoid(viz.model(torch.tensor(out_lst[i][np.newaxis, np.newaxis]))).item()
        im = ax.imshow(out_lst[i].transpose((1, 2, 0)).reshape(8, 8 * out.shape[0]), vmin=_min, vmax=_max)
        ax.set_xticks(np.arange(-.5, 8 * out.shape[0] + 0.5, 8), minor=True)
        ax.set_xticklabels([])
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        ax.grid(visible=None, which='major', axis='both')
        ax.set_title(cls_lst[i])
        ax.set_ylabel(f'Model Output:\n{model_out_lst[i]:0.2f}', rotation=0, labelpad=35)

    fig.suptitle(f'Fourier Transformed Generated Inputs for {args.model_name} Model')
    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')
    plt.show()

    return {cls_lst[0]: (model_out_lst[0], out_lst[0]), cls_lst[1]: (model_out_lst[1], out_lst[1])}


if __name__ == "__main__":
    args, parser = TestArguments().parse(verbose=True)
    LOGGER = get_logger(script_name=__name__,
            log_file=os.path.join(args.log_dir, f" output_logs_{args.model_name}_{args.filename_suffix}.log", ), )

    viz = Visualizations(args=args, logger=LOGGER)
    out = make_gen_csi(viz, use_fft=True, blur=0.1)
    pelm = torch.tensor(out['pre-ELM'][1]).unsqueeze(0).unsqueeze(0)
    elm = torch.tensor(out['ELM'][1]).unsqueeze(0).unsqueeze(0)
    print(torch.sigmoid(viz.model(pelm)).item(), torch.sigmoid(viz.model(elm)).item())
