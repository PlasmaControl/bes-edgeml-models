import numpy as np
from pytest import param
import torch
import torch.nn as nn
import logging
import inspect

import pywt
from pytorch_wavelets.dwt.transform1d import DWT1DForward

try:
    from . import dct
except ImportError:
    from bes_ml.main import dct

class _Base_Features(nn.Module):

    def __init__(
        self,
        signal_window_size: int = 64,
        spatial_maxpool_size: int = 1,
        time_interval: int = 1,  # time domain interval (i.e. time[::interval])
        subwindow_size: int = -1,  # power of 2, or -1 (default) for full signal window
        negative_slope: float = 1e-3,
        dropout_rate: float = 0.1,
        logger: logging.Logger = None,
        **kwargs,
    ):
        super().__init__()

        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())

        self._print_kwargs(cls=_Base_Features, copy_locals=locals().copy())

        # spatial maxpool
        self.spatial_maxpool_size = spatial_maxpool_size
        assert self.spatial_maxpool_size in [1, 2, 4]
        self.maxpool = None
        if self.spatial_maxpool_size > 1:
            self.maxpool = nn.MaxPool3d(
                kernel_size=[1, self.spatial_maxpool_size, self.spatial_maxpool_size],
            )

        # signal window
        self.signal_window_size = signal_window_size
        assert np.log2(self.signal_window_size) % 1 == 0  # ensure power of 2

        # time slice interval
        self.time_interval = time_interval
        assert self.time_interval >= 1
        assert np.log2(self.time_interval) % 1 == 0  # ensure power of 2
        self.time_points = self.signal_window_size // self.time_interval

        # subwindows
        self.subwindow_size = subwindow_size
        if self.subwindow_size == -1:
            self.subwindow_size = self.time_points
        assert np.log2(self.subwindow_size) % 1 == 0  # ensure power of 2
        assert self.subwindow_size <= self.time_points
        self.subwindow_nbins = self.time_points // self.subwindow_size
        assert self.subwindow_nbins >= 1
        
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout = nn.Dropout3d(p=dropout_rate)

        self.num_kernels = None  # set in subclass
        self.conv = None  # set in subclass

    @classmethod
    def _get_init_kwargs_and_defaults(cls) -> dict:
        init_signature = inspect.signature(cls)
        init_kwargs_and_defaults = {parameter.name: parameter.default 
            for parameter in init_signature.parameters.values()}
        return init_kwargs_and_defaults

    def _print_kwargs(
        self, 
        cls = None, 
        copy_locals: dict = None,
    ):
        # print input keyword arguments
        self.logger.info(f"Class `{cls.__name__}` keyword arguments:")
        parent_kwargs = cls._get_init_kwargs_and_defaults()
        for key, default_value in parent_kwargs.items():
            if key in ['kwargs', 'logger']: continue
            value = copy_locals[key]
            if value == default_value:
                self.logger.info(f"  {key:22s}:  {value}")
            else:
                self.logger.info(f"  {key:22s}:  {value} (default {default_value})")

    def _time_interval_and_maxpool(self, x: torch.Tensor) -> torch.Tensor:
        if self.time_interval > 1:
            x = x[:, :, ::self.time_interval, :, :]
        if self.maxpool:
            x = self.maxpool(x)
        return x

    def _dropout_relu_flatten(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.relu(self.dropout(x)), 1)


class Dense_Features(_Base_Features):
    def __init__(
        self,
        dense_num_kernels: int = 8,
        **kwargs,
    ):
        """
        Use the raw BES channels values as features. This function takes in a 5-dimensional
        tensor of size: `(N, 1, signal_window_size, 8, 8)`, N=batch_size and
        performs maxpooling to downsample the spatial dimension by half, perform a
        3-d convolution with a filter size identical to the spatial dimensions of the
        input to avoid the sliding of the kernel over the input. Finally, a feature map
        is generated that can be concatenated with other features.

        Args:
        -----
            args (argparse.Namespace): Command line arguments containing the information
                about signal_window.
            dropout_rate (float, optional): Fraction of total hidden units that will
                be turned off for drop out. Defaults to 0.2.
            negative_slope (float, optional): Slope of LeakyReLU activation for negative
                `x`. Defaults to 0.02.
            maxpool_size (int, optional): Size of the kernel used for maxpooling. Use
                0 to skip maxpooling. Defaults to 2.
            num_kernels (int, optional): Dimensionality of the output space.
                Essentially, it gives the number of output kernels after convolution.
                Defaults to 10.
        """
        super().__init__(**kwargs)

        self._print_kwargs(cls=self.__class__, copy_locals=locals().copy())

        # filters per subwindow
        self.num_kernels = dense_num_kernels

        filter_size = (
            self.subwindow_size,
            8 // self.spatial_maxpool_size,
            8 // self.spatial_maxpool_size,
        )

        # list of conv. filter banks (each with self.num_kernels) with size self.subwindow_bins
        self.conv = nn.ModuleList(
            [
                nn.Conv3d(
                    in_channels=1,
                    out_channels=self.num_kernels,
                    kernel_size=filter_size,
                ) for i_subwindow in range(self.subwindow_nbins)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._time_interval_and_maxpool(x)
        x_new_size = [
            x.shape[0],
            self.num_kernels,
            self.subwindow_nbins,
            1,
            1,
        ]
        x_new = torch.empty(size=x_new_size, dtype=x.dtype, device=x.device)
        for i_bin in range(self.subwindow_nbins):
            i_start = i_bin * self.subwindow_size
            i_stop = (i_bin+1) * self.subwindow_size
            if torch.any(torch.isnan(self.conv[i_bin].weight)) or torch.any(torch.isnan(self.conv[i_bin].bias)):
                assert False
            x_new[:, :, i_bin:i_bin+1, :, :] = self.conv[i_bin](
                x[:, :, i_start:i_stop, :, :]
            )
        x = self._dropout_relu_flatten(x_new)
        return x


class CNN_Features(_Base_Features):

    def __init__(
        self, 
        layer1_num_kernels: int = 8,
        layer1_kernel_time_size: int = 8,
        layer1_kernel_spatial_size: int = 3,
        layer1_maxpool_time_size: int = 1,
        layer1_maxpool_spatial_size: int = 2,
        layer2_num_kernels: int = 8,
        layer2_kernel_time_size: int = 8,
        layer2_kernel_spatial_size: int = 3,
        layer2_maxpool_time_size: int = 1,
        layer2_maxpool_spatial_size: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # CNN only valid with subwindow_size == time_points == signal_window_size
        assert self.subwindow_size == self.signal_window_size
        assert self.time_interval == 1
        assert self.subwindow_nbins == 1
        assert self.time_points == self.signal_window_size
        assert self.spatial_maxpool_size == 1

        input_shape = (1, self.signal_window_size, 8, 8)

        def test_bad_shape(shape):
            assert np.any(np.array(shape)<1), f"Bad shape: {shape}"

        self.layer1_num_kernels = layer1_num_kernels
        self.layer1_kernel_time_size = layer1_kernel_time_size
        self.layer1_kernel_spatial_size = layer1_kernel_spatial_size
        self.layer1_maxpool_time_size = layer1_maxpool_time_size
        self.layer1_maxpool_spatial_size = layer1_maxpool_spatial_size
        self.layer2_num_kernels = layer2_num_kernels
        self.layer2_kernel_time_size = layer2_kernel_time_size
        self.layer2_kernel_spatial_size = layer2_kernel_spatial_size
        self.layer2_maxpool_time_size = layer2_maxpool_time_size
        self.layer2_maxpool_spatial_size = layer2_maxpool_spatial_size

        self.layer1_conv = nn.Conv3d(
            in_channels=1,
            out_channels=self.layer1_num_kernels,
            kernel_size=(
                self.layer1_kernel_time_size,
                self.layer1_kernel_spatial_size,
                self.layer1_kernel_spatial_size,
            ),
            stride=(1, 1, 1),
            padding=((self.layer1_kernel_time_size-1)//2, 0, 0),
        )

        output_shape = [
            self.layer1_num_kernels,
            input_shape[1],
            input_shape[2]-(self.layer1_kernel_spatial_size-1),
            input_shape[3]-(self.layer1_kernel_spatial_size-1),
        ]
        test_bad_shape(output_shape)

        self.layer1_maxpool = nn.MaxPool3d(
            kernel_size=(
                self.layer1_maxpool_time_size,
                self.layer1_maxpool_spatial_size,
                self.layer1_maxpool_spatial_size,
            ),
        )

        output_shape = [
            output_shape[0],
            output_shape[1] // self.layer1_maxpool_time_size,
            output_shape[2] // self.layer1_maxpool_spatial_size,
            output_shape[3] // self.layer1_maxpool_spatial_size,
        ]
        test_bad_shape(output_shape)

        self.layer2_conv = nn.Conv3d(
            in_channels=self.layer1_num_kernels,
            out_channels=self.layer2_num_kernels,
            kernel_size=(
                self.layer2_kernel_time_size,
                self.layer2_kernel_spatial_size,
                self.layer2_kernel_spatial_size,
            ),
            stride=(1, 1, 1),
            padding=((self.layer2_kernel_time_size-1)//2, 0, 0),
        )

        output_shape = [
            self.layer2_num_kernels,
            output_shape[1],
            output_shape[2] - (self.layer2_kernel_spatial_size-1),
            output_shape[3] - (self.layer2_kernel_spatial_size-1),
        ]
        test_bad_shape(output_shape)

        self.layer2_maxpool = nn.MaxPool3d(
            kernel_size=(
                self.layer2_maxpool_time_size,
                self.layer2_maxpool_spatial_size,
                self.layer2_maxpool_spatial_size,
            ),
        )

        output_shape = [
            output_shape[0],
            output_shape[1] // self.layer2_maxpool_time_size,
            output_shape[2] // self.layer2_maxpool_spatial_size,
            output_shape[3] // self.layer2_maxpool_spatial_size,
        ]
        test_bad_shape(output_shape)

        self.num_kernels = np.prod(output_shape, dtype=int)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.dropout(self.layer1_conv(x)))
        x = self.layer1_maxpool(x)
        x = self.relu(self.dropout(self.layer2_conv(x)))
        x = self.layer2_maxpool(x)
        return torch.flatten(x, 1)



class FFT_Features(_Base_Features):
    def __init__(
        self, 
        fft_nbins: int = 4,
        fft_num_kernels: int = 8,
        **kwargs,
    ):
        """
        Use the raw BES channels values as input and perform a Fast Fourier Transform
        to input signals. This function takes in a 5-dimensional
        tensor of size: `(N, 1, signal_window_size, 8, 8)`, N=batch_size and
        performs a FFT followed by an absolute value of the input tensor. It
        then performs a 3-d convolution with a filter size identical to the spatial
        dimensions of the input so that the receptive field is the same
        size as input in both spatial and temporal axes. Again, we will use the
        feature map and combine it with other features before feeding it into a
        classifier.

        Args:
        -----
            args (argparse.Namespace): Command line arguments containing the information
                about signal_window.
            dropout_rate (float, optional): Fraction of total hidden units that will
                be turned off for drop out. Defaults to 0.2.
            negative_slope (float, optional): Slope of LeakyReLU activation for negative
                `x`. Defaults to 0.02.
            num_kernels (int, optional): Dimensionality of the output space.
                Essentially, it gives the number of output kernels after convolution.
                Defaults to 10.
        """
        super().__init__(**kwargs)

        self.fft_nbins = fft_nbins
        assert np.log2(self.fft_nbins) % 1 == 0  # ensure power of 2

        self.nfft = self.subwindow_size // self.fft_nbins
        self.nfreqs = self.nfft // 2 + 1

        self.num_kernels = fft_num_kernels

        filter_size = (
            self.nfreqs, 
            8 // self.spatial_maxpool_size, 
            8 // self.spatial_maxpool_size,
        )

        # list of conv. filter banks (each with self.num_kernels) with size self.subwindow_bins
        self.conv = nn.ModuleList(
            [
                nn.Conv3d(
                    in_channels=1,
                    out_channels=self.num_kernels,
                    kernel_size=filter_size,
                ) for _ in range(self.subwindow_nbins)
            ]
        )

    def forward(self, x):
        x = x.to(self.args.device)  # needed for PowerPC architecture
        x = self._time_interval_and_maxpool(x)
        fft_features_size = [
            x.shape[0],
            self.subwindow_nbins,
            self.num_kernels,
            1,
            1,
            1,
        ]
        fft_features = torch.empty(size=fft_features_size, dtype=x.dtype, device=x.device)
        for i_sw in torch.arange(self.subwindow_nbins):
            fft_bins_size = [
                x.shape[0],
                self.fft_nbins,
                self.nfreqs,
                x.shape[3],
                x.shape[4],
            ]
            fft_bins = torch.empty(size=fft_bins_size, dtype=x.dtype, device=x.device)
            x_sw = x[:, :, i_sw*self.subwindow_size:(i_sw+1)*self.subwindow_size, :, :]
            for i_bin in torch.arange(self.fft_nbins):
                fft_bins[:, i_bin: i_bin + 1, :, :, :] = torch.abs(
                    torch.fft.rfft(
                        x_sw[:, :, i_bin * self.nfft:(i_bin+1) * self.nfft, :, :], 
                        dim=2,
                    )
                )
            fft_sw = torch.mean(fft_bins, dim=1, keepdim=True)
            fft_sw_features = self.conv[i_sw](fft_sw)
            fft_features[:, i_sw:i_sw+1, :, :, :, :] = \
                torch.unsqueeze(fft_sw_features, 1)
        output_features = self._dropout_relu_flatten(fft_features)
        return output_features


class DCT_Features(_Base_Features):
    def __init__(
        self, 
        dct_nbins: int = 2,
        dct_num_kernels: int = 8,
        **kwargs,
    ):
        """
        Use the raw BES channels values as input and perform a Fast Fourier Transform
        to input signals. This function takes in a 5-dimensional
        tensor of size: `(N, 1, signal_window_size, 8, 8)`, N=batch_size and
        performs a DCT followed by an absolute value of the input tensor. It
        then performs a 3-d convolution with a filter size identical to the spatial
        dimensions of the input so that the receptive field is the same
        size as input in both spatial and temporal axes. Again, we will use the
        feature map and combine it with other features before feeding it into a
        classifier.

        Args:
        -----
            args (argparse.Namespace): Command line arguments containing the information
                about signal_window.
            dropout_rate (float, optional): Fraction of total hidden units that will
                be turned off for drop out. Defaults to 0.2.
            negative_slope (float, optional): Slope of LeakyReLU activation for negative
                `x`. Defaults to 0.02.
            num_kernels (int, optional): Dimensionality of the output space.
                Essentially, it gives the number of output kernels after convolution.
                Defaults to 10.
        """
        super().__init__(**kwargs)

        self.dct_nbins = dct_nbins
        assert np.log2(self.dct_nbins) % 1 == 0  # ensure power of 2

        self.ndct = self.subwindow_size // self.dct_nbins
        # self.nfreqs = self.ndct // 2 + 1
        self.nfreqs = self.ndct

        self.num_kernels = dct_num_kernels

        filter_size = (
            self.nfreqs, 
            8 // self.spatial_maxpool_size, 
            8 // self.spatial_maxpool_size,
        )

        # list of conv. filter banks (each with self.num_kernels) with size self.subwindow_bins
        self.conv = nn.ModuleList(
            [
                nn.Conv3d(
                    in_channels=1,
                    out_channels=self.num_kernels,
                    kernel_size=filter_size,
                ) for _ in range(self.subwindow_nbins)
            ]
        )

    def forward(self, x):
        x = x.to(self.args.device)  # needed for PowerPC architecture
        x = self._time_interval_and_maxpool(x)
        dct_features_size = [
            x.shape[0],
            self.subwindow_nbins,
            self.num_kernels,
            1,
            1,
            1,
        ]
        dct_features = torch.empty(size=dct_features_size, dtype=x.dtype, device=x.device)
        for i_sw in torch.arange(self.subwindow_nbins):
            dct_bins_size = [
                x.shape[0],
                self.dct_nbins,
                self.nfreqs,
                x.shape[3],
                x.shape[4],
            ]
            dct_bins = torch.empty(size=dct_bins_size, dtype=x.dtype, device=x.device)
            x_subwindow = x[:, :, i_sw*self.subwindow_size:(i_sw+1)*self.subwindow_size, :, :]
            for i_bin in torch.arange(self.dct_nbins):
                dct_bins[:, i_bin: i_bin + 1, :, :, :] = dct.dct_3d(
                    x_subwindow[:, :, i_bin * self.ndct:(i_bin+1) * self.ndct, :, :]
                )
            dct_sw = torch.mean(dct_bins, dim=1, keepdim=True)
            dct_sw_features = torch.unsqueeze( self.conv[i_sw](dct_sw), 1)
            dct_features[:, i_sw:i_sw+1, :, :, :, :] = dct_sw_features
        output_features = self._dropout_relu_flatten(dct_features)
        return output_features
            

class DWT_Features(_Base_Features):
    def __init__(
        self, 
        dwt_wavelet: str = 'db4',
        dwt_level: int = -1,
        dwt_num_kernels: int = 8,
        **kwargs,
    ):
        """
        Use features from the output of 1D discrete wavelet transform. Based on:
        https://github.com/fbcotter/pytorch_wavelets
        The model architecture is similar to the `RawFeatureModel`. This model takes in a 6-dimensional
        tensor of size: `(N, 1, signal_window_size, n_scales, 8, 8)`, where `N`=batch_size, and
        `n_scales`=number of different scales used (which are equal to `signal_window_size`).
        For each signal block, only the scales and BES channels for the leading time
        steps are used as model input which is a 5-dimensional tensor of size (`N, 1, n_scales, 8, 8)`.
        The model then performs a 3-d convolution with a filter size identical to
        the spatial dimensions of the input so that the receptive field is the same
        size as input in both spatial and temporal axes. In the end, we will have
        a feature map that can be concatenated with other features.

        Args:
        -----
            args (argparse.Namespace): Command line arguments containing the information
                about signal_window.
            dropout_rate (float, optional): Fraction of total hidden units that will
                be turned off for drop out. Defaults to 0.2.
            negative_slope (float, optional): Slope of LeakyReLU activation for negative
                `x`. Defaults to 0.02.
            maxpool_size (int, optional): Size of the kernel used for maxpooling. Use
                0 to skip maxpooling. Defaults to 2.
            num_kernels (int, optional): Dimensionality of the output space.
                Essentially, it gives the number of output kernels after convolution.
                Defaults to 10.
        """
        super().__init__(**kwargs)

        self.dwt_wavelet = dwt_wavelet
        self.dwt_level = dwt_level

        max_level = pywt.dwt_max_level(
            self.subwindow_size, 
            self.dwt_wavelet
        )

        if dwt_level == -1:
            dwt_level = max_level

        assert dwt_level <= max_level

        # DWT and sample calculation to get new time domain size
        self.dwt = DWT1DForward(
            wave=dwt_wavelet,
            J=dwt_level,
            mode="reflect",
        )
        x_tmp = torch.empty(1, 1, self.subwindow_size)
        x_lo, x_hi = self.dwt(x_tmp)
        self.dwt_output_length = sum(
            [x_lo.shape[2]] + [hi.shape[2] for hi in x_hi]
        )

        self.num_kernels = dwt_num_kernels

        filter_size = (
            self.dwt_output_length, 
            8 // self.spatial_maxpool_size, 
            8 // self.spatial_maxpool_size,
        )

        # list of conv. filter banks (each with self.num_kernels) with size self.subwindow_bins
        self.conv = nn.ModuleList(
            [
                nn.Conv3d(
                    in_channels=1,
                    out_channels=self.num_kernels,
                    kernel_size=filter_size,
                ) for _ in range(self.subwindow_nbins)
            ]
        )

    def forward(self, x):
        x = self._time_interval_and_maxpool(x)
        dwt_features_size = [
            x.shape[0],
            self.subwindow_nbins,
            self.num_kernels,
            1,
            1,
            1,
        ]
        dwt_features = torch.empty(size=dwt_features_size, dtype=x.dtype, device=x.device)
        for i_sw in torch.arange(self.subwindow_nbins):
            x_sw = x[:, :, i_sw*self.subwindow_size:(i_sw+1)*self.subwindow_size, :, :]
            dwt_sw_size = [
                x_sw.shape[0],
                x_sw.shape[1],
                self.dwt_output_length,
                x_sw.shape[3],
                x_sw.shape[4],
            ]
            dwt_sw = torch.empty(dwt_sw_size, dtype=x.dtype, device=x.device)
            for i_batch in torch.arange(x.shape[0]):
                x_tmp = (
                    x_sw[i_batch, 0, :, :, :]
                    .permute(1, 2, 0)
                )  # make 3D and move time dim. to last
                x_lo, x_hi = self.dwt(x_tmp)  # multi-level DWT on last dim.
                coeff = [x_lo] + [hi for hi in x_hi]  # make list of coeff.
                dwt_sw[i_batch, 0, :, :, :] =  (
                    torch.cat(coeff, dim=2)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )  # concat list in dwt coeff. dim, unpermute, and expand
            dwt_sw_features = self.conv[i_sw](dwt_sw)
            dwt_features[:, i_sw:i_sw+1, :, :, :, :] = \
                torch.unsqueeze(dwt_sw_features, 1)
        output_features = self._dropout_relu_flatten(dwt_features)
        return output_features


class Multi_Features_Model(nn.Module):
    def __init__(
        self,
        dense_num_kernels: int = 8,
        fft_num_kernels: int = 0,
        dct_num_kernels: int = 0,
        dwt_num_kernels: int = 0,
        cnn_layer1_num_kernels: int = 0,
        cnn_layer2_num_kernels: int = 0,
        mlp_layer1_size: int = 32,
        mlp_layer2_size: int = 16,
        mlp_output_size: int = 1,
        negative_slope: float = 1e-3,
        dropout_rate: float = 0.1,
        logger: logging.Logger = None,
        **kwargs,
    ):
        """Encapsulate all the feature models to create a composite model that
        uses all the feature maps. It takes in the class instances of
        `RawFeatureModel`, `FFTFeatureModel` and `CWTFeatureModel` along with the
        dropout rate and negative slope of the LeakyReLU activation. Once all
        the feature maps are computed, it concatenates them together and pass them
        through a couple of fully connected layers wich act as the classifier.

        Args:
        -----
            args (argparse.Namespace): Command line arguments.
        """
        super().__init__()

        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            logger.addHandler(logging.StreamHandler())

        self.dense_features = (
            Dense_Features(
                dense_num_kernels=dense_num_kernels, 
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                logger=logger, 
                **kwargs,
            )
            if dense_num_kernels > 0
            else None
        )
        self.fft_features = (
            FFT_Features(
                fft_num_kernels=fft_num_kernels, 
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                logger=logger, 
                **kwargs,
            )
            if fft_num_kernels > 0
            else None
        )
        self.dct_features = (
            DCT_Features(
                dct_num_kernels=dct_num_kernels, 
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                logger=logger, 
                **kwargs,
            )
            if dct_num_kernels > 0
            else None
        )
        self.dwt_features = (
            DWT_Features(
                dwt_num_kernels=dwt_num_kernels, 
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                logger=logger, 
                **kwargs,
            )
            if dwt_num_kernels > 0
            else None
        )
        self.cnn_features = (
            CNN_Features(
                layer1_num_kernels=cnn_layer1_num_kernels,
                layer2_num_kernels=cnn_layer2_num_kernels,
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                logger=logger, 
                **kwargs,
            )
            if cnn_layer1_num_kernels > 0 and cnn_layer2_num_kernels > 0
            else None
        )

        self.total_features = 0
        for features in [
            self.dense_features,
            self.fft_features,
            self.dwt_features,
            self.dct_features,
            self.cnn_features,
        ]:
            if features is not None:
                self.total_features += features.num_kernels * features.subwindow_nbins
        logger.info(f"Total features: {self.total_features}")

        self.mlp_layer1 = nn.Linear(in_features=self.total_features, out_features=mlp_layer1_size)
        self.mlp_layer2 = nn.Linear(in_features=mlp_layer1_size, out_features=mlp_layer2_size)
        self.mlp_layer3 = nn.Linear(in_features=mlp_layer2_size, out_features=mlp_output_size)
        logger.info(f"MLP layer 1 size: {mlp_layer1_size}")
        logger.info(f"MLP layer 2 size: {mlp_layer2_size}")
        logger.info(f"MLP output size: {mlp_output_size}")

        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        dense_features = self.dense_features(x) if self.dense_features else None
        fft_features = self.fft_features(x) if self.fft_features else None
        dwt_features = self.dwt_features(x) if self.dwt_features else None
        dct_features = self.dct_features(x) if self.dct_features else None
        cnn_features = self.cnn_features(x) if self.cnn_features else None

        all_features = [
            features
            for features in [dense_features, fft_features, dwt_features, dct_features, cnn_features]
            if features is not None
        ]

        x = torch.cat(all_features, dim=1)

        x = self.relu(self.dropout(self.mlp_layer1(x)))
        x = self.relu(self.dropout(self.mlp_layer2(x)))
        x = self.mlp_layer3(x)

        assert not torch.any(torch.isnan(x)), f"NaN in result"

        return x

if __name__=='__main__':
    m = Multi_Features_Model()