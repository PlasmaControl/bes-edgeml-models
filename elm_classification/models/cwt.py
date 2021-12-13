import math
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy import signal
import torch
import torch.nn as nn

# Following class implements the Morlet wavelet
class Morlet(object):
    def __init__(self, w0=6):
        """Implementation of Morlet Wavelet.
        See: https://psl.noaa.gov/people/gilbert.p.compo/Torrence_compo1998.pdf
        for more details.

        This implementation is heavily inspired from:
        https://github.com/tomrunia/PyTorchWavelets/tree/master/wavelets_pytorch

        w0 is the nondimensional frequency. w0=6 is the default value used in
        paper.
        """
        self.w0 = w0
        if w0 == 6:
            # Reconstruction factor, see Torrence et al. 98, table 2.
            self.C_d = 0.776

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t: float, s: float = 1.0, complete: bool = True):
        """Complex Morlet wavelet, centered at 0. See Torrence and Compo 1998
        for more information.

        Args:
        -----
            t (float): Time. If `s` is not specified, this can be used as
                dimensionless time `t/s`.
            s (float, optional): Scaling factor. Defaults to 1.0.
            complete (bool, optional): If true, use the complete expression of
                the wavelet function including the correction term. Defaults to True.

        Returns:
        --------
            output (complex): Complex number giving the value of the Morlet wavelet
                at a given time.
        """
        x = t / s
        w = self.w0

        output = np.exp(1j * w * x)

        if complete:
            output -= np.exp(-0.5 * (w ** 2))

        output *= np.exp(-0.5 * (x ** 2)) * np.pi ** (-0.25)

        return output

    def fourier_wavelength(self, s: float):
        """Equivalent Fourier period for a given scale.

        Args:
        -----
            s (float): Scaling factor.
        """
        w = self.w0
        return (4 * np.pi * s) / (w + (2 + w ** 2) ** 0.5)

    def scale_from_wavelength(self, lambd_a: float):
        """Wavelet transform scale from a given fourier wavelength.

        Args:
        -----
            lambd_a (float): Fourier period.
        """
        w = self.w0
        return (lambd_a * (w + (2 + w ** 2) ** 0.5)) / (4 * np.pi)

    def frequency_domain(self, w: float, s: float = 1.0):
        """Frequency domain representation of the Morlet wavelet.

        Args:
        -----
            w (float): Angular frequency. If `s` is not defined (i.e. `s=1`),
            it can be used as the dimensionless angular frequency `w * s`.
            s (float, optional): Scaling factor. Defaults to 1.0.

        Returns:
        --------
            output (complex): Complex number giving the frequency domain value
                of the Morlet wavelet at a given frequency.
        """
        x = w * s
        w0 = self.w0

        # Heaviside step function
        Hw = 1 if w > 0 else 0

        output = np.pi ** (-0.25) * Hw * np.exp(-0.5 * (x - w0) ** 2)

        return output


class CWT(nn.Module):
    def __init__(
        self,
        dj: float = 0.0625,
        dt: float = 1 / 2048,
        wavelet: object = Morlet(),
        fmin: int = 20,
        fmax: int = 500,
        output_format: str = "Magnitude",
        trainable: bool = False,
        hop_length: int = 1,
    ):
        """This implementation of Continuous Wavelet Transform is heavily inspired
        from:
        https://github.com/tomrunia/PyTorchWavelets/tree/master/wavelets_pytorch

        Args:
        -----
            dj (float): [description]
            dt (float): [description]
            wavelet (object, optional): [description]. Defaults to Morlet().
            fmin (int, optional): [description]. Defaults to 20.
            fmax (int, optional): [description]. Defaults to 500.
            output_form (str, optional): [description]. Defaults to "Magnitude".
            trainable (bool, optional): [description]. Defaults to False.
            hop_length (int, optional): [description]. Defaults to 1.
        """
        super().__init__()
        self.wavelet = wavelet
        self.dj = dj
        self.dt = dt
        self.fmin = fmin
        self.fmax = fmax

        self.output_format = output_format
        self.trainable = trainable  # TODO make kernel a trainable parameter
        self.stride = (1, hop_length)
        self.padding = 0

        self._scale_minimum = self.compute_minimum_scale()

        self.signal_length = None
        self._channels = None

        self._scales = None
        self._kernel = None
        self._kernel_real = None
        self._kernel_imag = None

    def compute_optimal_scales(self):
        if self.signal_length is None:
            raise ValueError(
                "Please specify `signal_length` before computing optimal scales."
            )
        J = int(
            (1 / self.dj)
            * np.log2(self.signal_length * self.dt / self._scale_minimum)
        )
        print(f"Number of scales used: {J}")
        scales = np.array(
            self._scale_minimum * 2 ** (self.dj * np.arange(0, J))
        )

        # # get frequencies from the scales
        # frequencies = np.array(
        #     [1 / self.wavelet.fourier_wavelength(s) for s in scales]
        # )

        # # filter frequencies between [fmin, fmax]
        # if self.fmin:
        #     frequencies = frequencies[frequencies >= self.fmin]

        # if self.fmax:
        #     frequencies = frequencies[frequencies <= self.fmax]

        return scales

    def compute_minimum_scale(self):
        dt = self.dt

        def __solver(s):
            return self.wavelet.fourier_wavelength(s) - 2 * dt

        return fsolve(__solver, 1)[0]

    def _build_filters(self):
        self._scale_minimum = self.compute_minimum_scale()
        self._scales = self.compute_optimal_scales()

        self._filters = []

        for scale_idx, scale in enumerate(self._scales):
            # number of points needed to capture wavelet
            M = 10 * scale / self.dt

            # times to use, centered at zero
            t = torch.arange((-M + 1) / 2.0, (M + 1) / 2.0) * self.dt
            if len(t) % 2 == 0:
                t = t[0:-1]  # requires odd filter size

            # build wavelet corresponding to a scale and normalize
            norm = (self.dt / scale) ** 0.5
            filter_ = norm * self.wavelet(t, scale)
            self._filters.append(torch.conj(torch.flip(filter_, [-1])))

        self._pad_filters()

    def _pad_filters(self):
        filter_len = self._filters[-1].shape[0]
        padded_filters = []

        for f in self._filters:
            pad = (filter_len - f.shape[0]) // 2
            padded_filters.append(nn.functional.pad(f, (pad, pad)))

        self._filters = padded_filters

    def _build_wavelet_bank(self):
        self._build_filters()
        wavelet_bank = torch.stack(self._filters)
        print(f"Wavelet bank shape: {wavelet_bank.shape}")
        wavelet_bank = wavelet_bank.view(
            wavelet_bank.shape[0], 1, 1, wavelet_bank.shape[1]
        )
        return wavelet_bank

    # Dynamically pad input x with 'SAME' padding for conv with specified args
    def _pad_same(
        self,
        x,
        k: List[int],
        s: List[int],
        d: List[int] = (1, 1),
        value: float = 0,
    ):
        ih, iw = x.size()[-2:]
        pad_h, pad_w = self._get_same_padding(
            ih, k[0], s[0], d[0]
        ), self._get_same_padding(iw, k[1], s[1], d[1])
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(
                x,
                [
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                ],
                value=value,
            )
        return x

    # Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
    def _get_same_padding(self, x: int, k: int, s: int, d: int):
        return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

    def _conv2d_same(
        self,
        x,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
    ):
        x = self._pad_same(x, weight.shape[-2:], stride, dilation)
        return nn.functional.conv2d(
            x, weight, bias, stride, (0, 0), dilation, groups
        )

    def forward(self, x):
        if self.signal_length is None:
            self.signal_length = x.shape[-1]
            self.channels = x.shape[-2]
            self._scales = self.compute_optimal_scales()
            self._kernel = self._build_wavelet_bank()

            if self._kernel.is_complex():
                self._kernel_real = self._kernel.real
                self._kernel_imag = self._kernel.imag

        x = x.unsqueeze(1)

        if self._kernel.is_complex():
            if (
                x.dtype != self._kernel_real.dtype
                or x.device != self._kernel_real.device
            ):
                self._kernel_real = self._kernel_real.to(
                    device=x.device, dtype=x.dtype
                )
                self._kernel_imag = self._kernel_imag.to(
                    device=x.device, dtype=x.dtype
                )

            # Strides > 1 not yet supported for "same" padding
            # output_real = nn.functional.conv2d(
            #     x, self._kernel_real, padding=self.padding, stride=self.stride
            # )
            # output_imag = nn.functional.conv2d(
            #     x, self._kernel_imag, padding=self.padding, stride=self.stride
            # )
            output_real = self._conv2d_same(
                x, self._kernel_real, stride=self.stride
            )
            output_imag = self._conv2d_same(
                x, self._kernel_imag, stride=self.stride
            )
            output_real = torch.transpose(output_real, 1, 2)
            output_imag = torch.transpose(output_imag, 1, 2)

            if self.output_format == "Magnitude":
                return torch.sqrt(output_real ** 2 + output_imag ** 2)
            else:
                return torch.stack([output_real, output_imag], -1)

        else:
            if x.device != self._kernel.device:
                self._kernel = self._kernel.to(device=x.device, dtype=x.dtype)

            # output = nn.functional.conv2d(
            #     x, self._kernel, padding=self.padding, stride=self.stride
            # )
            output = self._conv2d_same(x, self._kernel, stride=self.stride)
            return torch.transpose(output, 1, 2)


if __name__ == "__main__":
    x = np.load("single_elm_event.npy")
    x = torch.tensor(x)
    x = torch.unsqueeze(x, 0)
    # cwt = CWT()
    # y = cwt(x)
    # bank = cwt._build_wavelet_bank()
    # print(torch.abs(bank))
    # print(y.shape)

    # sig_pt = torch.tensor(sig, dtype=torch.float32)
    # sig_pt = torch.stack([sig_pt] * 3)  # 3 channels
    # sig_pt = torch.stack([sig_pt] * 32)  # Batch size of 32
    print(x.shape)

    pycwt = CWT(dt=1 / 400)
    out = pycwt(x)
    print(out.shape)
    y = torch.transpose(out, 2, 1)
    print(y.shape)
    # plt.imshow(out[0, 0].numpy(), aspect="auto")
    plt.figure(figsize=(8, 6))
    y_arr = y[0, :, :, 0].numpy()
    y_arr = np.abs(y_arr)
    print(y_arr.shape)
    print(y_arr)
    plt.imshow(
        y_arr,
        cmap="PuOr",
        aspect="auto",
        vmax=abs(y_arr).max(),
        vmin=-abs(y_arr).max(),
    )
    plt.colorbar()
    plt.title("BES Ch:1", fontsize=14)
    plt.show()
