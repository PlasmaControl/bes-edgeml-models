from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Morlet(object):
    def __init__(self, w0=6):
        """w0 is the nondimensional frequency constant. If this is
        set too low then the wavelet does not sample very well: a
        value over 5 should be ok; Terrence and Compo set it to 6.

        https://psl.noaa.gov/people/gilbert.p.compo/Torrence_compo1998.pdf
        """
        self.w0 = w0
        if w0 == 6:
            # value of C_d from the paper
            self.C_d = 0.776

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0, complete=True):
        """
        Complex Morlet wavelet, centred at zero.
        Parameters
        ----------
        t : float
            Time. If s is not specified, this can be used as the
            non-dimensional time t/s.
        s : float
            Scaling factor. Default is 1.
        complete : bool
            Whether to use the complete or the standard version.
        Returns
        -------
        out : complex
            Value of the Morlet wavelet at the given time
        See Also
        --------
        scipy.signal.gausspulse
        Notes
        -----
        The standard version::
            pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))
        This commonly used wavelet is often referred to simply as the
        Morlet wavelet.  Note that this simplified version can cause
        admissibility problems at low values of `w`.
        The complete version::
            pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))
        The complete version of the Morlet wavelet, with a correction
        term to improve admissibility. For `w` greater than 5, the
        correction term is negligible.
        Note that the energy of the return wavelet is not normalised
        according to `s`.
        The fundamental frequency of this wavelet in Hz is given
        by ``f = 2*s*w*r / M`` where r is the sampling rate.
        """
        w = self.w0

        x = t / s

        output = np.exp(1j * w * x)

        if complete:
            output -= np.exp(-0.5 * (w ** 2))

        output *= np.exp(-0.5 * (x ** 2)) * np.pi ** (-0.25)

        return output


class CWT(nn.Module):
    def __init__(
        self,
        scales: Union[np.ndarray, list],
        dt: float = 1 / 2,
        wavelet: Morlet = Morlet(),
    ):
        super(CWT, self).__init__()
        self.scales = scales
        self.dt = dt
        self.wavelet = wavelet

        self._scale_minimum = 1
        self._filters = []

    def _build_filters(self):
        for scale_idx, scale in enumerate(self.scales):
            # number of points needed to capture wavelet (arbitrary)
            M = scale / self.dt
            # Times to use, centred at zero
            t = torch.arange((-M + 1) / 2.0, (M + 1) / 2.0) * self.dt
            if len(t) % 2 == 0:
                t = t[0:-1]  # requires odd filter size
            # Sample wavelet and normalise
            norm = (self.dt / scale) ** 0.5
            filter_ = norm * self.wavelet(t, scale)
            self._filters.append(torch.conj(torch.flip(filter_, [-1])))
        self._pad_filters()
        # return self.filters

    def _pad_filters(self):
        filter_len = self._filters[-1].shape[0]
        padded_filters = []

        for f in self._filters:
            pad_length = (filter_len - f.shape[0] + 1)
            pad_filter = F.pad(f, (pad_length, 0))
            # pad_filter = torch.abs(pad_filter)
            padded_filters.append(pad_filter)

        self._filters = padded_filters

    def build_wavelet_bank(self):
        self._build_filters()
        wavelet_bank = torch.stack(self._filters)
        return wavelet_bank


if __name__ == "__main__":
    t = np.linspace(0, 512, 100)
    scales = [1, 2, 4, 8, 16]
    morl = Morlet()
    pycwt = CWT(scales=scales)
    bank = pycwt.build_wavelet_bank()
    print(bank.shape)
    plt.figure(figsize=(8, 6), dpi=100)
    for i, scale in enumerate(scales):
        cwt_filter = bank[i, :]
        cwt_filter_real = torch.real(cwt_filter)
        # cwt_filter_imag = torch.imag(cwt_filter)
        plt.plot(cwt_filter_real.numpy(), label=f'Filter {i + 1} real, scale: {scale}')
        # plt.plot(cwt_filter_imag.numpy(), ls='--', label=f'Filter {i + 1} imag, scale: {scale}')
    plt.legend(loc='lower left', fontsize=8)
    plt.show()

