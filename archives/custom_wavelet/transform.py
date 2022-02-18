from typing import Union

import torch
import numpy as np

try:
    import cwt
except ImportError:
    from . import cwt


def continuous_wavelet_transform(
    signal_window_size: int,
    scales: Union[np.ndarray, list],
    signal: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    morl = cwt.Morlet()
    pycwt = cwt.CWT(scales=scales, wavelet=morl)
    wavelet_bank = pycwt.build_wavelet_bank()
    assert signal_window_size == wavelet_bank[-1, :].shape[0], (
        f"Signal window size ({signal_window_size}) does not match the number of time points needed "
        f"({wavelet_bank[-1, :].shape[0]}) for the largest scale ({scales[-1]})!\n"
        f"Maximum number of time points are given by `10 * max_scale / dt` where dt: {pycwt.dt}."
    )
    transformed_signal = []
    signal = signal.permute(0, 1, 3, 2, 4)
    for i, scale in enumerate(scales):
        cwt_filter = wavelet_bank[i, :]
        cwt_filter_real = torch.real(cwt_filter).to(device)
        transformed = torch.matmul(cwt_filter_real, signal)
        transformed_signal.append(transformed)
    transformed_signal = torch.stack(transformed_signal)
    transformed_signal = transformed_signal.permute(1, 2, 0, 3, 4)
    return transformed_signal


if __name__ == "__main__":
    sws = 512
    scales = [1, 2, 4, 8, 16]
    x = torch.rand(1, 1, sws, 8, 8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_tsfmd = continuous_wavelet_transform(sws, scales, x, device)
    print(x_tsfmd.shape)
