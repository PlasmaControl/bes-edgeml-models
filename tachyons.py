from matplotlib import pyplot as plt
import numpy as np
import pywt


def plot_unbatched(ax):
    arr = np.zeros((max_scale * mult,))
    idx = np.linspace(0, 1, max_scale)
    arr[-max_scale:] = np.cos(20 * np.pi * idx)

    cwt, freqs = pywt.cwt(arr, scales=[1024], wavelet="morl", axis=0)
    first_nonzero = (np.around(cwt, decimals=4) != 0).argmax()
    print('First non-zero unbatched: ', first_nonzero)

    ticks = np.arange(0, arr.size, max_scale)

    ax.plot(np.arange(arr.size), cwt[0] / cwt[0].max(), label='CWT Scale 1024 (Normalized)')
    ax.plot(np.arange(arr.size), arr, label='Input Signal')
    ax.vlines(ticks, -0.1, 0.1, color='k', zorder=5)

    ax.set_title('Wavelet Transform on Entire Dataset')
    ax.annotate('First nonzero (rounded to 4th decimal)',
                (first_nonzero, 0),
                xytext=(first_nonzero, 0.5),
                ha='center',
                va='bottom',
                arrowprops=dict(facecolor='black', shrink=0.05))

    ax.legend()

    return ax, cwt


def plot_batched(ax):
    arr = np.zeros((max_scale * mult,))
    idx = np.linspace(0, 1, max_scale)
    arr[-max_scale:] = np.cos(20 * np.pi * idx)

    leading_freqs = [0] * max_scale
    for stop, _ in enumerate(arr[max_scale:], start=max_scale):

        start = stop - max_scale
        cwt, freqs = pywt.cwt(arr[start:stop], scales=[1024], wavelet="morl", axis=0)
        lf = cwt[0][-1]
        leading_freqs.append(lf)

    leading_freqs = np.array(leading_freqs)

    first_nonzero = (np.around(leading_freqs, decimals=4) != 0).argmax()
    print('First non-zero batched: ', first_nonzero)
    ticks = np.arange(0, arr.size, max_scale)

    ax.plot(np.arange(arr.size), leading_freqs / leading_freqs.max(), label='CWT Scale 1024 (Normalized)')
    ax.plot(np.arange(arr.size), arr, label='Input Signal')
    ax.vlines(ticks, -0.1, 0.1, color='k', zorder=5)

    ax.set_title('Wavelet Transform on batched Dataset')
    ax.annotate('First nonzero (rounded to 4th decimal)',
                (first_nonzero, 0),
                xytext=(first_nonzero, 0.5),
                ha='right',
                va='bottom',
                arrowprops=dict(facecolor='black', shrink=0.05))

    ax.legend()

    return ax


if __name__ == '__main__':
    max_scale = 1024
    mult = 6

    fig, (ax1, ax2) = plt.subplots(2, 1)
    plot_unbatched(ax1)
    plot_batched(ax2)

    fig.suptitle('Demonstration of CWT Tachyons on Entire Dataset Without Batching')

    plt.show()
