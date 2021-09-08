import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns

# plt.style.use("/home/lakshya/plt_custom.mplstyle")
sns.set_style("darkgrid")
sns.set_palette("deep")

if __name__ == "__main__":
    signal = np.load("single_elm_event.npy")
    signal = signal / np.max(signal)

    channels = [0, 31, 63]
    wavelets = ["db2", "db8", "sym4", "coif2"]
    lengths = [4, 16, 8, 12]
    fig, axs = plt.subplots(4, 3, figsize=(15, 18), constrained_layout=True)
    axs = axs.flatten()

    idx = 0
    for wavelet, length in zip(wavelets, lengths):
        coeff = pywt.wavedec(signal, wavelet=wavelet, mode="symmetric")
        uthresh = 1
        coeff[1:] = (
            pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:]
        )
        reconst_signal = pywt.waverec(coeff, wavelet=wavelet, mode="symmetric")
        for ch in channels:
            ax = axs[idx]
            ax.plot(signal[:, ch], label="original", c="#636EFA", lw=1.25)
            ax.plot(
                reconst_signal[:, ch],
                label="wavelet transformed",
                c="#EF553B",
                lw=0.75,
            )
            ax.set_title(
                f"Channel: {ch + 1}, wavelet: {wavelet}, length: {length}",
                fontsize=11,
            )
            ax.legend(fontsize=8, frameon=False)
            idx += 1
    # plt.tight_layout()
    plt.show()
