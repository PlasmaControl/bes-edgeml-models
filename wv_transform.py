import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("Dark2")
sns.set_style("darkgrid")


if __name__ == "__main__":
    wavelet = 'db4'
    ch = 63
    signal = np.load("single_elm_event.npy")
    signal = signal / np.max(signal)
    coeff = pywt.wavedec(signal, wavelet=wavelet, mode="symmetric")
    uthresh = 1
    coeff[1:] = (
        pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:]
    )
    reconst_signal = pywt.waverec(coeff, wavelet=wavelet, mode="symmetric")
    plt.plot(signal[:, ch], label="original signal")
    plt.plot(reconst_signal[:, ch], label="reconstructed signal")
    plt.title(f'Wavelet transformed signal, wavelet: {wavelet}, threshold: {uthresh}, channel: {ch+1}')
    plt.legend()
    plt.tight_layout()
    plt.show()
