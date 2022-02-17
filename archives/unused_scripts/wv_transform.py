import numpy as np
import pywt
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", fontscale=1.15)

if __name__ == "__main__":
    signal = np.load("single_elm_event.npy")
    # signal = signal / np.max(signal)
    print(signal.shape)

    channels = [0, 31, 63]
    wavelet_fam = {
        "db": ["db2", "db4", "db10", "db20"],
        "sym": ["sym2", "sym9", "sym14", "sym18"],
        "bior": ["bior2.8", "bior3.1", "bior3.9", "bior5.5"],
        "coif": ["coif1", "coif4", "coif10", "coif17"],
    }
    fig, axs = plt.subplots(4, 3, figsize=(15, 21), constrained_layout=True)
    axs = axs.flatten()

    offsets = [0, 1, 2, 3, 4]
    xtick_labels = list(range(1900, 2400, 50))
    idx = 0
    for wv_fam in list(wavelet_fam.keys()):
        reconst_signals = {}
        print(f"Using wavelet family: {wv_fam}")
        for wavelet in wavelet_fam[wv_fam]:
            print(f"Using wavelet: {wavelet}")
            coeff = pywt.wavedec(
                signal, wavelet=wavelet, mode="symmetric", axis=0
            )
            uthresh = 1
            coeff[1:] = (
                pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:]
            )
            reconst_signal = pywt.waverec(
                coeff, wavelet=wavelet, mode="symmetric", axis=0
            )
            reconst_signals[wavelet] = reconst_signal
        for ch in channels:
            ax = axs[idx]
            # plt.setp(ax.get_yticklabels(), fontsize=7)
            ax.tick_params(axis="both", which="major", labelsize=7)
            for (k, v), off in zip(reconst_signals.items(), offsets):
                w = pywt.Wavelet(k)
                ax.plot(
                    v[1900:2300, ch] + off,
                    label=f"{k}, length: {w.dec_len}",
                    lw=1.5,
                )
            ax.plot(
                signal[1900:2300, ch] + offsets[-1],
                label="original",
                lw=1.0,
                c="slategray",
            )
            ax.set_title(
                f"Channel: {ch + 1}",
                fontsize=11,
            )
            ax.legend(fontsize=8, frameon=False)
            ax.grid(axis="y")
            ticks_loc = ax.get_xticks().tolist()
            ticks_loc = ticks_loc[1:]
            # taken from: https://stackoverflow.com/questions/63723514/userwarning-fixedformatter-should-only-be-used-together-with-fixedlocator
            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            ax.set_xticklabels(xtick_labels)
            idx += 1
    # plt.tight_layout()
    plt.show()
