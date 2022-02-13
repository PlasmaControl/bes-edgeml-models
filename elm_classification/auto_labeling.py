"""
Helper script to perform the automatic labeling using the trained LSTM
autoencoder model available at `model_checkpoints/signal_window_16/lstm_ae_sws_16_la_0.pth`.
Make sure to use all the data from the HDF5 file using the command line argument
`--use_all_data`.

Kept this script and `lstm_autoencoder.py` a bit standalone so there might be some
hardcoded parameters.
"""
print(__doc__)
import os
import h5py
from typing import Union, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.7)


def normalize_data(signal: np.ndarray) -> np.ndarray:
    """
    Normalize the data before making the inference.

    Args:
        signal (np.ndarray): NumPy array containing the BES signal.

    Returns:
        Normalized signal.
    """
    assert signal.shape[1] == 64, "Signal must be reshaped into (-1, 64)"
    signal[:, :32] = signal[:, :32] / np.max(signal[:, :32])
    signal[:, 32:] = signal[:, 32:] / np.max(signal[:, 32:])
    return signal


def temporalize(
    sws: int, signals: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Traverse the input ELM event based on the size of the signal window. A rolling
    signal window of size `sws` time steps will be created at a time. The
    corresponding label for that signal window will be the label for the leading
    time step.

    For instance, a BES signal of size (1000, 64) will be temporalized to a size
    (985, 16, 64) for a signal window of 16.

    Args:
        sws (int): Size of the signal window.
        signals (np.ndarray): BES signals.
        labels (np.ndarray): Manual labels.

    Returns:
        Tuple containing temporalized signals and corresponding manual labels.
    """
    X_unlabeled = []
    y = []
    for i in range(len(signals) - sws + 1):
        X_unlabeled.append(signals[i : i + sws])
        y.append(labels[i + sws - 1])
    X_unlabeled = np.array(X_unlabeled)
    y = np.array(y)
    return X_unlabeled, y


def make_tensors(
    X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create PyTorch tensors from NumPy arrays."""
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)

    return X, y


def create_tensor_dataset(
    X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]
) -> torch.utils.data.Dataset:
    """Create PyTorch dataset from the input tensors. All the PyTorch dataset
    specific methods like `__len__` and `__getitem__` can be used."""
    X, y = make_tensors(X, y)
    dataset = torch.utils.data.TensorDataset(X, y)
    return dataset


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        seq_len: int,
        n_features: int,
        n_layers: int,
        dropout: float,
    ):
        """
        Encoder part of the autoencoder using LSTM layers to create an
        intermediate representation. The architecture is loosely based on paper:
        https://arxiv.org/abs/1502.04681.

        It takes an argument `seq_len` which is the length of the input sequence.
        It is basically the size of the rolling signal window of the BES signal.
        `n_features` argument takes in the size of the input to the LSTM layer,
        this is the 64 channels of the BES signal.

        All the dimensional gymnastics are put as line comments at each step in
        the forward pass.

        Args:
        -----
            hidden_dim (int): Size of the hidden dimension.
            seq_len (int): Length of the input sequence i.e. `--signal_window_size`.
            n_features (int): Input size to the LSTM layer i.e. 64 BES channels.
            n_layers (int): Number of layers in the LSTM.
            dropout (float): Dropout for the LSTM layer, intrinsically ignored if `n_layers = 1`.
        """
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            batch_first=True,
            num_layers=self.n_layers,
            dropout=self.dropout,
        )

    def forward(self, x):
        _, (hidden, _) = self.rnn(x)
        # hidden size: (num_layers, batch_size, hidden_size)
        # hidden = hidden.reshape(
        #     batch_size, -1
        # )  # (batch_size, num_layers*hidden_size)
        hidden = (
            hidden.transpose(0, 1)
            .contiguous()
            .view(-1, self.n_layers * self.hidden_dim)
        )
        return hidden


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        seq_len: int,
        n_features: int,
        n_layers: int,
        dropout: float,
    ):
        """
        Decoder part of the autoencoder using LSTM layers to create an
        intermediate representation. The architecture is loosely based on paper:
        https://arxiv.org/abs/1502.04681.

        **The arguments and their meaning is mostly similar to that of the encoder.**

        It takes an argument `seq_len` which is the length of the input sequence.
        It is basically the size of the rolling signal window of the BES signal.
        `n_features` argument takes in the size of the input to the LSTM layer,
        this is the 64 channels of the BES signal.

        All the dimensional gymnastics are put as line comments at each step in
        the forward pass.

        Args:
        -----
            hidden_dim (int): Size of the hidden dimension.
            seq_len (int): Length of the input sequence i.e. `--signal_window_size`.
            n_features (int): Input size to the LSTM layer i.e. 64 BES channels.
            n_layers (int): Number of layers in the LSTM.
            dropout (float): Dropout for the LSTM layer, ignored if `n_layers = 1`.
        """
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn = nn.LSTM(
            input_size=self.n_layers * self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            num_layers=self.n_layers,
            dropout=self.dropout,
        )
        self.fc = nn.Linear(self.hidden_dim, self.n_features)

    def forward(self, x):
        # x: (batch_size, num_layers*hidden_size)
        x = x.repeat(1, self.seq_len, 1)
        x = x.reshape(-1, self.seq_len, self.n_layers * self.hidden_dim)
        x, _ = self.rnn(x)  # x: (batch_size, seq_len, hidden_dim)
        x = self.fc(x)  # x: (batch_size, seq_len, n_features)

        return x


class LSTMAutoencoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        """Autoencoder class that encapsulates both encoder and decoder of the
        LSTM autoencoder.

        Args:
        -----
            encoder (Encoder): Encoder object.
            decoder (Decoder): Decoder object.
        """
        super(LSTMAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions for both encoder and decoder must be equal"

        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder should have same number of layers"

    def forward(self, x):
        # encode
        hidden = self.encoder(x)
        # decode
        y = self.decoder(hidden)

        return y.squeeze(0)


def load_model(
    base_path: str, saved_model_path: str, model: LSTMAutoencoder
) -> None:
    """
    Load model from a checkpoint.
    Args:
        base_path (str): Path to the top level directory.
        saved_model_path (str): Path to the model checkpoint from top level dir.
        model (LSTMAutoencoder): Instance of LSTMAutoencoder class.

    Returns:
        None
    """
    model_ckpt_path = os.path.join(base_path, saved_model_path)
    print(f"Loading model from:\n{model_ckpt_path}")
    device = torch.device("cpu")
    model = model.to(device)
    model.load_state_dict(
        torch.load(
            model_ckpt_path,
            map_location=device,
        )
    )


def create_model() -> LSTMAutoencoder:
    """
    Instantiate the LSTM Autoencoder model. Hardcoding the values because these
    are the values the model is trained with.

    Returns:
        LSTMAutoencoder model instance.
    """
    seq_len = 16  # sws
    n_features = 64  # num BES channels
    n_layers = 2  # num stacked LSTM layers
    pct = 0.3  # dropout fraction
    hidden_dim = 32  # hidden dimension
    encoder = Encoder(
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        n_features=n_features,
        n_layers=n_layers,
        dropout=pct,
    )
    decoder = Decoder(
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        n_features=n_features,
        n_layers=n_layers,
        dropout=pct,
    )
    model = LSTMAutoencoder(encoder, decoder)
    return model


def main(
    data_dir: str,
    input_file_name: str,
    output_file_name: str,
    model: LSTMAutoencoder,
    threshold: float = 0.023,
    show_plots: bool = True,
    output_dir: str = "outputs/ts_anomaly_detection_plots/automatic_labels",
) -> None:
    """
    Actual function creating the automatic labels and plot for each ELM event.

    Args:
        data_dir (str): Path to the data directory.
        input_file_name (str): Name of the input HDF5 file.
        output_file_name (str): Name of the output HDF5 file.
        model (LSTMAutoencoder): Instance of the LSTM model.
        threshold (float): Threshold for the reconstruction error.
        show_plots (bool): If true, display plots.
        output_dir (str): Path to the directory where output plots are saved.

    Returns:
        None

    """
    hf = h5py.File(os.path.join(data_dir, input_file_name), "r")
    hf_out = h5py.File(os.path.join(data_dir, output_file_name), "w")
    elm_ids = list(hf.keys())
    num_pages = int(elm_ids // 12) + 1
    elm_ids_per_page = [
        elm_ids[i * 12 : (i + 1) * 12] for i in range(num_pages)
    ]
    print(f"Total ELM events: {len(elm_ids)}")
    # iterate through all the pages
    for page_num, page in enumerate(elm_ids_per_page):
        print(f"Drawing plots on page: {page_num+1}")
        if len(page) == 12:
            fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(12, 14), dpi=150)
        else:
            remaining_elms = len(elm_ids) - 12 * (num_pages - 1)
            if remaining_elms <= 4:
                rows = remaining_elms
                cols = 1
            else:
                rows = 4
                cols = int(np.ceil(remaining_elms / rows))
            fig, ax = plt.subplots(
                nrows=rows, ncols=cols, figsize=(10, 12), dpi=150
            )
        ax = ax.flatten()
        # iterate through the ELM events in each page
        for i, id in enumerate(page):
            mae = []
            print(
                f"Processing {i+1} elm event out of {len(page)} with id: {id}"
            )
            # create group in the output HDF5 file
            hf_out_group = hf_out.create_group(f"{id}")
            signal = np.array(hf[id]["signals"])
            label = np.array(hf[id]["labels"])
            # store the original signals and manual labels in the output HDF5 file
            hf_out_group.create_dataset(
                "signals", data=signal, dtype=np.float32
            )
            hf_out_group.create_dataset(
                "manual_labels", data=label, dtype=np.uint16
            )
            signal = signal.T
            signal = normalize_data(signal)
            signal, label = temporalize(sws=16, signals=signal, labels=label)
            signal, label = make_tensors(signal, label)
            signal_dataset = create_tensor_dataset(signal, label)
            data_loader = torch.utils.data.DataLoader(
                signal_dataset, batch_size=1, num_workers=0
            )
            for input_sequence, _ in data_loader:
                # calculate loss from the predicted sequence and store it
                pred_sequence = model(input_sequence)
                loss = torch.mean(
                    torch.abs(torch.squeeze(input_sequence, 0) - pred_sequence)
                )
                mae.append(loss.cpu().detach().numpy())
            # compute predictions from the reconstruction error using mean absolute error (MAE)
            predictions = (np.array(mae) > threshold).astype(int)
            # Clean up the predictions: suppress the predictions in pre-ELM and
            # post-ELM regions using the manual labels
            for j in range(len(label)):
                if (label[j] or predictions[j]) and (label[j] == 0):
                    predictions[j] = 0
            x = list(np.where(predictions == 1)[0])
            # label all the time steps between the first and the last time step of
            # active ELM as active; this will fail for ELM shots with more than
            # one active ELM per event
            if not x:
                print(f"Found no active elm with id: {id}.")
            else:
                predictions[x[0] : x[-1]] = 1
            # store predictions as automatic labels
            hf_out_group.create_dataset(
                "automatic_labels", data=predictions, dtype=np.uint16
            )
            print(signal.shape, label.shape, np.array(mae).shape)
            # plot
            ax[i].plot(signal[:, -1, 0], label="signal")
            ax[i].plot(label, label="manual label")
            ax[i].plot(predictions, label="auto label")
            ax[i].plot(mae, label="recon. error")
            ax[i].set_title(f"Elm ID: {id}", fontsize=9)
            ax[i].legend(fontsize=8, frameon=False)
        if page_num == 0:
            plt.suptitle(
                f"Automatic labeling, LSTM autoencoder",
                fontsize=18,
            )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=1.5, h_pad=1.5)
        if show_plots:
            plt.show()
        fname = os.path.join(
            output_dir, f"page_{page_num+1}_elm_{page[0]}_{page[-1]}.png"
        )
        plt.savefig(fname, dpi=150)
        plt.close()
    hf.close()
    hf_out.close()


if __name__ == "__main__":
    base_path = os.getcwd()
    model_ckpt = "model_checkpoints/signal_window_16/lstm_ae_sws_16_la_0.pth"
    model = create_model()
    load_model(base_path, model_ckpt, model)
    main(
        data_dir="data",
        input_file_name="labeled-elm-events.hdf5",
        output_file_name="elm-events-manual-automatic-labels.hdf5",
        model=model,
        threshold=0.023,
        show_plots=False,
        output_dir="outputs/ts_anomaly_detection_plots/automatic_labels",
    )
