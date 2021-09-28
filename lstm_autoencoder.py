import time
import argparse
import logging
from typing import Tuple, Union
import warnings

warnings.filterwarnings(action="ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn import metrics

from options.train_arguments import TrainArguments
from src import utils, dataset, run

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.25)
# COLORS_PALETTE = [
#     "#01BEFE",
#     "#FFDD00",
#     "#FF7D00",
#     "#FF006D",
#     "#93D30C",
#     "#8F00FF",
# ]
# sns.set_palette(sns.color_palette(COLORS_PALETTE))
LABELS = ["no ELM", "ELM"]


def get_all_data(
    args: argparse.Namespace, logger: logging.Logger
) -> Tuple[tuple, tuple, tuple]:
    data_cls = utils.create_data(args.data_preproc)
    data_obj = data_cls(args, logger)
    train_data, valid_data, test_data = data_obj.get_data()
    return train_data, valid_data, test_data


def print_data_info(args: argparse.Namespace, data: tuple, verbose=0) -> None:
    signals = data[0]
    labels = data[1]
    allowed_indices = data[2]
    window_start = data[3]

    print(f"Signals shape: {signals.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Allowed indices: {allowed_indices}")
    print(allowed_indices.shape)
    print(f"Window start: {window_start}")

    if verbose > 0:
        num_elms = len(window_start)
        for i_elm in range(num_elms):
            i_start = window_start[i_elm]
            if i_elm < num_elms - 1:
                i_stop = window_start[i_elm + 1] - args.signal_window_size
            else:
                i_stop = labels.size - args.signal_window_size
            print(i_start, i_stop)
            print(signals[i_start:i_stop].shape)
            print(
                labels[
                    i_start
                    + args.signal_window_size
                    - 1 : i_stop
                    + args.signal_window_size
                    - 1
                ].shape
            )


# def scale(X, scaler):
#     for i in range(X.shape[0]):
#         X[i, :, :] = scaler.transform(X[i, :, :])
#     return X


def print_arrays_shape(X: np.ndarray, y: np.ndarray, mode: str) -> None:
    print(f"X_{mode} shape: {X.shape}")
    print(f"y_{mode} shape: {y.shape}")


def temporalize(
    args: argparse.Namespace,
    signals: np.ndarray,
    labels: np.ndarray,
    allowed_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for i_allowed in range(len(allowed_indices)):
        elm_idx = allowed_indices[i_allowed]
        signal_window = signals[elm_idx : elm_idx + args.signal_window_size]
        label = labels[
            elm_idx + args.signal_window_size + args.label_look_ahead - 1
        ]
        X.append(signal_window)
        y.append(label)
    X = np.array(X).reshape(-1, args.signal_window_size, 64)
    X = X.astype(np.float32)
    y = np.array(y).astype(np.uint8)
    return X, y


def make_tensors(
    X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)

    return X, y


def create_tensor_dataset(
    X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]
) -> torch.utils.data.Dataset:
    X, y = make_tensors(X, y)
    dataset = torch.utils.data.TensorDataset(X, y)
    return dataset


class FCAutoencoder(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        input_features: int = 1024,
        num_nodes: list = [128, 32],
        dropout: float = 0.3,
    ) -> None:
        super(FCAutoencoder, self).__init__()
        self.args = args
        self.num_nodes = num_nodes
        self.fc1 = nn.Linear(
            in_features=input_features, out_features=num_nodes[0]
        )
        self.fc2 = nn.Linear(
            in_features=num_nodes[0], out_features=num_nodes[1]
        )
        self.fc3 = nn.Linear(
            in_features=num_nodes[1], out_features=num_nodes[0]
        )
        self.fc4 = nn.Linear(
            in_features=num_nodes[0], out_features=input_features
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.relu(self.dropout(self.fc3(x)))
        x = self.fc4(x)

        return x.view(-1, self.args.signal_window_size, 64)


class Encoder(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        seq_len: int,
        n_features: int,
        n_layers: int,
        dropout: float,
    ):
        super(Encoder, self).__init__()
        self.args = args
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = self.args.hidden_size
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
        args: argparse.Namespace,
        seq_len: int,
        n_features: int,
        n_layers: int,
        dropout: float,
    ):
        super(Decoder, self).__init__()
        self.args = args
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = self.args.hidden_size
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
        super(LSTMAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions for both encoder and decoder must be equal"

        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder should have same number of layers"

    def forward(self, input):
        # input = torch.unsqueeze(input, 0)
        # encode
        hidden = self.encoder(input)
        # decode
        y = self.decoder(hidden)

        return y.squeeze(0)


def train_model(
    args: argparse.Namespace,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
):
    seq_len = 16
    n_features = 64
    n_layers = 2
    pct = 0.3
    encoder = Encoder(
        args,
        seq_len=seq_len,
        n_features=n_features,
        n_layers=n_layers,
        dropout=pct,
    )
    decoder = Decoder(
        args,
        seq_len=seq_len,
        n_features=n_features,
        n_layers=n_layers,
        dropout=pct,
    )
    # model = LSTMAutoencoder(encoder, decoder)
    model = FCAutoencoder(args, input_features=1024)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss(reduction="sum")
    history = dict(train=[], valid=[])

    for epoch in range(args.n_epochs):
        model = model.train()
        ts = time.time()
        train_losses = []

        for data in train_dataloader:
            seq_in = data[0]
            seq_in = seq_in.to(args.device)

            optimizer.zero_grad()

            seq_out = model(seq_in)

            loss = criterion(seq_out, seq_in)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        valid_losses = []
        model.eval()
        with torch.no_grad():
            for data in valid_dataloader:
                seq_in = data[0]
                seq_in = seq_in.to(args.device)

                seq_out = model(seq_in)

                loss = criterion(seq_out, seq_in)

                valid_losses.append(loss.item())
        te = time.time()
        train_epoch_loss = np.mean(train_losses)
        valid_epoch_loss = np.mean(valid_losses)

        history["train"].append(train_epoch_loss)
        history["valid"].append(valid_epoch_loss)

        print(f"Epoch: {epoch+1}, time taken: {(te-ts):.3f}")
        print(
            f"\ttrain loss: {train_epoch_loss:.5f}, valid loss: {valid_epoch_loss:.5f}"
        )

    return model, history


def plot_loss(args: argparse.Namespace, history: dict):
    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(history["train"], label="train", lw=2.5)
    plt.plot(history["valid"], label="valid", lw=2.5)
    plt.ylabel("Loss")
    plt.xlabel("epoch")
    plt.title("Loss over training epochs")
    plt.legend(frameon=False)
    if not args.dry_run:
        plt.savefig(
            "outputs/ts_anomaly_detection_plots/train_valid_loss_fc_ae.png",
            dpi=200,
        )
    plt.show()


def precision_recall_curve(
    args, error_df
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # plot precision-recall curve
    precision, recall, threshold = metrics.precision_recall_curve(
        error_df.ground_truth.values, error_df.reconstruction_error.values
    )
    print(
        f"Precision, recall, thresh shape: {precision.shape}, {recall.shape}, {threshold.shape}"
    )
    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(threshold, precision[1:], label="Precision", lw=2.5)
    plt.plot(threshold, recall[1:], label="Recall", lw=2.5)
    plt.title("Precision and recall for different thresholds")
    plt.xlabel("Threshold")
    plt.ylabel("Precision/Recall")
    plt.legend(frameon=False)
    plt.tight_layout()
    if not args.dry_run:
        plt.savefig(
            "outputs/ts_anomaly_detection_plots/fc_ae_precision_recall_curve.png",
            dpi=200,
        )
    plt.show()
    return precision, recall, threshold


def plot_recons_loss_dist(args, error_df, plot_log=False):
    if plot_log:
        # plot the distribution of reconstruction losses in log scale
        plt.figure(figsize=(8, 6), dpi=200)
        sns.distplot(
            error_df["log_reconstruction_error"],
            bins=50,
            kde=True,
            label=r"$\log$(1+MAE)",
        )
        plt.title(r"Error distribution ($\log$ scale)")
        plt.legend(frameon=False)
        plt.tight_layout()
        if not args.dry_run:
            plt.savefig(
                "outputs/ts_anomaly_detection_plots/fc_ae_error_distribution_log.png",
                dpi=200,
            )
        plt.show()
    else:
        # plot the distribution of reconstruction losses
        plt.figure(figsize=(8, 6), dpi=200)
        sns.distplot(
            error_df["reconstruction_error"],
            bins=50,
            kde=True,
            label="MAE",
        )
        plt.title("Error distribution")
        plt.legend(frameon=False)
        plt.tight_layout()
        if not args.dry_run:
            plt.savefig(
                "outputs/ts_anomaly_detection_plots/fc_ae_error_distribution.png",
                dpi=200,
            )
        plt.show()


def plot_recons_loss_with_signals(
    args,
    error_df: pd.DataFrame,
    X_valid,
    precision=None,
    recall=None,
    threshold=None,
    plot_thresh=False,
) -> Union[None, float]:
    # fig = plt.figure(figsize=(14, 12), dpi=200)
    # for i, id in enumerate(error_df["id"].unique().tolist()):
    #     print(f"ID: {id}")
    #     df = error_df[error_df["id"] == id]
    #     ax = plt.subplot(4, 3, i + 1)
    #     groups = df.groupby("ground_truth")
    #     # df = df.reset_index(drop=True)
    #     indices = df.index.tolist()
    #     start_idx = indices[0]
    #     end_idx = indices[-1]
    #     print(f"Start index: {start_idx}, end index: {end_idx}")
    #     for (name, group), alpha in zip(groups, [1, 0.5]):
    #         ax.plot(
    #             # group.index,
    #             group.reconstruction_error_scaled,
    #             marker="o",
    #             ms=3,
    #             linestyle="",
    #             label=LABELS[1] if name == 1 else LABELS[0],
    #             alpha=alpha,
    #         )
    #     ticklabels = [item.get_text() for item in ax.get_xticklabels()]
    #     labels = list(range(start_idx, end_idx + 1))
    #     ticklabels = [labels[i] for i in range(len(labels))]
    #     ax.set_xticklabels(ticklabels)
    #     ax.plot(
    #         X_valid[start_idx:end_idx, 0, 21],
    #         zorder=-1,
    #         label="Ch:22",
    #         c="slategrey",
    #     )
    #     # plt.plot(df.ground_truth, label="ground truth")
    #     plt.ylabel("Reconstruction Loss", fontsize=6)
    #     plt.xlabel("Data point index", fontsize=6)
    #     plt.tick_params(axis="x", labelsize=4)
    #     plt.tick_params(axis="y", labelsize=4)
    #     plt.legend(fontsize=5, frameon=False)
    #     plt.grid(axis="x")
    # plt.suptitle("Reconstruction error for different classes")
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # if not args.dry_run:
    #     plt.savefig(
    #         "outputs/ts_anomaly_detection_plots/lstm_ae_recon_error_with_signals.png",
    #         dpi=200,
    #     )
    # plt.show()
    # plot reconstruction loss with signals
    if plot_thresh:
        if precision is None or recall is None or threshold is None:
            raise TypeError(
                "Precision, recall and threshold values are not provided!"
            )
        else:
            # plot reconstruction loss without signals and with a threshold
            precision_recall_eq = np.where(precision == recall)[0][0]
            threshold_val = threshold[precision_recall_eq - 1]
            print(f"Using threshold value: {threshold_val}")
            groups = error_df.groupby("ground_truth")
            fig = plt.figure(figsize=(12, 6), dpi=200)
            ax = fig.add_subplot()
            for (name, group), alpha in zip(groups, [1, 0.5]):
                ax.plot(
                    group.index,
                    group.reconstruction_error,
                    marker="o",
                    ms=3,
                    linestyle="",
                    label=LABELS[1] if name == 1 else LABELS[0],
                    alpha=alpha,
                )
            ax.axhline(
                y=threshold_val,
                zorder=10,
                ls="--",
                lw=1.0,
                c="crimson",
                label="Threshold",
            )
            ax.set_ylabel("Reconstruction Loss")
            ax.set_xlabel("Data point index")
            ax.set_title("Reconstruction error for different classes")
            ax.legend(frameon=False)
            plt.tight_layout()
            if not args.dry_run:
                plt.savefig(
                    "outputs/ts_anomaly_detection_plots/fc_ae_recon_error_with_threshold.png",
                    dpi=200,
                )
            plt.show()
            return threshold_val
    else:
        plt.figure(figsize=(12, 6), dpi=200, constrained_layout=True)
        groups = error_df.groupby("ground_truth")
        for (name, group), alpha in zip(groups, [1, 0.5]):
            plt.plot(
                # group.index,
                group.reconstruction_error_scaled,
                marker="o",
                ms=3,
                linestyle="",
                label=LABELS[1] if name == 1 else LABELS[0],
                alpha=alpha,
            )
        plt.plot(
            X_valid[:, 0, 21],
            zorder=-1,
            label="Ch:22",
            c="slategrey",
        )
        plt.plot(error_df.ground_truth, label="ground truth")
        plt.ylabel("Reconstruction Loss")
        plt.xlabel("Data point index")
        plt.legend(frameon=False)
        plt.title("Reconstruction error for different classes")
        plt.tight_layout()  # rect=[0, 0.03, 1, 0.95])
        if not args.dry_run:
            plt.savefig(
                "outputs/ts_anomaly_detection_plots/fc_ae_recon_error_with_signals.png",
                dpi=200,
            )
        plt.show()


def plot_metrics(
    args: argparse.Namespace, error_df: pd.DataFrame, X_valid: np.ndarray
):
    precision, recall, threshold = precision_recall_curve(args, error_df)
    plot_recons_loss_dist(args, error_df, plot_log=False)
    plot_recons_loss_dist(args, error_df, plot_log=True)
    plot_recons_loss_with_signals(
        args, error_df, X_valid, precision, recall, threshold, plot_thresh=False
    )
    threshold_val = plot_recons_loss_with_signals(
        args, error_df, X_valid, precision, recall, threshold, plot_thresh=True
    )

    # confusion matrix
    y_pred = np.array(
        [
            1 if error > threshold_val else 0
            for error in error_df.reconstruction_error.values
        ]
    )

    conf_matrix = metrics.confusion_matrix(error_df.ground_truth.values, y_pred)
    plt.figure(figsize=(8, 6), dpi=100)
    sns.heatmap(
        conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d"
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    if not args.dry_run:
        plt.savefig(
            "outputs/ts_anomaly_detection_plots/fc_ae_confusion_matrix.png",
            dpi=200,
        )
    plt.show()


def main():
    # initialize the argparse and the logger
    args, parser = TrainArguments().parse(verbose=True)
    utils.test_args_compat(args, parser)
    logger = utils.get_logger(script_name=__name__)

    # get train and valid data
    train_data, valid_data, _ = get_all_data(args, logger)

    # reshape the train signals and print info
    (
        train_signals,
        train_labels,
        train_allowed_indices,
        _,
    ) = train_data
    train_signals = train_signals.reshape(-1, 64)
    print_data_info(args, train_data)

    # reshape the valid signals and print info
    (
        valid_signals,
        valid_labels,
        valid_allowed_indices,
        valid_window_start,
    ) = valid_data
    valid_signals = valid_signals.reshape(-1, 64)
    print_data_info(args, valid_data)
    del valid_data, train_data

    # normalize the data
    train_signals_032_scaled = train_signals[:, :32] / np.max(
        train_signals[:, :32]
    )
    train_signals_3264_scaled = train_signals[:, 32:] / np.max(
        train_signals[:, 32:]
    )
    train_signals_scaled = np.concatenate(
        [train_signals_032_scaled, train_signals_3264_scaled], axis=1
    )
    del train_signals, train_signals_032_scaled, train_signals_3264_scaled

    valid_signals_032_scaled = valid_signals[:, :32] / np.max(
        valid_signals[:, :32]
    )
    valid_signals_3264_scaled = valid_signals[:, 32:] / np.max(
        valid_signals[:, 32:]
    )
    valid_signals_scaled = np.concatenate(
        [valid_signals_032_scaled, valid_signals_3264_scaled], axis=1
    )
    del valid_signals, valid_signals_032_scaled, valid_signals_3264_scaled
    # (
    #     test_signals,
    #     test_labels,
    #     test_allowed_indices,
    #     test_window_start,
    # ) = test_data

    # create train signals and labels suited for an RNN
    X_train, y_train = temporalize(
        args, train_signals_scaled, train_labels, train_allowed_indices
    )
    print_arrays_shape(X_train, y_train, mode="train")

    # create valid signals and labels suited for an RNN
    X_valid, y_valid = temporalize(
        args, valid_signals_scaled, valid_labels, valid_allowed_indices
    )
    print_arrays_shape(X_valid, y_valid, mode="valid")

    # autoencoders will only be trained on the negative classes
    X_train_y0 = X_train[y_train == 0]
    X_train_y1 = X_train[y_train == 1]
    del X_train
    y_train_y0_idx = np.where(y_train == 0)[0]
    y_train_y1_idx = np.where(y_train == 1)[0]
    y_train_y0 = y_train[y_train_y0_idx]
    y_train_y1 = y_train[y_train_y1_idx]
    del y_train

    print_arrays_shape(X_train_y0, y_train_y0, mode="train_y0")
    print_arrays_shape(X_train_y1, y_train_y1, mode="train_y1")

    X_valid_y0 = X_valid[y_valid == 0]
    X_valid_y1 = X_valid[y_valid == 1]
    y_valid_y0_idx = np.where(y_valid == 0)[0]
    y_valid_y1_idx = np.where(y_valid == 1)[0]
    y_valid_y0 = y_valid[y_valid_y0_idx]
    y_valid_y1 = y_valid[y_valid_y1_idx]

    # with open("X_valid.npy", "wb") as f:
    #     np.save(f, X_valid)
    # with open("y_valid.npy", "wb") as f:
    #     np.save(f, y_valid)

    print_arrays_shape(X_valid_y0, y_valid_y0, mode="valid_y0")
    print_arrays_shape(X_valid_y1, y_valid_y1, mode="valid_y1")

    train_dataset = create_tensor_dataset(X_train_y0, y_train_y0)
    valid_dataset = create_tensor_dataset(X_valid_y0, y_valid_y0)
    test_dataset = create_tensor_dataset(X_valid, y_valid)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        # shuffle=True,
    )
    # encoder = Encoder(args, seq_len=16, n_features=64, n_layers=2, dropout=0.3)
    # decoder = Decoder(args, seq_len=16, n_features=64, n_layers=2, dropout=0.3)
    # x = torch.rand(1, 16, 64)
    # lstm_ae = LSTMAutoencoder(encoder, decoder)
    # y = lstm_ae(x)
    # print(y.shape)
    # # for seq in train_dataset:
    # #     print(seq[0].shape)
    model, history = train_model(args, train_loader, valid_loader)

    # save the model
    # model_path = "lstm_ae.pth"
    model_path = "fc_ae.pth"
    torch.save(model, model_path)
    plot_loss(args, history)

    # create ELM event unique identifier
    identifier = list(range(1, len(valid_window_start) + 1))
    num_repeats = list(valid_window_start)
    num_repeats.append(len(X_valid))
    diffs = np.diff(num_repeats)
    broadcast_identifier = np.repeat(identifier, repeats=diffs)
    print(broadcast_identifier)
    print(broadcast_identifier.shape, X_valid.shape)

    # # classification
    with torch.no_grad():
        mae = []
        for data in test_loader:
            seq = data[0]
            seq = seq.to(args.device)
            pred_seq = model(seq)
            loss = torch.sum(torch.abs(torch.squeeze(seq, 0) - pred_seq))
            mae.append(loss.cpu().numpy())
        mae = np.array(mae)
    error_df = pd.DataFrame(
        {
            "reconstruction_error": mae,
            "reconstruction_error_scaled": mae / np.max(mae),
            "log_reconstruction_error": np.log1p(mae),
            "ground_truth": y_valid.tolist(),
            "id": broadcast_identifier.tolist(),
        }
    )
    print(error_df)
    plot_metrics(args, error_df, X_valid)


if __name__ == "__main__":
    main()
