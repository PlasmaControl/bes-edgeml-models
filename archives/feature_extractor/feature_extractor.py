"""Script to extract features from the convolutional neural network. These
features can be used as input features for classical machine learning models
like logistic regression, random forest etc. For instance, we took the features
out of the 3D convolution layer from the `FeatureModel` in the following script.

See here for more details:
https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks
"""
print(__doc__)
import os

import numpy as np
import pandas as pd
import torch

from src import utils, dataset
from options.test_arguments import TestArguments

activation = dict()


def get_activation(name):
    """
    Get the activation out of the given layer from a neural network.

    Args:
        name (str): Name of the layer.

    Returns:
        Callable.
    """

    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def elm_feature_df(elm_indices: np.ndarray, all_data: tuple):
    """
    Create a pandas dataframe where the feature map from the forward hook can be
    stored.

    Args:
        elm_indices (np.ndarray): NumPy array containing the ELM IDs.
        all_data (tuple): Tuple containing the signals, labels, valid_indices
        and window start indices.

    Returns:
        pd.DataFrame: Dataframe containing the feature map for each ELM event.
    """
    signals, labels, valid_indices, window_start = all_data

    print(f"Signals shape: {signals.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Valid indices shape: {valid_indices.shape}")
    print(f"Window start shape: {window_start.shape}")

    df = pd.DataFrame()
    elm_event_id = [1]
    count = 1
    for i in range(1, len(valid_indices)):
        i_prev = i - 1
        if valid_indices[i] - valid_indices[i_prev] == 1:
            elm_event_id.append(count)
        else:
            count += 1
            elm_event_id.append(count)
    df["elm_id"] = elm_event_id
    df["valid_indices"] = valid_indices
    elm_ids = dict(enumerate(elm_indices, start=1))
    df["elm_event"] = df["elm_id"].map(elm_ids).apply(lambda x: f"{x:05d}")
    return df


if __name__ == "__main__":
    # instantiate the elm_model and load the checkpoint
    model_ckpt_dir = "model_checkpoints/signal_window_16"
    args, parser = TestArguments().parse(verbose=True)

    # load the output paths
    paths = utils.create_output_paths(args, infer_mode=True)
    roc_dir = paths[-1]

    # create the logger object and instantiate the model
    LOGGER = utils.get_logger(script_name=__name__)
    model_cls = utils.create_model_class(args.model_name)
    model = model_cls(args)
    device = torch.device("cpu")
    model = model.to(device)
    print(model)

    # load the saved model
    model_ckpt_path = os.path.join(
        model_ckpt_dir,
        f"{args.model_name}_lookahead_{args.label_look_ahead}_{args.data_preproc}{args.filename_suffix}.pth",
    )
    print(f"Using elm_model checkpoint: {model_ckpt_path}")
    model.load_state_dict(
        torch.load(
            model_ckpt_path,
            map_location=device,
        )["model"]
    )
    # register the model layer with the forward hooks
    model.conv.register_forward_hook(get_activation("conv"))

    # create data object from data preprocessing pipeline
    data_cls = utils.create_data_class(args.data_preproc)
    data_obj = data_cls(args, LOGGER)

    # load all the ELM events (need to use `--use_all_data` command line arg in
    # order to do so)
    all_elms, all_data = data_obj.get_data()
    signals, labels, valid_indices, window_start = all_data
    df = elm_feature_df(all_elms, all_data)

    # create PyTorch dataset from all data
    data = dataset.ELMDataset(
        args, signals, labels, valid_indices, window_start, LOGGER
    )
    cnn_feature_map = []
    look_ahead_labels = []

    # iterate through all the rows of the dataframe
    for i in range(len(df)):
        i_prev = 0 if i == 0 else i - 1
        if df.loc[i, "elm_id"] != df.loc[i_prev, "elm_id"]:
            print(
                f'Processing input tensor from elm event {df.loc[i, "elm_id"]}'
            )
        input_signals, input_labels = data.__getitem__(i)
        input_signals = input_signals.unsqueeze(0)
        model_out = model(input_signals)
        feature_map = activation["conv"]
        feature_map = torch.flatten(feature_map, start_dim=0)
        cnn_feature_map.append(feature_map.numpy().tolist())
        look_ahead_labels.append(input_labels.numpy().tolist())
    df[f"{args.model_name}_feature_map"] = cnn_feature_map
    df["label"] = look_ahead_labels

    # clean the dataframe to have one column per feature
    expanded_feature_df = pd.DataFrame(
        df["cnn_feature_map"].to_list(),
        columns=[f"f_{i+1}" for i in range(len(cnn_feature_map[0]))],
        dtype=np.float32,
    )
    print(expanded_feature_df.info(memory_usage="deep"))
    df.drop([f"{args.model_name}_feature_map"], axis=1, inplace=True)
    df = pd.concat([df, expanded_feature_df], axis=1)
    print(df)

    # save the dataframe as a CSV file (not very memory efficient, could use HDF5.
    # Pandas support HDF5 format).
    df.to_csv(
        os.path.join(
            roc_dir, f"{args.model_name}_feature_df_{args.label_look_ahead}.csv"
        ),
        index=False,
    )
