import argparse
import os

import numpy as np
import pandas as pd
import torch

from src import utils, dataset
from options.test_arguments import TestArguments

activation = dict()


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def elm_feature_df(elm_indices: np.ndarray, all_data: tuple):
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
    utils.test_args_compat(args, parser, infer_mode=True)
    paths = utils.create_output_paths(args, infer_mode=True)
    roc_dir = paths[-1]

    LOGGER = utils.make_logger(script_name=__name__)
    model_cls = utils.create_model(args.model_name)
    model = model_cls(args)
    device = torch.device("cpu")
    model = model.to(device)
    print(model)

    model_ckpt_path = os.path.join(model_ckpt_dir,
            f"{args.model_name}_lookahead_{args.label_look_ahead}_truncate.pth", )
    print(f"Using elm_model checkpoint: {model_ckpt_path}")
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device, )["model"])
    model.pool.register_forward_hook(get_activation("pool"))

    data_cls = utils.create_data(args.data_preproc)
    data_obj = data_cls(args, LOGGER)

    all_elms, all_data = data_obj.get_data()
    signals, labels, valid_indices, window_start = all_data
    df = elm_feature_df(all_elms, all_data)

    data = dataset.ELMDataset(args, signals, labels, valid_indices, window_start, LOGGER)
    cnn_feature_map = []
    look_ahead_labels = []
    for i in range(len(df)):
        i_prev = 0 if i == 0 else i - 1
        if df.loc[i, "elm_id"] != df.loc[i_prev, "elm_id"]:
            print(f'Processing input tensor from elm event {df.loc[i, "elm_id"]}')
        input_signals, input_labels = data.__getitem__(i)
        input_signals = input_signals.unsqueeze(0)
        model_out = model(input_signals)
        feature_map = activation["pool"]
        feature_map = torch.flatten(feature_map, start_dim=0)
        cnn_feature_map.append(feature_map.numpy().tolist())
        look_ahead_labels.append(input_labels.numpy().tolist())
    df["cnn_feature_map"] = cnn_feature_map
    df["label"] = look_ahead_labels
    expanded_feature_df = pd.DataFrame(df["cnn_feature_map"].to_list(),
            columns=[f"f_{i + 1}" for i in range(len(cnn_feature_map[0]))], dtype=np.float32, )
    print(expanded_feature_df.info(memory_usage="deep"))
    df.drop(["cnn_feature_map"], axis=1, inplace=True)
    df = pd.concat([df, expanded_feature_df], axis=1)
    print(df)
    df.to_csv(os.path.join(roc_dir, f"cnn_feature_df_{args.label_look_ahead}.csv"), index=False, )
