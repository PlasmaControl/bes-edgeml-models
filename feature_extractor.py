import os

import pandas as pd
import torch

from src import utils, dataset
from options.test_arguments import TestArguments

activation = dict()


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach().numpy()

    return hook


if __name__ == "__main__":
    # instantiate the elm_model and load the checkpoint
    model_ckpt_dir = "model_checkpoints/signal_window_16"
    args, parser = TestArguments().parse(verbose=True)
    utils.test_args_compat(args, parser, infer_mode=True)

    LOGGER = utils.get_logger(script_name=__name__)
    model_cls = utils.create_model(args.model_name)
    model = model_cls(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(model)

    model_ckpt_path = os.path.join(
        model_ckpt_dir,
        f"{args.model_name}_{args.data_mode}_lookahead_{args.label_look_ahead}.pth",
    )
    print(f"Using elm_model checkpoint: {model_ckpt_path}")
    model.load_state_dict(
        torch.load(
            model_ckpt_path,
            map_location=device,
        )["model"]
    )
    model.conv.register_forward_hook(get_activation("conv"))

    data_cls = utils.create_data(args.data_preproc)
    data_obj = data_cls(args, LOGGER)
    all_elms, all_data = data_obj.get_data()
    signals, labels, valid_indices, window_start = all_data

    print(f"Signals shape: {signals.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Valid indices shape: {valid_indices.shape}")
    print(f"Window start shape: {window_start.shape}")

    data = dataset.ELMDataset(
        args, signals, labels, valid_indices, window_start, LOGGER
    )
    input_signals, input_labels = data.__getitem__(0)
    print(input_signals.shape)
    y = model(input_signals.unsqueeze_(0))
    print(y.shape)
    print(activation["conv"].shape)
