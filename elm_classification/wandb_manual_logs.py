import os
import pickle

import wandb
import numpy as np

wandb.login()


def main(
    sws: int, la: int, pkl_fname: str, suffix: str = "", n_epochs: int = 20
):
    path = f"outputs/signal_window_{sws}/label_look_ahead_{la}/training_metrics"
    with open(os.path.join(path, pkl_fname), "rb") as f:
        metrics = pickle.load(f)

    with wandb.init(project="multi_features_12142021"):
        wandb.run.name = f"multi_features_sws_{sws}_la_{la}{suffix}"
        for i in range(n_epochs):
            wandb.log(
                {
                    "epoch": i + 1,
                    "training loss": metrics["train_loss"][i],
                    "validation loss": metrics["valid_loss"][i],
                    "roc": metrics["roc_scores"][i],
                    "f1": metrics["f1_scores"][i],
                },
                step=i,
            )


if __name__ == "__main__":
    main(
        sws=512,
        la=1000,
        pkl_fname="multi_features.pkl",
        suffix="",
    )
