"""
Track the model experimentation using Weights and Biases: https://wandb.ai/site

Create an account and save the API key in the terminal and install their python
package.

The following script expects a training metrics pickle file containing the keys:
`train_loss`, `valid_loss`, `roc_scores` and `f1_scores`.
"""
print(__doc__)
import os
import pickle

import wandb

wandb.login()


def main(
    sws: int, la: int, pkl_fname: str, suffix: str = "", n_epochs: int = 20
) -> None:
    """
    Function to manually log the outputs of the machine learning training expt.
    Logging to wandb can be done with a project name in weights and biases with
    a unique run name for an experiment under the project.

    Args:
        sws (int): Size of the signal window.
        la (int): Label lookahead.
        pkl_fname (str): Name of the pickle file containing training metrics.
        suffix (str): Any special tag for the given run.
        n_epochs (int): Number of training epochs. Defaults to 20.

    Returns:
        None
    """
    path = f"outputs/signal_window_{sws}/label_look_ahead_{la}/training_metrics"
    with open(os.path.join(path, pkl_fname), "rb") as f:
        metrics = pickle.load(f)

    with wandb.init(project="multi_features_12202021"):
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
        la=0,
        pkl_fname="multi_features_batchwise_cwt.pkl",
        suffix="_batchwise_cwt",
    )
