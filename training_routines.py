import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm

from options.train_arguments import TrainArguments
from src import utils
from train import train_loop


def plot_loss(train_loss, valid_loss):
    fig = plt.figure()
    ax = fig.add_subplot()

    fig.suptitle('Training and Validation ELBO Loss')
    ax.plot(train_loss, label='Training Loss')
    ax.plot(valid_loss, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ELBO Loss')
    ax.legend()

    plt.show()


def plot_loss_beta(betas, losses):
    # routine to plot losses of a model with respect to beta
    N = len(betas)
    fig = plt.figure()
    ax1 = fig.add_subplot()
    fig.suptitle(f'Model {args.model_name}')

    # create the new map
    cmap = cm.get_cmap('jet', len(betas))

    # Make dummie mappable
    dummie_cax = ax1.scatter(np.arange(0, N), (np.arange(0, N)), c=betas, cmap=cmap)
    # Clear axis
    ax1.cla()
    for i, l in enumerate(losses):
        ax1.plot(l, c=cmap(i))

    cbar = fig.colorbar(dummie_cax, ticks=betas)
    cbar.set_label('Beta')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('ELBO Loss')
    ax1.set_title(f'Validation Loss vs. Beta')

    plt.show()


if __name__ == "__main__":
    args, parser = TrainArguments().parse(verbose=True)
    LOGGER = utils.make_logger(script_name=__name__, log_file=os.path.join(args.log_dir,
                                                                           f"output_logs_{args.model_name}{args.filename_suffix}.log", ), )
    data_cls = utils.create_data(args.data_preproc)
    data_obj = data_cls(args, LOGGER)
    N = 25
    losses = []
    betas = list(range(0, 2))
    betas[0] = 1
    for beta in betas:
        args.vae_beta = beta
        train_loss, valid_loss, kl_loss, recon_loss = train_loop(args, data_obj,
                                                                 test_datafile_name=f"test_data_lookahead_{args.label_look_ahead}_{args.data_preproc}.pkl", )
        losses.append(valid_loss)

    plot_loss_beta(betas, losses)
