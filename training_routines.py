import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import pickle

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


def plot_loss_beta(betas, use_saved=True):
    # routine to plot losses of a model with respect to beta

    # Directories of saved data
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Training Model
    losses = []
    for beta in betas:
        print(f'BETA: {beta}')
        args.vae_beta = beta

        fpath = os.path.join(ROOT_DIR, f'visualizations/outputs/pickle_objs/loss_dicts/{args.model_name}/beta'
                                       f'{args.vae_beta}_epochs{args.n_epochs}.pkl')

        if use_saved and os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                loss_dict = pickle.load(f)
        else:
            data_cls = utils.create_data(args.data_preproc)
            data_obj = data_cls(args, LOGGER)
            loss_dict = train_loop(args, data_obj, test_datafile_name=f"test_data_lookahead_{args.label_look_ahead}"
                                                                      f"_{args.data_preproc}.pkl", )
            with open(fpath, 'w+b') as f:
                pickle.dump(loss_dict, f)

        losses.append(loss_dict.get('validation').get('loss'))

    # Plotting
    N = len(betas)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.suptitle(f'Model {args.model_name}')

    # create the new map
    cmap = cm.get_cmap('jet', len(betas))

    # Make dummie mappable
    dummie_cax = ax1.scatter(np.arange(0, N), (np.arange(0, N)), c=betas, cmap=cmap)
    # Clear axis
    ax1.cla()
    for i, l in enumerate(losses):
        ax1.plot(l, c=cmap(i))

    cbar = fig.colorbar(dummie_cax, ax=ax1, ticks=betas)
    cbar.set_label(r'$\beta$')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('ELBO Loss')
    ax1.set_title(r'Validation Loss vs. $\beta$')

    np_losses = np.array(losses)
    min_losses = np.min(np_losses, axis=1)
    ax2.plot(betas, min_losses)
    ax2.set_title(r'Minimum Loss vs. $\beta$')
    ax2.set_xlabel(r'$\beta$')
    ax2.set_ylabel('Min Loss')

    plt.show()


def show_mse_loss(use_saved=True, superimposed=True):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    fpath = os.path.join(ROOT_DIR, f'visualizations/outputs/pickle_objs/loss_dicts/{args.model_name}/beta'
                                   f'{args.vae_beta}_epochs{args.n_epochs}'
                                   f'{"_" + args.balance_data if args.balance_data else ""}.pkl')

    if use_saved and os.path.exists(fpath):
        with open(fpath, 'rb') as f:
            loss_dict = pickle.load(f)
    else:
        data_cls = utils.create_data(args.data_preproc)
        data_obj = data_cls(args, LOGGER)

        loss_dict = train_loop(args, data_obj, test_datafile_name=f"test_data_lookahead_{args.label_look_ahead}"
                                                                  f"_{args.data_preproc}.pkl", )
        with open(fpath, 'w+b') as f:
            pickle.dump(loss_dict, f)

    epochs = np.arange(1, args.n_epochs + 1)
    train_losses = loss_dict.get('training')
    val_losses = loss_dict.get('validation')

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    train_loss = np.array(train_losses.get('loss'))
    train_klloss = args.vae_beta * np.array(train_losses.get('kl_loss'))
    train_logloss = np.array(train_losses.get('log_likelihood_loss'))
    train_mseloss = np.array(train_losses.get('mse_loss'))
    train_norm = np.max(np.abs(np.array([train_loss, train_logloss, train_mseloss], copy=False)))

    val_loss = np.array(val_losses.get('loss'))
    val_klloss = args.vae_beta * np.array(val_losses.get('kl_loss'))
    val_logloss = np.abs(np.array(val_losses.get('log_likelihood_loss')))
    val_mseloss = np.abs(np.array(val_losses.get('mse_loss')))
    val_norm = np.max(np.abs(np.array([val_loss, val_logloss, val_mseloss], copy=False)))

    print(f'Training Likelihood Normalization Factor: {train_norm}')
    print(f'Training MSE Normalization Factor: {train_mseloss.max()}')
    print(f'Validation Likelihood Normalization Factor: {val_norm}')
    print(f'Validation MSE Normalization Factor: {val_mseloss.max()}')

    ax1.plot(epochs, train_loss / train_norm, label='Total Loss', c='k')
    ax1.plot(epochs, -1 * train_logloss / train_norm, label='Log Likelihood Loss', c='g')

    ax2.plot(epochs, val_loss / val_norm, c='k')
    ax2.plot(epochs, -1 * val_logloss / val_norm, c='g')

    if superimposed:
        ax1.plot(epochs,
                 train_klloss / train_norm - (train_klloss.min() / train_klloss.max() - train_loss.min() / train_norm),
                 label=f'{args.vae_beta} * $D_{{kl}}$',
                 c='r')
        ax1.plot(epochs,
                 train_mseloss / train_mseloss.max() - (
                         train_mseloss.min() / train_mseloss.max() - train_loss.min() / train_norm),
                 label='MSE Loss',
                 c='b')
        ax2.plot(epochs,
                 val_klloss / val_norm - (val_klloss.min() / val_klloss.max() - val_loss.min() / val_norm),
                 c='r')
        ax2.plot(epochs,
                 val_mseloss / val_mseloss.max() - (val_mseloss.min() / val_mseloss.max() - val_loss.min() / val_norm),
                 c='b')
    else:
        ax1.plot(epochs, train_klloss / train_norm, label=f'{args.vae_beta} * KL Divergence', c='r')
        ax1.plot(epochs, train_mseloss / train_mseloss.max(), c='b')

        ax2.plot(epochs, val_klloss / val_norm, c='r')
        ax2.plot(epochs, val_mseloss / val_mseloss.max(), c='b')

    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Normalized Loss')
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Normalized Loss')

    fig.suptitle(f'{args.model_name} MSE and Log Likelihood Loss (Normalized)')
    fig.legend()

    plt.show()
    return


if __name__ == "__main__":
    args, parser = TrainArguments().parse(verbose=True)
    LOGGER = utils.make_logger(script_name=__name__,
                               log_file=os.path.join(args.log_dir,
                                                     f"output_logs_{args.model_name}{args.filename_suffix}.log", ), )

    b = list(range(0, 226, 25))
    b[0] = 1
    plot_loss_beta(betas=b)
    exit(0)
