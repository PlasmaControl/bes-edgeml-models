import matplotlib.pyplot as plt
import numpy as np


def plot(avg_losses, all_losses):
    fig, (ax_l, ax_c, ax_r) = plt.subplots(1,3) 
    fig.set_size_inches(15, 5)
    
    # Left box - avg losses
    ax_l.plot(np.arange(1, len(avg_losses)+1), avg_losses, linestyle='-', marker='o', color='b')
    # ax_l.xticks(np.arange(1, len(avg_losses) + 1, 1.0))
    ax_l.set_title('Test Loss vs. Epochs')
    ax_l.set_ylabel('Avg Test Loss')
    ax_l.set_xlabel('Epochs')

    # Right box - all Losses histogram
    ax_r.hist(all_losses, edgecolor = 'black', bins = 5)
    ax_r.set_xlim(0,60)
    ax_r.set_title('Last Epoch - All Losses Histogram')
    ax_r.set_ylabel('Count')
    ax_r.set_xlabel('Loss bins')
    
    plt.tight_layout()
    plt.show()
    # plt.savefig('./plots/loss_plot.png')


avg_losses = [.7, .65, .53, .45, .3, .14, .09, .087]
all_losses = [10, 15, 25, 27, 28, 59, 70, 74, 75, 80]

plot(avg_losses, all_losses)
