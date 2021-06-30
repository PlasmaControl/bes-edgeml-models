import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle
from matplotlib import pyplot as plt
from autoencoder import AE_simple, device
import config, data
import numpy as np

def train(model: AE_simple, 
    train_dataloader: DataLoader, 
    test_dataloader: DataLoader, 
    optimizer,
    scheduler, 
    loss_fn, 
    epochs: int = config.epochs,
    print_output: bool = True):

    tb = SummaryWriter(log_dir = 'runs/one_hidden_layer_', comment = '300')

    epoch_avg_losses = []
    all_losses = None

    for t in range(epochs):
        if(print_output):
            print(f"Epoch {t+1}\n-------------------------------")

        train_loop(model, train_dataloader, optimizer, loss_fn)

        if(t == epochs - 1):
            epoch_avg_loss, all_losses = validation_loop(model, test_dataloader, loss_fn, all_losses = True)
        else:
            epoch_avg_loss = validation_loop(model, test_dataloader, loss_fn)

        epoch_avg_losses.append(epoch_avg_loss)
        tb.add_scalar('Epoch Avg Loss', epoch_avg_loss, t)

        # Change optimizer learning rate
        scheduler.step(epoch_avg_loss)
    
    if(print_output):
        print("Done Training!")

    return epoch_avg_losses, all_losses
    
def train_loop(model, dataloader: DataLoader, optimizer, loss_fn, print_output: bool = True):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print (name, param.data)
            #         break
            # param = model.parameters()[0][0,0]
            if(print_output):
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def validation_loop(model, dataloader: DataLoader, loss_fn, all_losses: bool = False, print_output: bool = True):
    size = len(dataloader.dataset)
    test_loss = 0

    if(all_losses):
        all_epoch_losses = []

    model.eval()
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            cur_sample_loss = loss_fn(pred, y).item()
            test_loss += cur_sample_loss 

            if(all_losses):
                all_epoch_losses.append(cur_sample_loss)

    test_loss /= size

    if(print_output):
        print(f"Validation Dataset Avg loss: {test_loss:>8f} \n")

    if(all_losses):
        return test_loss, all_epoch_losses
    else:
        return test_loss

def plot(avg_losses, all_losses):
    fig, (ax_l, ax_r) = plt.subplots(1,2) 
    fig.set_size_inches(15, 5)
    
    # Left box - avg losses
    ax_l.plot(np.arange(1, len(avg_losses)+1), avg_losses, linestyle='-', marker='o', color='b')
    # ax_l.xticks(np.arange(1, len(avg_losses) + 1, 1.0))
    ax_l.set_title('Test Loss vs. Epochs')
    ax_l.set_ylabel('Avg Test Loss')
    ax_l.set_xlabel('Epochs')

    # Right box - all Losses histogram
    # ax_r.set_xlim(0,.5)
    num_bins = 20
    ax_r.hist(all_losses, edgecolor = 'black', bins = num_bins, log = True)

    ax_r.set_xticks(np.linspace(np.amin(all_losses), np.amax(all_losses), num = num_bins, endpoint = True))
    ax_r.set_xticklabels(ax_r.get_xticks(), rotation = 90)

    ax_r.set_title('Last Epoch - All Losses histogram')
    ax_r.set_ylabel('Batch Count')
    ax_r.set_xlabel('Loss bins')
    
    plt.tight_layout()
    plt.show()
    # plt.savefig('./plots/loss_plot.png')

if __name__ == '__main__':
    # The model
    model = AE_simple(300)
    model = model.to(device)

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=config.learning_rate, 
        momentum=0.9, 
        weight_decay=config.l2_factor)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            verbose=True,
            eps=1e-6,
        )    

    # Get datasets and form dataloaders
    data_ = data.Data(kfold=False, balance_classes=config.balance_classes)
    train_data, valid_data, test_data = data_.get_data(shuffle_sample_indices=True) 

    test_data_file = 'test_dataset'
    # dump test data into to a file
    with open(test_data_file, "wb") as f:
        pickle.dump(
            {
                "signals": test_data[0],
                "labels": test_data[1],
                "sample_indices": test_data[2],
                "window_start": test_data[3],
            },
            f,
        )

    train_dataset = data.ELMDataset(
        *train_data,
        config.signal_window_size,
        config.label_look_ahead,
        stack_elm_events=False,
        transform=None,
        for_autoencoder = True
    )

    valid_dataset = data.ELMDataset(
        *valid_data,
        config.signal_window_size,
        config.label_look_ahead,
        stack_elm_events=False,
        transform=None,
        for_autoencoder = True
    )

    batch_size = config.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    epoch_avg_losses, all_losses = train(model, 
        train_dataloader, 
        valid_dataloader, 
        optimizer, 
        scheduler, 
        loss_fn)

    # plot(epoch_avg_losses, all_losses)

    # Save the model - weights and structure
    model_save_path = './trained_models/simple_ae.pth'
    torch.save(model, model_save_path)