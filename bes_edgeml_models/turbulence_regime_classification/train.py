# Python imports
import shutil
import pickle
import sys
from pathlib import Path
import time
import io
import numpy as np
from typing import Union

# Local Imports
from matplotlib import pyplot as plt

from bes_edgeml_models.base.src import utils, trainer
from bes_edgeml_models.turbulence_regime_classification.analyze import Analysis
from bes_edgeml_models.turbulence_regime_classification.options.train_arguments import TrainArguments
from bes_edgeml_models.turbulence_regime_classification.src.dataset import TurbulenceDataset
from bes_edgeml_models.turbulence_regime_classification.src.sampler import RandomBatchSampler
from bes_edgeml_models.turbulence_regime_classification.src.utils import make_labels, plot_confusion_matrix
from bes_edgeml_models.turbulence_regime_classification.models.multi_features_classification_model import MultiFeaturesClassificationModel

# ML imports
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import torch
import torchinfo
from torch.utils.data import DataLoader, BatchSampler
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    import optuna
except ImportError:
    optuna = None

def train_loop(input_args: dict,
               trial=None,
               _rank: Union[int, None] = None,
               ) -> dict:

    # parse input args
    args_obj = TrainArguments()
    if input_args and isinstance(input_args, dict):
        # format dict into list
        arg_list = []
        for key, value in input_args.items():
            if isinstance(value, bool):
                if value is True:
                    arg_list.append(f'--{key}')
            else:
                arg_list.append(f'--{key}={value}')
        input_args = arg_list
    args = args_obj.parse(arg_list=input_args)

    # output directory and files
    output_dir = Path(args.output_dir).resolve()
    args.output_dir = output_dir.as_posix()
    shutil.rmtree(output_dir.as_posix(), ignore_errors=True)
    output_dir.mkdir(parents=True)

    output_file = output_dir / args.output_file
    log_file = output_dir / args.log_file
    args_file = output_dir / args.args_file
    test_data_file, checkpoint_file = utils.create_output_paths(args)

    # create LOGGER
    LOGGER = utils.get_logger(script_name=__name__, log_file=log_file)
    LOGGER.info(args_obj.make_args_summary_string())

    LOGGER.info(f"  Output directory: {output_dir.resolve().as_posix()}")

    # save args
    LOGGER.info(f"  Saving argument file: {args_file.as_posix()}")
    with args_file.open('wb') as f:
        pickle.dump(args, f)
    LOGGER.info(f"  File size: {args_file.stat().st_size / 1e3:.1f} kB")

    # setup device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if _rank is not None:
        # override args.device for multi-GPU distributed data parallel training
        args.device = f'cuda:{_rank}'
        LOGGER.info(f'  Distributed data parallel: process rank {_rank} on GPU {args.device}')
    device = torch.device(args.device)
    LOGGER.info(f'------>  Target device: {device}')

    LOGGER.info(f'Checking for labeled datasets.')
    make_labels(Path(__file__).parent, LOGGER,
                data_dir=args.input_data_dir)
    model = MultiFeaturesClassificationModel(args).to(device)
    # distribute model for data-parallel training
    if _rank is not None:
        model = DDP(model, device_ids=[_rank])

    LOGGER.info(f"------>  Model: {args.model_name}       ")

    # display model details
    input_size = (
        args.batch_size,
        1,
        args.signal_window_size,
        8,
        8,
    )
    x = torch.rand(*input_size)
    x = x.to(device)
    if _rank is None:
        # skip torchinfo.summary if DistributedDataParallel
        tmp_io = io.StringIO()
        sys.stdout = tmp_io
        torchinfo.summary(model, input_size=input_size, device=device)
        sys.stdout = sys.__stdout__
        LOGGER.info("\t\t\t\tMODEL SUMMARY")
        LOGGER.info(tmp_io.getvalue())
    LOGGER.info(f'  Batched input size: {x.shape}')
    LOGGER.info(f"  Batched output size: {model(x).shape}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"  Model contains {n_params} trainable parameters!")

    # optimizer
    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            dampening=args.dampening,
        )
    else:
        raise ValueError

    # get the lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        verbose=True,
    )

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    # define variables for ROC and loss
    best_score_epoch = None
    best_score = -np.inf
    best_loss = np.inf
    best_loss_epoch = None
    best_cr = None
    best_cm = None

    # instantiate training object
    use_rnn = True if args.data_preproc == "rnn" else False
    engine = trainer.Run(
        model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        use_focal_loss=args.focal_loss,
        use_rnn=use_rnn,
        inverse_label_weight=args.inverse_label_weight,
    )

    # containers to hold train and validation losses
    train_loss = np.empty(0)
    valid_loss = np.empty(0)
    roc_scores = np.empty(0)

    outputs = {}

    # Create datasets
    dataset = TurbulenceDataset(args, LOGGER)
    train_set, valid_set = dataset.train_test_split(args.fraction_valid, seed=42)

    if args.dataset_to_ram:
        train_set.load_datasets()
        valid_set.load_datasets()

    training_start_time = time.time()
    # iterate through all the epochs
    LOGGER.info(f"  Begin training loop with {args.n_epochs} epochs")
    for epoch in range(args.n_epochs):
        start_time = time.time()
        with train_set as ts:
            train_loader = DataLoader(ts,
                                      batch_size=None,  # must be disabled when using samplers
                                      sampler=BatchSampler(RandomBatchSampler(ts, args),
                                                           batch_size=args.batch_size,
                                                           drop_last=True)
                                      )
            # train over an epoch
            avg_loss = engine.train(
                train_loader,
                epoch,
                print_every=args.train_print_every,
            )

        train_loss = np.append(train_loss, avg_loss)

        with valid_set as vs:
            valid_loader = DataLoader(vs,
                                      batch_size=None,  # must be disabled when using samplers
                                      sampler=BatchSampler(RandomBatchSampler(vs, args),
                                                           batch_size=args.batch_size,
                                                           drop_last=True)
                                      )
            # evaluate validation data
            avg_val_loss, preds, valid_labels = engine.evaluate(
                valid_loader,
                print_every=args.valid_print_every
            )

        valid_loss = np.append(valid_loss, avg_val_loss)

        # step the learning rate scheduler
        scheduler.step(avg_val_loss)

        # Make classification report
        class_labels = ['L-Mode',
                        'H-Mode',
                        'QH-Mode',
                        'WP QH-Mode'
                        ]
        # one_hot = np.zeros((len(valid_labels), 4))
        # for i, label in zip(one_hot, valid_labels):
        #     i[int(label)] = 1
        # roc_auc_score 'ovo' computes average of all pairwise combinations of classes. Insensitive to imbalance.
        roc_score = roc_auc_score(valid_labels, preds, multi_class='ovo', average='macro', labels=[0,1,2,3])
        roc_scores = np.append(roc_scores, roc_score)
        cm = confusion_matrix(valid_labels, preds.argmax(axis=1))
        print(cm)
        cr = classification_report(valid_labels, preds.argmax(axis=1), target_names=class_labels, zero_division=0, labels=[0,1,2,3])
        print(cr)

        elapsed = time.time() - start_time

        LOGGER.info(f"Epoch: {epoch + 1:03d} \tavg train loss: {avg_loss:.3f} \tavg val. loss: {avg_val_loss:.3f}")
        LOGGER.info(f"Epoch: {epoch + 1:03d} \tROC-AUC: {roc_score:.3f} \ttime elapsed: {elapsed:.1f} s")

        # update and save outputs
        outputs['train_loss'] = train_loss
        outputs['valid_loss'] = valid_loss
        outputs['roc_scores'] = roc_scores

        with open(output_file.as_posix(), "w+b") as f:
            pickle.dump(outputs, f)


        # track best loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_loss_epoch = epoch
        # track best roc score and save model
        if roc_score > best_score or epoch == 0:
            best_score_epoch = epoch
            best_score = roc_score
            best_cr = cr
            best_cm = cm
            LOGGER.info(f"Epoch: {epoch + 1:03d} \tBest Score: {best_score:.3f}")
            print(f"\tEpoch: {epoch + 1:03d} \nBest Confusion Matrix:\n{best_cm}")
            print(f"Classification report:\n{best_cr}")

            if not args.dry_run:
                LOGGER.info(f"  Saving model to: {checkpoint_file.as_posix()}")
                model_data = {
                    "model": model.state_dict(),
                    "preds": preds,
                }
                torch.save(model_data, checkpoint_file.as_posix())
                LOGGER.info(f"  File size: {checkpoint_file.stat().st_size / 1e3:.1f} kB")
                if args.save_onnx:
                    input_name = ['signal_window']
                    output_name = ['micro_prediction']
                    onnx_file = Path(args.output_dir) / 'checkpoint.onnx'
                    LOGGER.info(f"  Saving to ONNX: {onnx_file.as_posix()}")
                    torch.onnx.export(
                        model,
                        x[0].unsqueeze(0),
                        onnx_file.as_posix(),
                        input_names=input_name,
                        output_names=output_name,
                        verbose=True,
                        opset_version=11
                    )
                    LOGGER.info(f"  File size: {onnx_file.stat().st_size / 1e3:.1f} kB")

        # optuna hook to monitor training epochs
        if trial is not None and optuna is not None:
            trial.report(roc_score, epoch)
            # save outputs as lists in trial user attributes
            for key, item in outputs.items():
                trial.set_user_attr(key, item.tolist())
            if trial.should_prune():
                LOGGER.info("--------> Trial pruned by Optuna")
                for handler in LOGGER.handlers[:]:
                    handler.close()
                    LOGGER.removeHandler(handler)
                optuna.TrialPruned()

            LOGGER.info(roc_scores)
            LOGGER.info(trial.user_attrs['scores'])

    # Save best classification report
    LOGGER.info(f'\n{best_cr}')

    # Make plots of results
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))

    plot_confusion_matrix(best_cm,
                          classes=class_labels,
                          ax=ax1)

    ax1.text(-1.3, 0.5,
             f'{best_cr}\n'
             f'ROC: {best_score:0.2f} (epoch {best_score_epoch})    '
             f'Best Loss: {best_loss:0.2f} (epoch {best_loss_epoch})',
             transform=ax1.transAxes,
             ha='right', va='center', ma='left',
             bbox=dict(boxstyle="square", fc='w', lw=2))

    ax2.plot(outputs['valid_loss'], label='valid. loss')
    ax2.plot(outputs['train_loss'], label='training loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Performance Metrics by Epoch')
    ax2.legend()

    ax3 = ax2.twinx()
    ax3.plot(outputs['roc_scores'], label='ROC-AUC score', color='r')
    ax3.set_ylabel('ROC-AUC Score')
    ax3.legend()

    fig.suptitle(f'Summary results of {type(model).__name__}')
    plt.tight_layout()
    plt.savefig(output_dir / 'summary.png')
    plt.show()

    if args.do_analysis:
        run = Analysis(output_dir)
        run.plot_training_epochs()
        run.plot_valid_indices_analysis()

    total_elapsed = time.time() - training_start_time
    LOGGER.info(f'Training complete in {total_elapsed:0.1f}')

    # shut down logger handlers
    for handler in LOGGER.handlers[:]:
        handler.close()
        LOGGER.removeHandler(handler)

    return outputs


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # input arguments if no command line arguments in `sys.argv`
        work_dir = Path(__file__).parents[3] / 'bes-edgeml-work/regime_classification'
        n_epochs = 1
        sws = 256
        fft_num_filters = 16
        dwt_num_filters = 16
        args = {'model_name': 'multi_features_ds_v2',
                'device': 'cuda',
                'dry_run': False,
                'batch_size': 64,
                'n_epochs': n_epochs,
                'max_elms': 10,
                'fraction_valid': 0.25,
                'dataset_to_ram': True,
                'fft_num_filters': 16,
                'dwt_num_filters': 16,
                'signal_window_size': sws,
                'output_dir': work_dir / f'testdata_rc_{n_epochs}e_sws{sws}_fft{fft_num_filters}_dwt{dwt_num_filters}'
            }
    else:
        # use command line arguments in `sys.argv`
        args = None

    print('From ram')
    args['dataset_to_ram'] = True
    start = time.time()
    train_loop(input_args=args)
    print("Time loaded from ram: ", time.time() - start)

