# Python imports
import shutil
import pickle
import sys
from pathlib import Path
import time
import io
import numpy as np
from typing import Union
from matplotlib import pyplot as plt

# Local Imports
from bes_edgeml_models.base.src import utils, trainer
from bes_edgeml_models.turbulence_regime_classification.src.sampler import RandomBatchSampler
from bes_edgeml_models.velocimetry.options.train_arguments import TrainArguments
from bes_edgeml_models.velocimetry.analyze import Analysis
from bes_edgeml_models.velocimetry.src.dataset import VelocimetryDataset

# ML imports
from sklearn.metrics import mean_absolute_percentage_error as mape
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

    # make sure regression is set for error
    args.regression = True
    args.dataset_to_ram = True

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

    # Get model
    model_class = utils.create_model_class(args.model_name)
    model = model_class(args)
    model = model.to(device)

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

    criterion = torch.nn.MSELoss(reduction="none")

    # define variables for loss
    best_loss = np.inf
    best_score = np.inf

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
    scores = np.empty(0)

    outputs = {}

    # Create datasets
    dataset = VelocimetryDataset(args, LOGGER)
    train_set, valid_set = dataset.train_test_split(args.fraction_valid, seed=42)
    if args.dataset_to_ram:
        # Load datasets into ram
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

        score = mape(valid_labels, preds, multioutput='uniform_average')
        scores = np.append(scores, score)

        # step the learning rate scheduler
        scheduler.step(avg_val_loss)

        elapsed = time.time() - start_time

        LOGGER.info(f"Epoch: {epoch + 1:03d} \tavg train loss: {avg_loss:.3f} \tavg val. loss: {avg_val_loss:.3f}")
        LOGGER.info(f"Epoch: {epoch + 1:03d} \ttime elapsed: {elapsed:.1f} s")

        # update and save outputs
        outputs['train_loss'] = train_loss
        outputs['valid_loss'] = valid_loss
        outputs['scores'] = scores
        outputs['score_type'] = 'Mean Absolute Percentage Error'

        with open(output_file.as_posix(), "w+b") as f:
            pickle.dump(outputs, f)


        # track best loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f"Epoch: {epoch + 1:03d} \tBest Score: {best_loss:.3f}")

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

                dataset.save_test_data()

        # optuna hook to monitor training epochs
        if trial is not None and optuna is not None:
            trial.report(avg_val_loss, epoch)
            # save outputs as lists in trial user attributes
            for key, item in outputs.items():
                trial.set_user_attr(key, item.tolist())
            if trial.should_prune():
                LOGGER.info("--------> Trial pruned by Optuna")
                for handler in LOGGER.handlers[:]:
                    handler.close()
                    LOGGER.removeHandler(handler)
                optuna.TrialPruned()

            LOGGER.info(avg_val_loss)
            LOGGER.info(trial.user_attrs['scores'])

    fig, ax = plt.subplots(1, 1)

    ax.plot(valid_loss, label='Val. Loss')
    ax.plot(train_loss, label='Training Loss')
    ax.set_title('')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Performance Metrics by Epoch')
    ax.legend()

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
        work_dir = Path(__file__).parents[3] / 'bes-edgeml-work/velocimetry/'
        args = {'model_name': 'multi_features_velocimetry',
                'device': 'cuda',
                'dry_run': False,
                'batch_size': 16,
                'n_epochs': 10,
                'max_elms': -1,
                'fraction_valid': 0.15,
                'dataset_to_ram': True,
                'fft_num_filters': 16,
                'dwt_num_filters': 16,
                'signal_window_size': 64,
                'fc1_size': 1024,
                'fc2_size': 512,
                'output_dir': work_dir / 'test_data_mf_10e_sws256'
            }
    else:
        # use command line arguments in `sys.argv`
        args = None

    train_loop(input_args=args)

