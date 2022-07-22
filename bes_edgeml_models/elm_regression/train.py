"""
Main module for training ELM prediction models

To train an ELM prediction model, call `train_loop()` using
a recipe similar to `if __name__=="__main__"` block.
"""

import sys
import time
import pickle
import shutil
import io
from typing import Union
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, r2_score

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torchinfo

try:
    import optuna
except ImportError:
    optuna = None

try:
    from ..base.options.train_arguments import TrainArguments
    from ..base.src import utils, trainer, dataset
    from ..base.analyze import Analysis
except ImportError:
    from bes_edgeml_models.base.options.train_arguments import TrainArguments
    from bes_edgeml_models.base.src import utils, trainer, dataset
    from bes_edgeml_models.elm_regression.analyze import Analysis


def train_loop(
    input_args: Union[list,dict,None] = None,
    select_elms: Union[list, np.ndarray] = None,
    trial = None,  # optuna `trial` object
    _rank: Union[int, None] = None, # process rank for data parallel dist. training; *must* be last arg
) -> dict:
    """Run a training pipeline: parse inputs, prepare data, create model, train over epochs.

    Args:
    -----
        input_args (list|dict): (Optional) Input arguements as dict or list of strings
        trial: (Optional) Optuna trial object to report training progress and enable pruning
        _rank: (Optional) Used for Distributed Data Parallel training by `distributed_train.py`
    """

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

    args.data_preproc = 'regression'
    args.label_look_ahead = 0
    args.truncate_buffer = 0
    args.oversample_active_elm = False

    if args.regression != 'log':
        args.inverse_label_weight = False

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
    LOGGER.info(f"  File size: {args_file.stat().st_size/1e3:.1f} kB")

    # setup device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if _rank is not None:
        # override args.device for multi-GPU distributed data parallel training
        args.device = f'cuda:{_rank}'
        LOGGER.info(f'  Distributed data parallel: process rank {_rank} on GPU {args.device}')
    device = torch.device(args.device)
    LOGGER.info(f'------>  Target device: {device}')

    # create train, valid and test data
    data_cls = utils.create_data_class(args.data_preproc)
    data_obj = data_cls(args, LOGGER, select_elms=select_elms)
    train_data, valid_data, test_data = data_obj.get_data(verbose=False)

    # dump test data into a file
    if not args.dry_run:
        LOGGER.info(f"  Test data will be saved to: {test_data_file.as_posix()}")
        with test_data_file.open('wb') as f:
            pickle.dump(
                {
                    "signals": test_data[0],
                    "labels": test_data[1],
                    "sample_indices": test_data[2],
                    "window_start": test_data[3],
                    "elm_indices": test_data[4],
                },
                f,
            )
        LOGGER.info(f"  File size: {test_data_file.stat().st_size/1e6:.1f} MB")

    # create datasets
    train_dataset = dataset.ELMDataset(args, *train_data[0:4], logger=LOGGER)
    valid_dataset = dataset.ELMDataset(args, *valid_data[0:4], logger=LOGGER)

    # training and validation dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # model class and model instance
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

    # loss function
    criterion = torch.nn.MSELoss(reduction="none")

    # define variables for ROC and loss
    best_score = -np.inf

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

    training_start_time = time.time()
    # iterate through all the epochs
    LOGGER.info(f"  Begin training loop with {args.n_epochs} epochs")
    for epoch in range(args.n_epochs):
        start_time = time.time()

        # train over an epoch
        avg_loss = engine.train(
            train_loader, 
            epoch, 
            print_every=args.train_print_every,
        )
        avg_loss = np.sqrt(avg_loss)
        train_loss = np.append(train_loss, avg_loss)

        # evaluate validation data
        avg_val_loss, preds, valid_labels = engine.evaluate(
            valid_loader, 
            print_every=args.valid_print_every
        )
        avg_val_loss = np.sqrt(avg_val_loss)
        valid_loss = np.append(valid_loss, avg_val_loss)

        # step the learning rate scheduler
        scheduler.step(avg_val_loss)

        # R2 score
        score = r2_score(valid_labels, preds)
        scores = np.append(scores, score)
        score_type = 'R2'

        elapsed = time.time() - start_time

        LOGGER.info(
            f"Epoch: {epoch+1:03d} \ttrain loss: {avg_loss:.3f} \tval. loss: {avg_val_loss:.3f} " 
            f"\tscore: {score:.3f} \ttime elapsed: {elapsed:.1f} s"
        )

        # update and save outputs
        outputs['train_loss'] = train_loss
        outputs['valid_loss'] = valid_loss
        outputs['scores'] = scores
        outputs['score_type'] = score_type # A helper to keep track of which score is being used. Useful in plotting.

        with open(output_file.as_posix(), "w+b") as f:
            pickle.dump(outputs, f)

        # track best f1 score and save model
        if score > best_score or epoch == 0:
            best_score = score
            LOGGER.info(f"Epoch: {epoch+1:03d} \tBest Score: {best_score:.3f}")
            if not args.dry_run:
                LOGGER.info(f"  Saving model to: {checkpoint_file.as_posix()}")
                model_data = {
                    "model": model.state_dict(), 
                    "preds": preds,
                }
                torch.save(model_data, checkpoint_file.as_posix())
                LOGGER.info(f"  File size: {checkpoint_file.stat().st_size/1e3:.1f} kB")                
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
                    LOGGER.info(f"  File size: {onnx_file.stat().st_size/1e3:.1f} kB")                

        # optuna hook to monitor training epochs
        if trial is not None and optuna is not None:
            trial.report(score, epoch)
            # save outputs as lists in trial user attributes
            for key, item in outputs.items():
                trial.set_user_attr(key, item.tolist())
            if trial.should_prune():
                LOGGER.info("--------> Trial pruned by Optuna")
                for handler in LOGGER.handlers[:]:
                    handler.close()
                    LOGGER.removeHandler(handler)
                optuna.TrialPruned()

            LOGGER.info(scores)
            LOGGER.info(trial.user_attrs['scores'])

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


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # input arguments if no command line arguments in `sys.argv`
        input_args = {
            'model_name': 'multi_features_regression',
            'input_data_file': '/Users/drsmith/Documents/repos/edgeml-models/bes_models_2/sample_data/sample_elm_events.hdf5',
            'max_elms':10,
            'n_epochs':2,
            'fraction_valid':0.2,
            'fraction_test':0.2,
            'signal_window_size':64,
            'label_look_ahead':200,
            'lr':1e-5,
            # 'cnn_layer1_num_filters':8,
            # 'cnn_layer2_num_filters':8,
            # 'raw_num_filters':0,
            'regression':True,
            'optimizer':'adam',
            'inverse_label_weight':True,
        }
    else:
        # use command line arguments in `sys.argv`
        input_args = None
    train_loop(input_args=input_args)
