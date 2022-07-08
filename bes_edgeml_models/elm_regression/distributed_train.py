import os
import sys

import torch

try:
    from models.elm_prediction.train import train_loop
    from models.bes_edgeml_models.options.train_arguments import TrainArguments
except ImportError:
    from models.elm_prediction.train import train_loop
    from models.bes_edgeml_models.options.train_arguments import TrainArguments


def _train_loop_wrapper(rank, world_size, *train_loop_args):
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)
    train_loop(*train_loop_args, _rank=rank)
    torch.distributed.destroy_process_group()


def distributed_train_loop(*train_loop_args):
    """
    train_loop_args: Must be ordered argument list for train_loop(), excluding `rank`
    """
    args = train_loop_args[0]

    assert args.distributed != 1
    assert args.device.startswith('cuda')
    assert torch.distributed.is_available()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    world_size = torch.cuda.device_count() if args.distributed == -1 else args.distributed
    
    torch.multiprocessing.spawn(
        _train_loop_wrapper,
        args=(world_size, *train_loop_args),
        nprocs=world_size,
        join=True,
        )

if __name__=='__main__':
    if len(sys.argv) == 1:
        # input arguments if no command line arguments in `sys.argv`
        arg_list = [
            "--output_dir", "run_dir_distributed",
            "--device", "cuda",
            "--distributed", "-1",
        ]
    else:
        # use command line arguments in `sys.argv`
        arg_list = None
    args = TrainArguments().parse(arg_list=arg_list)
    distributed_train_loop(args)
