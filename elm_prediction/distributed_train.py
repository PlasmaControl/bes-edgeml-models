import os
import torch

try:
    from .train import train_loop
except ImportError:
    from train import train_loop


def _distributed_train_loop_wrapper(rank, world_size, *train_loop_args):
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)
    train_loop(*train_loop_args, _rank=rank)
    torch.distributed.destroy_process_group()


def run_distributed_train_loop(*train_loop_args):
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
        _distributed_train_loop_wrapper,
        args=(world_size, *train_loop_args),
        nprocs=world_size,
        join=True,
        )

