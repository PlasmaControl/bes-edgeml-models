#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-6
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-user=lmalhotra@wisc.edu
#SBATCH --output=logs/slurm_out_%j-%2t.log
#SBATCH --error=logs/slurm_err_%j-%2t.log
module list

source "/scratch/gpfs/lm9679/miniconda3/etc/profile.d/conda.sh"

conda activate torch

echo $(which python)

srun --exclusive -n 1 -c 4 python train.py --device cuda --model_name cnn_2d --data_preproc wavelet --normalize_data --truncate_inputs --train_print_every 5000 --valid_print_every 1000 --n_epochs 30 --signal_window_size 128 --label_look_ahead 0 --num_workers 0 --lr 0.002 --weight_decay 0.05 --filename_suffix _wavelet &
srun --exclusive -n 1 -c 4 python train.py --device cuda --model_name cnn_2d --data_preproc wavelet --normalize_data --truncate_inputs --train_print_every 5000 --valid_print_every 1000 --n_epochs 30 --signal_window_size 128 --label_look_ahead 50 --num_workers 0 --lr 0.002 --weight_decay 0.05 --filename_suffix _wavelet &
srun --exclusive -n 1 -c 4 python train.py --device cuda --model_name cnn_2d --data_preproc wavelet --normalize_data --truncate_inputs --train_print_every 5000 --valid_print_every 1000 --n_epochs 30 --signal_window_size 128 --label_look_ahead 100 --num_workers 0 --lr 0.002 --weight_decay 0.05 --filename_suffix _wavelet &
srun --exclusive -n 1 -c 4 python train.py --device cuda --model_name cnn_2d --data_preproc wavelet --normalize_data --truncate_inputs --train_print_every 5000 --valid_print_every 1000 --n_epochs 30 --signal_window_size 128 --label_look_ahead 150 --num_workers 0 --lr 0.002 --weight_decay 0.05 --filename_suffix _wavelet &
srun --exclusive -n 1 -c 4 python train.py --device cuda --model_name cnn_2d --data_preproc wavelet --normalize_data --truncate_inputs --train_print_every 5000 --valid_print_every 1000 --n_epochs 30 --signal_window_size 128 --label_look_ahead 200 --num_workers 0 --lr 0.002 --weight_decay 0.05 --filename_suffix _wavelet &
srun --exclusive -n 1 -c 4 python train.py --device cuda --model_name cnn_2d --data_preproc wavelet --normalize_data --truncate_inputs --train_print_every 5000 --valid_print_every 1000 --n_epochs 30 --signal_window_size 128 --label_look_ahead 300 --num_workers 0 --lr 0.002 --weight_decay 0.05 --filename_suffix _wavelet &
srun --exclusive -n 1 -c 4 python train.py --device cuda --model_name cnn_2d --data_preproc wavelet --normalize_data --truncate_inputs --train_print_every 5000 --valid_print_every 1000 --n_epochs 30 --signal_window_size 128 --label_look_ahead 400 --num_workers 0 --lr 0.002 --weight_decay 0.05 --filename_suffix _wavelet &
srun --exclusive -n 1 -c 4 python train.py --device cuda --model_name cnn_2d --data_preproc wavelet --normalize_data --truncate_inputs --train_print_every 5000 --valid_print_every 1000 --n_epochs 30 --signal_window_size 128 --label_look_ahead 500 --num_workers 0 --lr 0.002 --weight_decay 0.05 --filename_suffix _wavelet &
wait
