#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-12
#SBATCH --output=logs/slurm_out_%j.log
#SBATCH --error=logs/slurm_err_%j.log
module list

source "/scratch/gpfs/lm9679/miniconda3/etc/profile.d/conda.sh"

conda activate pt

echo $(which python)

srun --exclusive -n 1 -c 8 python train.py --device cuda --model_name multi_features --data_preproc unprocessed --signal_window_size 128 --label_look_ahead 0 --truncate_inputs --normalize_data --n_epochs 20 --max_elms -1 --multi_features --use_fft --filename_suffix _hop_2 &
srun --exclusive -n 1 -c 8 python train.py --device cuda --model_name multi_features --data_preproc unprocessed --signal_window_size 128 --label_look_ahead 200 --truncate_inputs --normalize_data --n_epochs 20 --max_elms -1 --multi_features --use_fft --filename_suffix _hop_2 &
srun --exclusive -n 1 -c 8 python train.py --device cuda --model_name multi_features --data_preproc unprocessed --signal_window_size 128 --label_look_ahead 400 --truncate_inputs --normalize_data --n_epochs 20 --max_elms -1 --multi_features --use_fft --filename_suffix _hop_2 &
wait
