#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-6
#SBATCH --output=logs/slurm_out_%j.log
#SBATCH --error=logs/slurm_err_%j.log
module list

source "/scratch/gpfs/lm9679/miniconda3/etc/profile.d/conda.sh"

conda activate pt

echo $(which python)

#srun -N 1 -n 1 python train.py --device cuda --model_name multi_features --data_preproc unprocessed --signal_window_size 512 --label_look_ahead 0 --normalize_data --n_epochs 20 --max_elms -1 --multi_features --use_fft
srun -N 1 -n 1 python train_ds.py
