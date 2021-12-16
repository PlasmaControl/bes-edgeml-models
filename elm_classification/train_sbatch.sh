#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=0-5
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-user=lmalhotra@wisc.edu
#SBATCH --output=logs/slurm_out_%j.log
#SBATCH --error=logs/slurm_err_%j.log
module list

source "/scratch/gpfs/lm9679/miniconda3/etc/profile.d/conda.sh"

conda activate torch

echo $(which python)

python train.py --device cuda --model_name multi_features --data_preproc unprocessed --signal_window_size 128 --label_look_ahead 0 --truncate_inputs --normalize_data --n_epochs 20 --max_elms -1 --multi_features --use_fft
