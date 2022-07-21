#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=0-8
#SBATCH -A pppl

module list

env | egrep "SLURM|HOST"

source "/scratch/gpfs/dsmith/miniconda/etc/profile.d/conda.sh"

which conda
which python3

conda activate tf

echo "Hostname: ${HOSTNAME}"

srun python3 multitrain-launch.py
