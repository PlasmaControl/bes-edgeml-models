#!/bin/bash

# sbatch scripts must specify `#SBATCH` directives prior to any regular command.
# Multiple hashes, like `##SBATCH`, are comments and ignored.

## nodes and time

#SBATCH --nodes=1  # compute nodes
#SBATCH --time=1-0  # time limit in format `days-hours`

## resources per node

#SBATCH --ntasks-per-node=1   # MPI tasks per node
#SBATCH --cpus-per-task=128  # logical CPUs per task
#SBATCH --gres=gpu:4  # total GPUs
#SBATCH --mem=240G  # memory per node (allow ~10 GB for OS)

## dependency

#SBATCH --dependency=164559  # run only after dependency job completes

#SBATCH --account pppl  # account to charge
#SBATCH --mail-type=all  # email for all events
#SBATCH --mail-user=drsmith@pppl.gov  # email address


# setup environment
module load edgeml
module list
source "/scratch/gpfs/dsmith/miniconda/etc/profile.d/conda.sh"
conda activate tf


# verify environment
which conda
which python3
env | egrep "SLURM|HOST"


# run job
srun python3 hpo-create.py
