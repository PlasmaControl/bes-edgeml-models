#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=1-0
#SBATCH -A pppl
###SBATCH --dependency=141655

module load edgeml
module list

source "/scratch/gpfs/dsmith/miniconda/etc/profile.d/conda.sh"
which conda
which python3
conda activate tf

env | egrep "SLURM|HOST"

db_file='/home/dsmith/scratch/optuna/hpo-02.db'

echo "Hostname: ${HOSTNAME}"
echo "DB file: ${db_file}"
echo "Study name: ${study_name}"

srun python3 hpo-create.py $db_file
