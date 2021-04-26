#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=1-0
#SBATCH -A pppl
#SBATCH --dependency=141655

db_file='/home/dsmith/scratch/optuna/hpo-01.db'
study_name='study-01'
n_startup_trials='0'

module list

env | egrep "SLURM|HOST"

source "/scratch/gpfs/dsmith/miniconda/etc/profile.d/conda.sh"

which conda
which python3

conda activate tf

echo "Hostname: ${HOSTNAME}"
echo "DB file: ${db_file}"
echo "Study name: ${study_name}"

srun python3 hpo-create.py $db_file $study_name $n_startup_trials
exitcode=$?

exit $exitcode