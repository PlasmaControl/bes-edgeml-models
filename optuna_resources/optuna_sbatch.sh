#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=2-0
#SBATCH --account=pppl

source "/scratch/gpfs/dsmith/miniconda/etc/profile.d/conda.sh"
conda activate pytorch
conda info -e
echo "Python exec: $(which python)"

PYTHON_CODE=$(cat <<END

import optuna_run

func = optuna_run.optuna_test

optuna_run.optuna_run(
    db_name=func.__name__,
    n_gpus=4,  # 4 for compute node
    n_workers_per_gpu=3,  # 3 max
    n_trials_per_worker=12,
    n_startup_trials=400,
    objective_func=func,
)

END
)


echo "Python code:"
echo "${PYTHON_CODE}"


srun python -c "${PYTHON_CODE}"