#!/bin/bash
#SBATCH --chdir=/scratch/gpfs/jz6896/turbulence_regime_classification/turbulence_regime_classification/work/
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=0-0:0:30
#SBATCH --account=pppl
#SBATCH --mail-type=all
#SBATCH --mail-user=jz6896@princeton.edu
#SBATCH --job-name=test_regime_classification_2
#SBATCH --output=%x.out

source "/home/jz6896/miniconda3/etc/profile.d/conda.sh"
conda activate pytorch
conda info -e
echo "Python exec: $(which python)"

PYTHON_CODE=$(cat <<END
from pathlib import Path
import sys
sys.path.append("/projects/EKOLEMEN/jzimmerman/turbulence_regime_classification/bes-edgeml-models")
from turbulence_regime_classification.train import test_sbatch
test_sbatch()
train_loop(
    input_data_dir= '/scratch/gpfs/jz6896/data/',
    labeled_data_dir='data/labeled_datasets',
    output_dir='${SLURM_JOB_NAME}',
    max_elms=-1,
    n_epochs=20,
    signal_window_size=128,
    lr=1e-4,
    raw_num_filters=16,
    fft_num_filters=16,
    fft_nbins=2,
    dwt_num_filters=16,
    dataset_to_ram=False,
    valid_indices_method=3,
)
END
)

echo "Python code:"
echo "${PYTHON_CODE}"

srun python -c "${PYTHON_CODE}"

