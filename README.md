# BES EdgeML project
BES EdgeML project is an effort to develop machine learning (ML) models for the real-time identification of edge-localized-mode (ELM) events and the turbulence properties of confinement regimes using the 2D Beam Emission Spectroscopy (BES) system at DIII-D. The “edge ML” models will be deployed on a high-throughput FPGA accelerator for integration in the real-time plasma control system (PCS).

The whole project can be structured as follows:
<p align='center'>
    <img src='assets/project_structure.png' alt='project_structure' width='300'/>
</p>

The code consists of the PyTorch implementations for various models which are located inside the `models/` directory.

`notebooks/` directory contains various jupyter notebooks which are used for experimentation with the data preprocessing pipelines and calculating ROC plots.

`src/` directory contains scripts for various utility functions, data preprocessing and boilerplate code used for training and evaluation. One of the key features of the data preprocessing pipelines is the way inputs and labels are created. Inputs are 3D-tensors where the leading dimension contains time-steps according to `signal_window_size` and the last two dimensions contain `8x8` BES spatial data comprising the output from the 64 channels of the detector. This can be understood in more detail by the following figure created by Dr. David Smith.

<p align='center'>
    <img src='assets/signal_window_diagram.png' alt='signal_window' width='1000'/>
</p>

`archives/model_tools/` is a Python module and the primary set of tools for training ML models.

- `model_tools/data.py` - Prepare BES data for training runs.  The module looks for BES data in `data/` (ignored
by git).
- `model_tools/model.py` - Specify a model
- `model_tools/train.py` - Train a model.  The trained model is saved in `models/` (ignored by git).
- `model_tools/analyze.py` - Analyze results of a trained model

`archives/hpo/` is a directory with python modules and Slurm scripts to perform hyper-parameter optimization with Optuna

- `hpo/hpo-sbatch.sh` - Slurm script.  Calls `hpo/hpo-create.py`.
- `hpo/hpo-create.py` - Create Optuna "study" and launch parallel processes that call `hpo/hpo-launch.py`.
- `hpo/hpo-launch.py` - Run trials in a single process.
- `hpo/hpo-check.py` - Check and plot results/status of a study.

`archives/multitrain/` - Out-of-date but similar to `hpo/`.  The scripts and python modules in `multi-train/` are intended
to perform multiple training runs for a single set of model parameters, for example, the "best" parameters from HPO.

`notebooks/` - Out-of-date notebooks from earlier development.
