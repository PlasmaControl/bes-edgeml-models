# BES EdgeML project
BES EdgeML project is an effort to develop machine learning (ML) models for the real-time identification of edge-localized-mode (ELM) events and the turbulence properties of confinement regimes using the 2D Beam Emission Spectroscopy (BES) system at DIII-D. The “edge ML” models will be deployed on a high-throughput FPGA accelerator for integration in the real-time plasma control system (PCS).

The whole project can be structured as follows:
<p align='center'>
    <img src='assets/project_structure.png' alt='project_structure' width='300'/>
</p>

The code consists of the PyTorch implementations for various models which are located inside the [`models/`](models/) directory.

[`data_preprocessing/`](data_preprocessing/) consists of scripts responsible for the data preparation from raw BES data 
for training and evaluation.

[`notebooks/`](notebooks/) directory contains various jupyter notebooks which are used for experimentation with the data preprocessing pipelines and calculating ROC plots.

[`src/`](src/) directory contains scripts for various utility functions, data preprocessing and boilerplate code used for training and evaluation. One of the key features of the data preprocessing pipelines is the way inputs and labels are created. Inputs are 3D-tensors where the leading dimension contains time-steps according to `signal_window_size` and the last two dimensions contain `8x8` BES spatial data comprising the output from the 64 channels of the detector. This can be understood in more detail by the following fantastic figure created by Dr. David Smith.

<p align='center'>
    <img src='assets/signal_window_diagram.png' alt='signal_window' width='1000'/>
</p>

[`options/`](options/) directory contains helper classes for various command line arguments which can be used to change the various parameters ranging from data-preprocessing to model training and inference.

[`model_checkpoints/`](model_checkpoints/) contains the saved models which can be used for inference.

[`archives/`](archives/) directory contains previous code files which are implemented in TensorFlow. It also contains plots generated earlier using PyTorch. It is just for reference and is not in active development.
`archives/model_tools/` is a Python module and the primary set of tools for training ML models. `archives/hpo/` is a directory with python modules and Slurm scripts to perform hyper-parameter optimization with Optuna. `archives/multitrain/` is out-of-date but similar to `hpo/`.  The scripts and python modules in `multi-train/` are intended
to perform multiple training runs for a single set of model parameters, for example, the "best" parameters from HPO.

[`train.py`](train.py) and [`analyze.py`](analyze.py) are the main scripts used for training, and inference respectively. 

## Getting started
There are certain command line arguments which should be passed along with the training script. You can find more help about these by running
```
python train.py --help
```
or 
```
python analyze.py --help
```
### Train the model
Training a model can be easily done by running something like (from the project directory) -
```
 python train.py --input_file labeled-elm-events.hdf5 --device cuda --model_name multi_features --data_preproc unprocessed --signal_window_size 512 --label_look_ahead 500 --normalize_data --n_epochs 5 --max_elms -1 --filename_suffix _dwt_db4_low_lr --raw_num_filters 48 --fft_num_filters 48 --wt_num_filters 48 --dwt_wavelet db4 --dwt_level 9 --lr 0.0003 --weight_decay 0.0025
```

`train.py` script expects the input `.hdf5` file to be stored in the `data/` directory 

### Test the model
Testing can be done similarly. The command line arguments would look like-
```
 python analyze.py --device cuda --model_name multi_features --data_preproc unprocessed --signal_window_size 512 --label_look_ahead 500 --truncate_inputs --normalize_data --n_epochs 20 --max_elms -1 --multi_features --use_fft --plot_data --show_metrics
```

If you just want to run `train.py` and `analyze.py` without saving any output files or plots, you can just add the flag `--dry_run` to either of the scripts.

### Tracking the model and experimentation
All the parameters of interest during training are logged into a pickle file which can be used with [Weights and Biases](https://wandb.ai/site) to track the experiments. More details 
can be found in [`train.py`](train.py) and [`wandb_manual_logs.py`](wandb_manual_logs.py).
