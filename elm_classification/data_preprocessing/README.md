# Data Preprocessing
Contents of this directory mostly deal with data preparation from raw BES data 
for training and evaluation. Different model architectures like RNN expect data 
prepared in specific form. Also, different techniques like adding Gaussian noise 
to data, interpolating the spatial dimensions to increase signal's spatial size
are available. Calculating features based on temporal or spatial gradients and 
denoising the signal with wavelet transforms have also been tried.

The general workflow for all the scripts in the data preprocessing phase is 
to inherit the `BaseData` class available in [`base_data.py`](base_data.py) and 
implement all the desired functionality in `_preprocess_data` method.

During execution, the desired data preprocessing technique can be used with the 
`--data_preproc` command line argument. There is also an option to turn on/off the
signal normalization (if it is implemented in the custom script) using 
`--normalize_data` argument. Also, in all the scripts in this directory which are 
derived from `base_data`, there is an option to truncate the input signals 
`--truncate_buffer` time steps after the last time step for the active ELM. This 
is used for ELM onset prediction tasks to reduce the compute time. This can also
be turned on/off using `--truncate_inputs` argument.

The output of these data preprocessing steps produce tuples for training, validation 
and testing that contain signals, labels, sample (or valid) indices, and window start 
indices.
