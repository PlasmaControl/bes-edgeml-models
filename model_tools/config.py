# name of the file containing the data
file_name = "labeled-elm-events.hdf5"

# output directory
output_dir = "outputs"
data_dir = "data"

# set universal random state
seed = 15

# fraction of whole data to be used as validation and test data
fraction_validate = 0.2
fraction_test = 0.1

# batch size
batch_size = 4

# maximum number of ELM events to analyze
max_elms = -1

# transition halfwidth (used to smoothen the labels)
transition_halfwidth = 3

# number of time data points to be used for a whole signal
signal_window_size = 8

# "look ahead", meaning the label for the entire signal window is taken to
# be label corresponding to the last element (0 ahead) of the signal window;
# (defaults to zero).
label_look_ahead = 0

# Number of k-folds if using k-fold cross validation
folds = 5

# hidden units in fully connected layers
dense_layers = (40, 20)

# dropout fraction
dropout_rate = 0.1

# l2-regularization
l2_factor = 2e-3

# slope of the leaky ReLU
relu_negative_slope = 0.002

# kwargs for cnn model
conv_size = (3,)
cnn_layers = ((4, 8),)

# kwargs for feature model
maxpool_size = 2  # 0 to skip maxpool
filters = 16

# optimization kwargs
epochs_per_halving = 4
initial_learning_rate = 3e-5
minimum_learning_rate_factor = 30
momentum = 0.2
