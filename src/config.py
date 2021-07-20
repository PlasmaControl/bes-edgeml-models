# name of the file containing the data
file_name = "labeled-elm-events.hdf5"

# output directory
output_dir = "../outputs"
data_dir = "../data"
model_dir = "../models"

# Autoencoder training and analyze outputs
ae_output_dir = "./outputs"
ae_test_datasets_dir = ae_output_dir + '/test_datasets' 
ae_trained_models_dir = ae_output_dir + '/trained_models'
ae_tensorboard_dir = ae_output_dir + '/tensorboard'
ae_plots_dir = ae_output_dir + '/plots'   

# set universal random state
seed = 15

# set the data mode: balanced or unbalanced
data_mode = "unbalanced"
balance_classes = False

# fraction of whole data to be used as validation and test data
fraction_validate = 0.2
fraction_test = 0.1

# batch size
batch_size = 32 
num_workers = 4

# maximum number of ELM events to analyze
max_elms = 6 #-1

# transition halfwidth (used to smoothen the labels)
transition_halfwidth = 3

# number of time data points to be used for a whole signal
signal_window_size = 32

# size of the input after stitching and resizing
size = 12
stack_elm_events = False

# "look ahead", meaning the label for the entire signal window is taken to
# be label corresponding to the last element (0 ahead) of the signal window;
# (defaults to zero).
label_look_ahead = 0

# Number of k-folds if using k-fold cross validation
folds = 5

# number of training epochs
epochs = 10

# hidden units in fully connected layers
dense_layers = (40, 20)

# dropout fraction
dropout_rate = 0.1

# l2-regularization
l2_factor = 2e-3

# slope of the leaky ReLU
relu_negative_slope = 0.02

# kwargs for cnn model
conv_size = (3,)
cnn_layers = ((4, 8),)

# kwargs for feature model
maxpool_size = 2  # 0 to skip maxpool
filters = 16

# optimization kwargs
learning_rate = .0001
weight_decay = 5e-5
scheduler = "CyclicLR2"
