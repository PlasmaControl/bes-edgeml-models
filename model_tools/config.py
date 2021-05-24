# name of the file containing the data
file_name = "labeled-elm-events.hdf5"

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
