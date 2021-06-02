"""
Package utilities
"""
from pathlib import Path
import numpy as np
import h5py
import tensorflow as tf


# package directories

package_dir = Path(__file__).parent

repo_dir = package_dir.parent

data_dir = repo_dir / "data"
data_dir.mkdir(exist_ok=True)

model_dir = repo_dir / "models"
model_dir.mkdir(exist_ok=True)


# custom learning rate schedule with exponential decay
class Exp_Learning_Rate_Schedule(
    tf.keras.optimizers.schedules.LearningRateSchedule
):
    def __init__(
        self,
        initial_learning_rate=1e-3,
        minimum_learning_rate_factor=10,
        steps_per_halving=50000,  # batches per halving
    ):
        super(Exp_Learning_Rate_Schedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.minimum_learning_rate_factor = minimum_learning_rate_factor
        self.steps_per_halving = steps_per_halving
        self.minimum_learning_rate = (
            self.initial_learning_rate / self.minimum_learning_rate_factor
        )
        print(
            f"Initial learning rate: {self.initial_learning_rate:.4g} (minimum {self.minimum_learning_rate:.4g})"
        )
        print(
            f"Learning rate halves every {self.steps_per_halving} steps (batches)"
        )

    def __call__(self, step):
        decay_factor = tf.math.pow(0.5, step / self.steps_per_halving)
        learning_rate = self.initial_learning_rate * decay_factor
        output_rate = tf.math.maximum(learning_rate, self.minimum_learning_rate)
        return output_rate

    def get_config(self):
        config = {
            "initial_learning_rate": self.initial_learning_rate,
            "minimum_learning_rate_factor": self.minimum_learning_rate_factor,
            "steps_per_halving": self.steps_per_halving,
        }
        return config


# print contents of HDF5 file


def traverse_h5py(input_filename):
    # private function to print attributes, if any
    # groups or datasets may have attributes
    def print_attributes(obj):
        for key, value in obj.attrs.items():
            if isinstance(value, np.ndarray):
                print(f"  Attribute {key}:", value.shape, value.dtype)
            else:
                print(f"  Attribute {key}:", value)

    # private function to recursively print groups/subgroups and datasets
    def recursively_print_info(input_group):
        print(f"Group {input_group.name}")
        print_attributes(input_group)
        # loop over items in a group
        # items may be subgroup or dataset
        # items are key/value pairs
        for key, value in input_group.items():
            if isinstance(value, h5py.Group):
                recursively_print_info(value)
            if isinstance(value, h5py.Dataset):
                print(f"  Dataset {key}:", value.shape, value.dtype)
                print_attributes(value)

    # the file object functions like a group
    # it is the top-level group, known as `root` or `/`
    print(f"Contents of {input_filename}")
    with h5py.File(input_filename, "r") as file:
        # loop over key/value pairs at file root;
        # values may be a group or dataset
        recursively_print_info(file)
