import numpy as np
import h5py


def traverse_h5py(input_filename):
    # private function to print attributes, if any
    # groups or datasets may have attributes
    def print_attributes(obj):
        for key, value in obj.attrs.items():
            if isinstance(value, np.ndarray):
                print(f'  Attribute {key}:', value.shape, value.dtype)
            else:
                print(f'  Attribute {key}:', value)

    # private function to recursively print groups/subgroups and datasets
    def recursively_print_info(input_group):
        print(f'Group {input_group.name}')
        print_attributes(input_group)
        # loop over items in a group
        # items may be subgroup or dataset
        # items are key/value pairs
        for key, value in input_group.items():
            if isinstance(value, h5py.Group):
                recursively_print_info(value)
            if isinstance(value, h5py.Dataset):
                print(f'  Dataset {key}:', value.shape, value.dtype)
                print_attributes(value)

    # the file object functions like a group
    # it is the top-level group, known as `root` or `/`
    print(f'Contents of {input_filename}')
    with h5py.File(input_filename, 'r') as file:
        # loop over key/value pairs at file root;
        # values may be a group or dataset
        recursively_print_info(file)


if __name__=='__main__':
    traverse_h5py('data/labeled-elm-events-smithdr.hdf5')