"""Deletes all the saved files depending on the parsed file extension.
Run at your own risk.
"""
print(__doc__)

import os
import glob


def remove_files(path: str, ext: str):
    files_path = os.path.join(path, "*" + ext)
    files = glob.glob(files_path)
    for f in files:
        if f.endswith(ext):
            print(f"Removing file: {f}")
            os.remove(f)
    files = glob.glob(files_path)
    print(f"Files after removing:\n{files}")


if __name__ == "__main__":
    pkl_path = "data/test_data/*"
    pth_path = "model_checkpoints/*"
    file_ext = input("Enter file extension to be deleted:")
    input_path = input(
        "Enter the corresponding base directory path without trailing slash:"
    )
    if file_ext == ".pkl" and input_path == pkl_path[:-2]:
        remove_files(pkl_path, ext=file_ext)
    elif file_ext == ".pth" and input_path == pth_path[:-2]:
        remove_files(pth_path, ext=file_ext)
    else:
        raise ValueError(
            "Path and the file extension are not compatible with each other."
        )
