import os
import re
from pathlib import Path

# This script is intended to clean the file names
# to retain only the image index


def rename_files(images_dir_str):
    images_dir = Path(images_dir_str)
    old_file_paths = [x for x in images_dir.iterdir() if x.is_file() and not str(x).endswith('.ini')]
    new_file_paths = get_new_file_paths(old_file_paths)
    for (old_path, new_path) in zip(old_file_paths, new_file_paths):
        rename_file(old_path, new_path)


def get_new_file_paths(old_file_paths):
    new_paths = []
    for path in old_file_paths:
        old_name = path.name
        reg = re.search(r'\((\d+)\)(.[a-zA-Z]+)', old_name)
        new_name = reg.group(1)
        new_extension = reg.group(2)
        new_paths.append(Path(path.parents[0]).joinpath(new_name+new_extension))
    return new_paths


def rename_file(old_file_path, new_file_path):
    os.rename(old_file_path, new_file_path)


rename_files('Data/samples/')