import os
import traceback
from os.path import dirname, basename, splitext, join
import pickle

"""
Methods for:
* Searching directories
* Changing file names
* Saving and loading objects with pickle
"""


def apply_to_path(f, path_name, extension):
    for root, dirs, files in os.walk(path_name):
        for file in files:
            if file.endswith(extension):
                f(os.path.join(root, file))


def apply_to_target_path(f, path_name, extension, target_path):
    if not os.path.isdir(target_path):
        os.mkdir(target_path)
    for root, directories, files in os.walk(path_name):
        for file in files:
            if file.endswith(extension):
                path = root.replace(path_name, target_path)
                if not os.path.isdir(path):
                    os.mkdir(path)
                f(os.path.join(root, file), path)


def return_from_path(f, path_name, extension):
    results = []
    for root, dirs, files in os.walk(path_name):
        for file in files:
            if file.endswith(extension):
                try:
                    result = f(os.path.join(root, file))
                    if result is not None:
                        results.append(result)
                except BaseException:
                    print(f'Unable to process {file} in {root}')
                    traceback.print_exc()
    return results


def append_to_file_name(path, new_str, extension):
    dir_name = dirname(path)
    file_name = basename(path)
    split = splitext(file_name)
    if not extension:
        extension = split[1]
    return join(dir_name, split[0] + new_str + extension)


def split_base_and_extension(path):
    split = os.path.splitext(os.path.basename(path))
    return split[0], split[1]


def split_path_base_and_extension(path):
    path, base = os.path.split(path)
    split = os.path.splitext(base)
    return path, split[0], split[1]


def save_object(obj, path):
    with open(path, 'wb') as fout:
        pickle.dump(obj, fout)


def load_object(path):
    with open(path, 'rb') as fin:
        return pickle.load(fin)
